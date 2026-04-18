#!/usr/bin/env python3
"""
NIDS Cascaded Inference — DAEMON → OSPREY
==========================================
Standalone inference script for the B.Tech Major Project.
Loads a pre-trained model bundle (nids_models.pkl) exported from pipeline.py
and runs a two-stage cascaded pipeline on new CIC-IDS2017 network flow data.

Pipeline Architecture:
    Input → Feature Engineering (73 features)
        ↓
    [Stage 1: DAEMON]  — Binary anomaly detection
        ├── BENIGN  →  🟢 BENIGN (stop)
        └── ANOMALY →  continue ↓
    [Stage 2: OSPREY]  — Multi-class classification + OOD rejection
        ├── Known class  →  🔴 <class name>
        └── OOD rejected →  ⚠ UNKNOWN ATTACK

Supports CSV files (CIC-IDS2017 format) and PCAP/PCAPNG packet captures.
PCAP mode uses CICFlowMeter to aggregate raw packets into bidirectional flows
and maps the extracted features to the CIC-IDS2017 schema.

Usage:
    python infer.py nids_models.pkl input.csv
    python infer.py nids_models.pkl capture.pcap
    python infer.py nids_models.pkl capture.pcapng --top 20
    python infer.py nids_models.pkl input.csv --top 20

Requirements:
    torch, numpy, pandas, scipy, scikit-learn (for RobustScaler unpickling)
    cicflowmeter (optional, required only for PCAP processing)
"""

import sys
import os
import pickle
import zlib
import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.spatial.distance import mahalanobis


# ═══════════════════════════════════════════════════════════════════════════════
#  OSPREY Architecture (inference-only, mirrors pipeline.py definitions)
# ═══════════════════════════════════════════════════════════════════════════════

def _ceil_to_multiple(x: int, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


class SparseTopKActivation(nn.Module):
    def __init__(self, k: float = 0.5):
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        activated = F.relu(x)
        if self.k >= 1.0:
            return activated
        n_keep = max(1, math.ceil(activated.size(-1) * self.k))
        threshold = activated.topk(n_keep, dim=-1).values[..., -1:]
        return activated * (activated >= threshold).float()


class GroupedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_groups: int, bias: bool = True):
        super().__init__()
        self.G = num_groups
        self.g_i = in_features // num_groups
        self.g_o = out_features // num_groups
        self.weight = nn.Parameter(torch.empty(num_groups, self.g_o, self.g_i))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x_g = x.view(B, self.G, self.g_i)
        out = torch.einsum("bgi,goi->bgo", x_g, self.weight).reshape(B, -1)
        if self.bias is not None:
            out = out + self.bias
        return out


class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int,
                 sparsity_k: float, dropout: float):
        super().__init__()
        neck = _ceil_to_multiple(max(dim // 2, num_groups), num_groups)
        self.fc1 = GroupedLinear(dim, neck, num_groups)
        self.bn1 = nn.BatchNorm1d(neck)
        self.act1 = SparseTopKActivation(sparsity_k)
        self.fc2 = GroupedLinear(neck, dim, num_groups)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = SparseTopKActivation(sparsity_k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act1(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.act2(out + x)


class SparseGroupedEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims, embed_dim,
                 num_groups=4, sparsity_k=0.5, dropout=0.1):
        super().__init__()
        dims = list(hidden_dims)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, dims[0], bias=False),
            nn.BatchNorm1d(dims[0]),
            SparseTopKActivation(sparsity_k),
        )
        self.stages = nn.ModuleList()
        for i, d in enumerate(dims):
            stage_layers = []
            if i > 0:
                prev = dims[i - 1]
                stage_layers += [
                    GroupedLinear(prev, d, num_groups),
                    nn.BatchNorm1d(d),
                    SparseTopKActivation(sparsity_k),
                    nn.Dropout(dropout),
                ]
            stage_layers.append(
                BottleneckResidualBlock(d, num_groups, sparsity_k, dropout)
            )
            self.stages.append(nn.Sequential(*stage_layers))
        self.embed_proj = nn.Linear(dims[-1], embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for stage in self.stages:
            h = stage(h)
        return l2_normalize(self.embed_proj(h), dim=-1)


class AdaptivePrototypeMemory(nn.Module):
    def __init__(self, num_known_classes, embed_dim,
                 proto_momentum=0.999, init_temperature=0.07,
                 learn_temperature=True):
        super().__init__()
        C, d = num_known_classes, embed_dim
        self.C = C
        protos = l2_normalize(torch.randn(C, d), dim=-1)
        self.prototypes = nn.Parameter(protos)
        log_tau = torch.full((C,), math.log(init_temperature))
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_temperature", log_tau)
        self.register_buffer("proto_shadow", protos.clone())

    @property
    def normalized(self) -> Tensor:
        return l2_normalize(self.prototypes, dim=-1)

    def forward(self, z: Tensor) -> Tensor:
        P = self.normalized
        sim = z @ P.T
        tau = self.log_temperature.exp().clamp(min=0.04, max=0.5)
        return sim / tau


class EnergyGate(nn.Module):
    def __init__(self, energy_temp=1.0, ood_margin=25.0):
        super().__init__()
        self.T = energy_temp
        self.register_buffer(
            "threshold",
            torch.tensor(-ood_margin / 2.0, dtype=torch.float32),
        )

    def forward(self, logits: Tensor) -> Tensor:
        return -self.T * torch.logsumexp(logits / self.T, dim=-1)


class OSPREY(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = SparseGroupedEncoder(
            in_features=config['in_features'],
            hidden_dims=config['hidden_dims'],
            embed_dim=config['embed_dim'],
            num_groups=config['num_groups'],
            sparsity_k=0.5,
            dropout=config.get('dropout', 0.1),
        )
        self.protos = AdaptivePrototypeMemory(
            num_known_classes=config['num_known_classes'],
            embed_dim=config['embed_dim'],
            proto_momentum=config.get('proto_momentum', 0.999),
        )
        self.energy = EnergyGate(
            energy_temp=1.0,
            ood_margin=config.get('ood_margin', 25.0),
        )

    def forward(self, x: Tensor):
        z = self.encoder(x)
        logits = self.protos(z)
        energy = self.energy(logits)
        return logits, z, energy


# ═══════════════════════════════════════════════════════════════════════════════
#  DAEMON Architecture (inference-only, mirrors pipeline.py DualPathAutoencoder)
# ═══════════════════════════════════════════════════════════════════════════════

class DualPathAutoencoder(nn.Module):
    def __init__(self, feature_info: dict,
                 bottleneck_dim: int = 8, dropout: float = 0.05):
        super().__init__()
        flow_dim = feature_info['flow_dim']
        behav_dim = feature_info['behav_dim']
        meta_dim = feature_info['meta_dim']
        flags_dim = feature_info['flags_dim']
        total_dim = feature_info['total_dim']

        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 48), nn.BatchNorm1d(48), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(48, 24), nn.BatchNorm1d(24), nn.SiLU(),
        )
        self.behav_encoder = nn.Sequential(
            nn.Linear(behav_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(),
        )
        self.flags_encoder = nn.Sequential(
            nn.Linear(flags_dim, 24), nn.BatchNorm1d(24), nn.SiLU(),
            nn.Linear(24, 12), nn.BatchNorm1d(12), nn.SiLU(),
        )

        _concat_dim = 24 + 32 + 32 + 12
        self.bottleneck = nn.Sequential(
            nn.Linear(_concat_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, _concat_dim), nn.BatchNorm1d(_concat_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(_concat_dim, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 96), nn.BatchNorm1d(96), nn.SiLU(),
            nn.Linear(96, total_dim),
        )

        self._flow_idx = feature_info['flow_idx']
        self._behav_idx = feature_info['behav_idx']
        self._meta_idx = feature_info['meta_idx']
        self._flags_idx = feature_info['flags_idx']

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h_flow = self.flow_encoder(x[:, self._flow_idx])
        h_behav = self.behav_encoder(x[:, self._behav_idx])
        h_meta = self.meta_encoder(x[:, self._meta_idx])
        h_flags = self.flags_encoder(x[:, self._flags_idx])
        z = self.bottleneck(torch.cat([h_flow, h_behav, h_meta, h_flags], dim=1))
        return self.decoder(z), z


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Engineering (mirrors pipeline.py §2.7)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame,
                      log_features: list,
                      ratio_cols: list) -> pd.DataFrame:
    """Apply the same feature engineering as pipeline.py §2.7."""
    df = df.copy()

    # 1. Sanitise numeric cols
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in ['fwd_header_length', 'bwd_header_length', 'min_seg_size_forward']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # 2. Log1p transforms
    for col in log_features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0)).astype(np.float32)

    # 3. Engineer 8 ratio features
    eps = 1e-6
    df['fwd_bwd_pkt_ratio'] = (
        df['total_fwd_packets'] / (df['total_backward_packets'] + eps))
    df['fwd_bwd_bytes_ratio'] = (
        df['total_length_of_fwd_packets'] / (df['total_length_of_bwd_packets'] + eps))
    df['fwd_bwd_pktlen_ratio'] = (
        df['fwd_packet_length_mean'] / (df['bwd_packet_length_mean'] + eps))
    df['fwd_bwd_iat_ratio'] = (
        df['fwd_iat_mean'] / (df['bwd_iat_mean'] + eps))
    df['pkt_per_sec'] = (
        (df['total_fwd_packets'] + df['total_backward_packets'])
        / (df['flow_duration'] + eps))
    df['bytes_per_pkt'] = (
        (df['total_length_of_fwd_packets'] + df['total_length_of_bwd_packets'])
        / (df['total_fwd_packets'] + df['total_backward_packets'] + eps))
    df['flag_density'] = (
        (df['fin_flag_count'] + df['rst_flag_count']
         + df['psh_flag_count'] + df['urg_flag_count'])
        / (df['total_fwd_packets'] + df['total_backward_packets'] + eps))
    df['win_ratio'] = (
        df['init_win_bytes_forward'] / (df['init_win_bytes_backward'] + eps))

    # 4. Log1p-compress ratio features
    for col in ratio_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0, upper=1e6)).astype(np.float32)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PCAP → CIC-IDS2017 Flow Extraction (via realtime_cicflow.py engine)
# ═══════════════════════════════════════════════════════════════════════════════

_PCAP_EXTENSIONS = ('.pcap', '.pcapng', '.cap', '.pcap.gz')


def pcap_to_dataframe(source: str,
                      idle_timeout: int = 15,
                      active_timeout: int = 120):
    """Extract CIC-IDS2017-compatible flow features from a PCAP file.

    Uses our faithful CICFlowMeter replica (realtime_cicflow.py) to aggregate
    raw packets into bidirectional flows. Produces ALL native CIC-IDS2017
    features including init_win_bytes and active/idle times.

    Returns:
        features_df:  DataFrame with 65 CIC-IDS2017 base columns
        metadata_df:  DataFrame with flow metadata (IPs, ports, protocol)
    """
    try:
        from realtime_cicflow import (
            FlowRecord, PacketInfo, extract_features, _make_flow_key,
        )
        from scapy.all import PcapReader, IP, TCP, UDP
    except ImportError as e:
        print('\n' + '=' * 72)
        print('ERROR: realtime_cicflow.py + scapy required for PCAP processing.')
        print(f'Import error: {e}')
        print('Ensure realtime_cicflow.py is in the same directory.')
        print('Install scapy: pip install scapy')
        print('=' * 72)
        sys.exit(1)

    print(f'  Extracting flows with CICFlowMeter replica...')
    print(f'    source = {source}')
    print(f'    idle_timeout  = {idle_timeout}s')

    flows = {}
    finished_features = []

    def _export_flow(flow):
        feat_df = extract_features(flow)
        if feat_df is not None:
            meta = {
                'src_ip': flow.src_ip, 'dst_ip': flow.dst_ip,
                'src_port': flow.src_port, 'dst_port': flow.dst_port,
                'protocol': flow.proto,
            }
            finished_features.append((feat_df, meta))

    try:
        reader = PcapReader(source)
        pkt_count = 0
        for packet in reader:
            pkt_count += 1
            if not packet.haslayer(IP):
                continue

            ip = packet[IP]
            now = float(packet.time)
            proto = ip.proto

            if proto == 6 and packet.haslayer(TCP):
                tcp = packet[TCP]
                src_port, dst_port = tcp.sport, tcp.dport
                flags = int(tcp.flags)
                win_size = tcp.window
                header_len = (ip.ihl * 4) + (tcp.dataofs * 4)
                payload_len = len(bytes(tcp.payload))
                pkt_len = len(packet[IP])
            elif proto == 17 and packet.haslayer(UDP):
                udp = packet[UDP]
                src_port, dst_port = udp.sport, udp.dport
                flags, win_size = 0, 0
                header_len = (ip.ihl * 4) + 8
                payload_len = len(bytes(udp.payload))
                pkt_len = len(packet[IP])
            else:
                continue

            src_ip, dst_ip = ip.src, ip.dst
            fwd_key = _make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
            bwd_key = _make_flow_key(dst_ip, src_ip, dst_port, src_port, proto)

            if fwd_key in flows:
                key, direction = fwd_key, 0
                flow = flows[key]
            elif bwd_key in flows:
                key, direction = bwd_key, 1
                flow = flows[key]
            else:
                key, direction = fwd_key, 0
                flow = FlowRecord(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    proto=proto, start_time=now, last_seen=now,
                )
                flows[key] = flow

            if (now - flow.last_seen) > idle_timeout:
                _export_flow(flow)
                flows[key] = FlowRecord(
                    src_ip=flow.src_ip, dst_ip=flow.dst_ip,
                    src_port=flow.src_port, dst_port=flow.dst_port,
                    proto=flow.proto, start_time=now, last_seen=now,
                )
                flow = flows[key]

            pinfo = PacketInfo(
                timestamp=now, ip_length=pkt_len,
                header_length=header_len,
                direction=direction, tcp_flags=flags,
                tcp_window=win_size, payload_len=payload_len,
            )
            flow.add_packet(pinfo)

        reader.close()

        for flow in flows.values():
            _export_flow(flow)

        print(f'  Processed {pkt_count:,} packets, {len(finished_features):,} flows')

    except Exception as e:
        print(f'  ERROR reading PCAP: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    if len(finished_features) == 0:
        print('  WARNING: No flows extracted from PCAP file.')
        return pd.DataFrame(), pd.DataFrame()

    all_features = pd.concat([f[0] for f in finished_features], ignore_index=True)
    all_metadata = pd.DataFrame([f[1] for f in finished_features])

    n_tcp = int((all_metadata.get('protocol', pd.Series()) == 6).sum())
    n_udp = int((all_metadata.get('protocol', pd.Series()) == 17).sum())
    n_other = len(all_features) - n_tcp - n_udp
    print(f'  Extracted {len(all_features):,} flows  '
          f'(TCP: {n_tcp:,}  UDP: {n_udp:,}  Other: {n_other:,})')

    return all_features, all_metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Engine — Cascaded Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def load_bundle(path: str) -> dict:
    """Load the compressed model bundle."""
    with open(path, 'rb') as f:
        compressed = f.read()
    return pickle.loads(zlib.decompress(compressed))


# Safety margin: borderline samples (score within this fraction of threshold)
# are forwarded to Stage 2 for a second opinion from OSPREY.
_SAFETY_MARGIN = 0.70


def run_daemon(model, X_np, feature_info, scorer_stats, weights, threshold):
    """Stage 1: DAEMON binary anomaly detection.

    Returns per-sample dict with:
      - daemon_verdict: 'ATTACK', 'BORDERLINE', or 'BENIGN'
      - daemon_score: float composite anomaly score

    Samples above threshold → ATTACK (definite anomaly).
    Samples between threshold * _SAFETY_MARGIN and threshold → BORDERLINE
    (sent to Stage 2 for a second opinion to catch DAEMON false negatives).
    Samples below threshold * _SAFETY_MARGIN → BENIGN (clearly normal).
    """
    model.eval()
    fi = feature_info['flow_idx']
    bi = feature_info['behav_idx']
    mi = feature_info['meta_idx']

    z_mean = np.array(scorer_stats['z_mean'])
    cov_inv = np.array(scorer_stats['cov_inv'])

    with torch.no_grad():
        X_t = torch.from_numpy(X_np).float()
        xr, z = model(X_t)
        err = (X_t - xr) ** 2
        mse = err.mean(dim=1).numpy()
        gmax = torch.stack([
            err[:, fi].mean(dim=1),
            err[:, bi].mean(dim=1),
            err[:, mi].mean(dim=1),
        ], dim=1).max(dim=1)[0].numpy()
        z_np = z.numpy()

    maha = np.array([mahalanobis(l, z_mean, cov_inv) for l in z_np])

    mse_z = (mse - scorer_stats['mse_mean']) / (scorer_stats['mse_std'] + 1e-8)
    gmax_z = (gmax - scorer_stats['gmax_mean']) / (scorer_stats['gmax_std'] + 1e-8)
    maha_z = (maha - scorer_stats['maha_mean']) / (scorer_stats['maha_std'] + 1e-8)

    composite = (weights['alpha'] * mse_z
                 + weights['gamma'] * gmax_z
                 + weights['beta'] * maha_z)

    borderline_threshold = threshold * _SAFETY_MARGIN

    results = []
    for i in range(len(X_np)):
        score = composite[i]
        if score > threshold:
            verdict = 'ATTACK'
        elif score > borderline_threshold:
            verdict = 'BORDERLINE'
        else:
            verdict = 'BENIGN'
        results.append({
            'daemon_verdict': verdict,
            'daemon_score': float(score),
        })
    return results


def run_osprey(model, X_np, thresholds, label_encoder):
    """Stage 2: OSPREY multi-class classification + OOD rejection.

    Runs only on samples that passed Stage 1 (DAEMON flagged as anomaly).
    Returns per-sample dict with classification or OOD rejection.
    """
    model.eval()
    results = []
    with torch.no_grad():
        X_t = torch.from_numpy(X_np).float()
        logits, embeddings, energy = model(X_t)
        proba = F.softmax(logits, dim=-1).numpy()
        energy_np = energy.numpy()
        preds = logits.argmax(dim=-1).numpy()

        P_norm = l2_normalize(model.protos.prototypes, dim=-1)
        cos_sim = (embeddings @ P_norm.T).max(dim=-1).values.numpy()

        entropy = -np.sum(np.clip(proba, 1e-10, 1.0)
                          * np.log(np.clip(proba, 1e-10, 1.0)), axis=1)

        for i in range(len(X_np)):
            # Triple-gate OOD: majority vote (2+ gates = OOD)
            ood_count = (int(energy_np[i] > thresholds['energy']) +
                         int(entropy[i] > thresholds['entropy']) +
                         int(cos_sim[i] < thresholds['cosine']))

            predicted_class = label_encoder.inverse_transform([preds[i]])[0]
            is_ood = ood_count >= 2

            if is_ood:
                verdict = 'UNKNOWN'
                class_name = 'UNKNOWN'
            else:
                verdict = 'KNOWN'
                class_name = predicted_class

            results.append({
                'osprey_verdict': verdict,
                'osprey_class': class_name,
                'osprey_predicted_class': predicted_class,
                'osprey_energy': float(energy_np[i]),
                'osprey_entropy': float(entropy[i]),
                'osprey_max_cos': float(cos_sim[i]),
                'osprey_ood_count': ood_count,
            })
    return results


def _empty_osprey_fields():
    """Default OSPREY fields for samples that never reached Stage 2."""
    return {
        'osprey_verdict': '—',
        'osprey_class': '—',
        'osprey_predicted_class': '—',
        'osprey_energy': float('nan'),
        'osprey_entropy': float('nan'),
        'osprey_max_cos': float('nan'),
        'osprey_ood_count': 0,
    }


def cascade_verdict(daemon_results, osprey_results_map):
    """Assemble final cascade verdicts.

    Decision logic:
      DAEMON ATTACK    → OSPREY classifies → known class or UNKNOWN ATTACK
      DAEMON BORDERLINE → OSPREY second opinion:
        - OSPREY says known attack → override to that attack (DAEMON was wrong)
        - OSPREY says OOD/unknown  → keep BENIGN (insufficient evidence)
      DAEMON BENIGN     → 🟢 BENIGN (pipeline stops, Stage 2 never ran)

    Args:
        daemon_results: list of dicts for ALL samples (Stage 1 output)
        osprey_results_map: dict mapping sample index → OSPREY result dict
                            (only for ATTACK + BORDERLINE samples)

    Returns:
        list of dicts with final cascade verdict for every sample
    """
    fused = []
    for i, dmn in enumerate(daemon_results):
        row = {**dmn}

        if dmn['daemon_verdict'] == 'BENIGN':
            # ── Clearly benign (below safety margin) → stop ──────────
            row['cascade_stage'] = 1
            row['cascade_class'] = 'BENIGN'
            row['cascade_verdict'] = '🟢 BENIGN'
            row.update(_empty_osprey_fields())

        elif dmn['daemon_verdict'] == 'ATTACK':
            # ── Definite anomaly → OSPREY classifies ─────────────────
            osp = osprey_results_map[i]
            row.update(osp)
            row['cascade_stage'] = 2

            if osp['osprey_verdict'] == 'UNKNOWN':
                row['cascade_class'] = 'UNKNOWN'
                row['cascade_verdict'] = '⚠ UNKNOWN ATTACK'
            else:
                row['cascade_class'] = osp['osprey_class']
                row['cascade_verdict'] = f'🔴 {osp["osprey_class"]}'

        elif dmn['daemon_verdict'] == 'BORDERLINE':
            # ── Safety margin: OSPREY gets a second opinion ──────────
            osp = osprey_results_map[i]
            row.update(osp)
            row['cascade_stage'] = 2

            if osp['osprey_verdict'] == 'KNOWN':
                # OSPREY confidently identifies an attack class
                # → override DAEMON's borderline-BENIGN decision
                row['cascade_class'] = osp['osprey_class']
                row['cascade_verdict'] = f'🟠 {osp["osprey_class"]} (safety override)'
            else:
                # OSPREY is also uncertain (OOD) → not enough evidence
                # → keep as BENIGN
                row['cascade_class'] = 'BENIGN'
                row['cascade_verdict'] = '🟢 BENIGN'

        fused.append(row)
    return fused


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='NIDS Cascaded Inference — DAEMON → OSPREY',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python infer.py nids_models.pkl traffic.csv\n'
               '  python infer.py nids_models.pkl capture.pcap\n'
               '  python infer.py nids_models.pkl dump.pcapng --top 20\n'
               '  python infer.py nids_models.pkl capture.pcap '
               '--idle-timeout 30\n')
    parser.add_argument('model_path', help='Path to nids_models.pkl')
    parser.add_argument('input_source',
                        help='Input: CSV file (.csv) or PCAP file '
                             '(.pcap / .pcapng / .cap)')
    parser.add_argument('--top', type=int, default=None,
                        help='Show only top N rows (default: all)')
    parser.add_argument('--idle-timeout', type=int, default=15,
                        help='Flow idle timeout in seconds '
                             '(PCAP mode, default: 15)')
    parser.add_argument('--active-timeout', type=int, default=120,
                        help='Flow active timeout in seconds '
                             '(PCAP mode, default: 120)')
    args = parser.parse_args()

    # ── Load bundle ──────────────────────────────────────────────────────
    print(f'Loading model bundle: {args.model_path}')
    bundle = load_bundle(args.model_path)

    # ── Load input data ──────────────────────────────────────────────────
    input_path = args.input_source
    is_pcap = input_path.lower().endswith(_PCAP_EXTENSIONS)
    flow_metadata = None

    if is_pcap:
        print(f'Reading PCAP: {input_path}')
        raw_df, flow_metadata = pcap_to_dataframe(
            input_path,
            idle_timeout=args.idle_timeout,
            active_timeout=args.active_timeout,
        )
        if len(raw_df) == 0:
            print('No flows to analyse. Exiting.')
            sys.exit(0)
    else:
        print(f'Reading CSV: {input_path}')
        raw_df = pd.read_csv(input_path, low_memory=False)
        # Drop label columns if present
        for col in ['label', 'attack', 'attack_type']:
            if col in raw_df.columns:
                raw_df.drop(columns=[col], inplace=True)

    n_rows = len(raw_df)
    print(f'  {n_rows:,} rows, {len(raw_df.columns)} columns')

    # ── Normalize column names ───────────────────────────────────────────
    raw_df.columns = (
        raw_df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('/', '_', regex=False)
    )

    # ── Feature engineering ──────────────────────────────────────────────
    print('Applying feature engineering...')
    eng_df = engineer_features(
        raw_df,
        bundle['log_transform_features'],
        bundle['ratio_cols'],
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1: DAEMON — Binary Anomaly Detection
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '━' * 72)
    print('  STAGE 1: DAEMON — Binary Anomaly Detection')
    print('━' * 72)

    daemon_features = bundle['daemon_feature_info']
    dc = daemon_features['all_features']

    # Fill missing columns and scale
    for col in dc:
        if col not in eng_df.columns:
            eng_df[col] = 0.0
    daemon_X = np.clip(
        bundle['daemon_scaler'].transform(eng_df[dc].values.astype(np.float32)),
        -5, 5,
    ).astype(np.float32)

    daemon_model = DualPathAutoencoder(
        bundle['daemon_feature_info'],
        bundle['daemon_bottleneck_dim'],
        bundle['daemon_dropout'],
    )
    daemon_model.load_state_dict(bundle['daemon_state_dict'])
    daemon_model.eval()
    print(f'  Model loaded: {sum(p.numel() for p in daemon_model.parameters()):,} parameters')
    print(f'  Threshold: {bundle["daemon_threshold"]:.4f}')

    daemon_results = run_daemon(
        daemon_model, daemon_X,
        bundle['daemon_feature_info'],
        bundle['daemon_scorer_stats'],
        bundle['daemon_composite_weights'],
        bundle['daemon_threshold'],
    )

    # Determine which samples proceed to Stage 2
    # Both ATTACK and BORDERLINE samples are forwarded to OSPREY
    attack_indices = [i for i, r in enumerate(daemon_results)
                      if r['daemon_verdict'] == 'ATTACK']
    borderline_indices = [i for i, r in enumerate(daemon_results)
                          if r['daemon_verdict'] == 'BORDERLINE']
    anomaly_indices = attack_indices + borderline_indices
    benign_count = n_rows - len(anomaly_indices)

    print(f'\n  Results:')
    print(f'    🟢 BENIGN:     {benign_count:,} ({100*benign_count/n_rows:.1f}%) — pipeline stops')
    print(f'    🔴 ANOMALY:    {len(attack_indices):,} ({100*len(attack_indices)/n_rows:.1f}%) — forwarded to Stage 2')
    print(f'    🟡 BORDERLINE: {len(borderline_indices):,} ({100*len(borderline_indices)/n_rows:.1f}%) — safety check via Stage 2')

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 2: OSPREY — Multi-Class Classification + OOD Rejection
    # ══════════════════════════════════════════════════════════════════════
    osprey_results_map = {}

    if len(anomaly_indices) > 0:
        print('\n' + '━' * 72)
        print('  STAGE 2: OSPREY — Multi-Class + Zero-Day Rejection')
        print('━' * 72)

        osprey_features = bundle['osprey_feature_names']
        osprey_expected_cols = bundle['osprey_scaler'].feature_names_in_

        # Ensure exact column presence and order for sklearn validation
        for col in osprey_expected_cols:
            if col not in eng_df.columns:
                eng_df[col] = 0.0

        # Scale and prune — only for anomaly subset
        osprey_X_all = eng_df.iloc[anomaly_indices][osprey_expected_cols].copy()

        osprey_X_scaled = pd.DataFrame(
            bundle['osprey_scaler'].transform(osprey_X_all),
            columns=osprey_expected_cols, index=osprey_X_all.index,
        ).astype(np.float32)

        osprey_X = osprey_X_scaled[osprey_features].values

        osprey_model = OSPREY(bundle['osprey_config'])
        osprey_model.load_state_dict(bundle['osprey_state_dict'])
        osprey_model.eval()
        print(f'  Model loaded: {sum(p.numel() for p in osprey_model.parameters()):,} parameters')
        print(f'  Features: {len(osprey_features)} mRMR-selected')
        thr = bundle['osprey_thresholds']
        print(f'  OOD thresholds — E: {thr["energy"]:.4f}  '
              f'H: {thr["entropy"]:.4f}  cos: {thr["cosine"]:.4f}')

        osprey_results = run_osprey(
            osprey_model, osprey_X,
            bundle['osprey_thresholds'],
            bundle['osprey_label_encoder'],
        )

        # Map Stage 2 results back to original sample indices
        for orig_idx, osp_result in zip(anomaly_indices, osprey_results):
            osprey_results_map[orig_idx] = osp_result

        n_known = sum(1 for r in osprey_results if r['osprey_verdict'] == 'KNOWN')
        n_unknown = sum(1 for r in osprey_results if r['osprey_verdict'] == 'UNKNOWN')
        print(f'\n  Results:')
        print(f'    🔴 Known attack:   {n_known:,} — classified')
        print(f'    ⚠  Unknown attack: {n_unknown:,} — OOD rejected')
    else:
        print('\n  Stage 2 skipped — no anomalies detected by Stage 1.')

    # ══════════════════════════════════════════════════════════════════════
    #  Cascade Verdict Assembly
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '━' * 72)
    print('  CASCADE VERDICT ASSEMBLY')
    print('━' * 72)

    fused = cascade_verdict(daemon_results, osprey_results_map)
    results_df = pd.DataFrame(fused)
    results_df.index.name = 'sample'

    # ── Summary ──────────────────────────────────────────────────────────
    n_total = len(results_df)
    n_benign = (results_df['cascade_class'] == 'BENIGN').sum()
    n_attack = (results_df['cascade_class'] != 'BENIGN').sum()
    n_unknown = (results_df['cascade_class'] == 'UNKNOWN').sum()
    n_known_attack = n_attack - n_unknown

    print(f'\n{"=" * 72}')
    print('CASCADED INFERENCE RESULTS')
    print(f'{"=" * 72}')
    print(f'  Total samples:       {n_total:,}')
    print(f'  🟢 Benign:           {n_benign:,} ({100*n_benign/n_total:.1f}%)')
    print(f'  🔴 Known attacks:    {n_known_attack:,} ({100*n_known_attack/n_total:.1f}%)')
    print(f'  ⚠  Unknown attacks:  {n_unknown:,} ({100*n_unknown/n_total:.1f}%)')
    print(f'\n  Stage 1 pass-through: {len(anomaly_indices):,}/{n_total:,} '
          f'({100*len(anomaly_indices)/n_total:.1f}%) → Stage 2')

    print(f'\nVerdict Distribution:')
    for verdict, count in results_df['cascade_verdict'].value_counts().items():
        print(f'  {verdict}: {count:,}')
    print(f'{"=" * 72}')

    # ── Per-sample table ─────────────────────────────────────────────────
    display_cols = ['cascade_stage', 'daemon_verdict', 'daemon_score',
                    'osprey_class', 'cascade_verdict']
    display_df = results_df[display_cols].copy()
    display_df['daemon_score'] = display_df['daemon_score'].map('{:.4f}'.format)

    if args.top:
        print(f'\nShowing top {args.top} rows:')
        print(display_df.head(args.top).to_string())
    else:
        print(f'\nAll {n_total} rows:')
        print(display_df.to_string())

    # ── Save results ─────────────────────────────────────────────────────
    # Prepend flow metadata (IPs, ports) for PCAP results
    if flow_metadata is not None and len(flow_metadata) == len(results_df):
        for mc in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']:
            if mc in flow_metadata.columns:
                results_df.insert(
                    results_df.columns.get_loc('daemon_verdict'),
                    mc, flow_metadata[mc].values)

    base, ext = os.path.splitext(input_path)
    out_path = f'{base}_results.csv'
    results_df.to_csv(out_path, index=True)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
