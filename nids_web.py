#!/usr/bin/env python3
"""
OSPREY × DAEMON NIDS — Web Dashboard
=====================================
Real-time network intrusion detection with a live web dashboard.
Cascaded pipeline: DAEMON (binary anomaly) → OSPREY (multi-class + OOD).

Usage:
    sudo python3 nids_web.py --iface ens33
    # Open http://<host-ip>:5000 in browser

Dependencies:
    pip install flask
"""
from __future__ import annotations

import os, sys, time, re, argparse, traceback, csv, json, threading, queue
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# ── Import inference engine from infer.py ─────────────────────
from infer import (
    OSPREY, DualPathAutoencoder,
    load_bundle, engineer_features, pcap_to_dataframe,
    run_osprey, run_daemon, cascade_verdict, _empty_osprey_fields,
    l2_normalize, _PCAP_EXTENSIONS,
)

# ── Import post-classification modules ────────────────────────
try:
    from ddos_aggregator import DDoSAggregator
    _HAS_POSTCLASS = True
except ImportError:
    _HAS_POSTCLASS = False

try:
    from flask import Flask, jsonify, request, Response
except ImportError:
    print("ERROR: Flask is required. Install with: pip install flask")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════
_PROTO = {1: "ICMP", 6: "TCP", 17: "UDP"}


def proto_name(n: int) -> str:
    return _PROTO.get(int(n), str(n))


def get_default_interface() -> str:
    try:
        ifaces = sorted(os.listdir("/sys/class/net/"))
        for iface in ifaces:
            if iface != "lo":
                return iface
        return ifaces[0] if ifaces else "eth0"
    except FileNotFoundError:
        return "eth0"


def get_all_interfaces() -> list[str]:
    try:
        return sorted(os.listdir("/sys/class/net/"))
    except FileNotFoundError:
        return ["eth0"]


def get_local_ips() -> set[str]:
    local = {"127.0.0.1", "::1", "0.0.0.0"}
    try:
        import socket
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            local.add(info[4][0])
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(["ip", "-o", "addr", "show"], text=True,
                                       stderr=subprocess.DEVNULL)
        for line in out.strip().splitlines():
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "inet" or p == "inet6":
                    local.add(parts[i + 1].split("/")[0])
    except Exception:
        pass
    return local


def _find_model_bundle() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "nids_models.pkl")
    if os.path.isfile(default_path):
        return default_path
    cwd_path = os.path.join(os.getcwd(), "nids_models.pkl")
    if os.path.isfile(cwd_path):
        return cwd_path
    return default_path


# ═══════════════════════════════════════════════════════════════
#  Inference Engine — cascaded DAEMON → OSPREY pipeline
# ═══════════════════════════════════════════════════════════════
class InferenceEngine:
    """Loads model bundle and runs cascaded DAEMON→OSPREY inference."""

    def __init__(self, bundle_path: str):
        self.bundle = load_bundle(bundle_path)
        b = self.bundle

        self.daemon_model = DualPathAutoencoder(
            b["daemon_feature_info"], b["daemon_bottleneck_dim"], b["daemon_dropout"],
        )
        self.daemon_model.load_state_dict(b["daemon_state_dict"])
        self.daemon_model.eval()

        self.osprey_model = OSPREY(b["osprey_config"])
        self.osprey_model.load_state_dict(b["osprey_state_dict"])
        self.osprey_model.eval()

        self.daemon_params = sum(p.numel() for p in self.daemon_model.parameters())
        self.osprey_params = sum(p.numel() for p in self.osprey_model.parameters())

        _DAEMON_OVERRIDE = 8.0
        self._original_tau = b["daemon_threshold"]
        b["daemon_threshold"] = _DAEMON_OVERRIDE

        if _HAS_POSTCLASS:
            self.ddos_aggregator = DDoSAggregator(
                window_sec=120, min_sources=3,
                excluded_ports={53},
            )
        else:
            self.ddos_aggregator = None

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
            .str.replace('/', '_', regex=False)
        )
        return df

    @staticmethod
    def _align_to_scaler(df: pd.DataFrame, scaler) -> pd.DataFrame:
        if hasattr(scaler, 'feature_names_in_'):
            expected = list(scaler.feature_names_in_)
        else:
            return df
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0
        return df[expected]

    def process(self, raw_df: pd.DataFrame) -> list[dict]:
        b = self.bundle
        raw_df = self._normalize_columns(raw_df)
        eng_df = engineer_features(raw_df, b["log_transform_features"], b["ratio_cols"])

        # Stage 1: DAEMON
        dc = b["daemon_feature_info"]["all_features"]
        for col in dc:
            if col not in eng_df.columns:
                eng_df[col] = 0.0
        dX = np.clip(
            b["daemon_scaler"].transform(eng_df[dc].values.astype(np.float32)),
            -5, 5,
        ).astype(np.float32)

        d_res = run_daemon(
            self.daemon_model, dX, b["daemon_feature_info"],
            b["daemon_scorer_stats"], b["daemon_composite_weights"], b["daemon_threshold"],
        )

        # Stage 2: OSPREY — only on ATTACK + BORDERLINE
        anomaly_indices = [i for i, r in enumerate(d_res)
                          if r['daemon_verdict'] in ('ATTACK', 'BORDERLINE')]

        osprey_results_map = {}
        if anomaly_indices:
            # ── Minimum-packet gate ──────────────────────────────
            # Flows with <5 fwd packets produce unreliable OSPREY
            # classifications (coin-flip at ≤3 fwd_pkts per R27 data).
            # Force UNKNOWN for these — DAEMON already flagged the attack.
            _MIN_PKTS = 3  # R29-tuned: was 5, lowered to preserve DoS accuracy
            _UNKNOWN_SYNTHETIC = {
                'osprey_verdict': 'UNKNOWN',
                'osprey_class': 'UNKNOWN',
                'osprey_predicted_class': 'UNKNOWN',
                'osprey_energy': float('nan'),
                'osprey_entropy': float('nan'),
                'osprey_max_cos': float('nan'),
                'osprey_ood_count': 3,
            }

            # Split: indices with enough packets vs too few
            osprey_eligible = []
            for idx in anomaly_indices:
                fwd_pkts = 0
                if 'total_fwd_packets' in raw_df.columns:
                    try:
                        fwd_pkts = int(raw_df.iloc[idx].get('total_fwd_packets', 0))
                    except (ValueError, TypeError):
                        fwd_pkts = 0
                if fwd_pkts < _MIN_PKTS:
                    osprey_results_map[idx] = dict(_UNKNOWN_SYNTHETIC)
                else:
                    osprey_eligible.append(idx)

            # Run OSPREY only on eligible flows
            if osprey_eligible:
                of = b["osprey_feature_names"]
                oX_df = self._align_to_scaler(eng_df, b["osprey_scaler"])
                oX_scaled = pd.DataFrame(
                    b["osprey_scaler"].transform(oX_df),
                    columns=oX_df.columns, index=oX_df.index,
                ).astype(np.float32)

                for col in of:
                    if col not in oX_scaled.columns:
                        oX_scaled[col] = 0.0

                oX_anomaly = oX_scaled.iloc[osprey_eligible][of].values

                o_res = run_osprey(
                    self.osprey_model, oX_anomaly,
                    b["osprey_thresholds"], b["osprey_label_encoder"],
                )

                for orig_idx, osp_result in zip(osprey_eligible, o_res):
                    osprey_results_map[orig_idx] = osp_result

        results = cascade_verdict(d_res, osprey_results_map)

        # Post-classification: DDoS aggregator only
        if self.ddos_aggregator:
            for i, r in enumerate(results):
                src_ip = ""
                dst_ip = ""
                dst_port = 0

                if 'src_ip' in raw_df.columns:
                    src_ip = str(raw_df.iloc[i].get('src_ip', ''))
                if 'dst_ip' in raw_df.columns:
                    dst_ip = str(raw_df.iloc[i].get('dst_ip', ''))
                if 'destination_port' in raw_df.columns:
                    try:
                        dst_port = int(raw_df.iloc[i].get('destination_port', 0))
                    except (ValueError, TypeError):
                        dst_port = 0

                if src_ip and dst_ip:
                    original_class = r.get("cascade_class", "")
                    new_class = self.ddos_aggregator.classify(
                        src_ip, dst_ip, dst_port, original_class)
                    if new_class != original_class:
                        r["_postclass_from"] = original_class
                        r["_postclass_by"] = "DDoS"
                        r["cascade_class"] = new_class
                        r["cascade_verdict"] = f"ATTACK: {new_class}"
                        r["postclass_override"] = f"DDoS rule: {original_class} -> {new_class}"

        return results


# ═══════════════════════════════════════════════════════════════
#  Global State (thread-safe)
# ═══════════════════════════════════════════════════════════════
class NIDSState:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine: InferenceEngine | None = None
        self.capturing = False
        self.stop_event = threading.Event()
        self.interface = ""
        self.capture_start = 0.0

        # Counters
        self.flow_count = 0
        self.benign_count = 0
        self.attack_count = 0
        self.zeroday_count = 0

        # Data stores (append-only for the session)
        self.flows: list[dict] = []       # {id, time, src, dst, proto, verdict, class, ...}
        self.alerts: list[dict] = []      # {id, time, msg}

        # CSV output
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _ts = datetime.now().strftime('%Y%m%d')
        self.csv_path = os.path.join(_script_dir, f"nids_flows_{_ts}.csv")
        self.csv_header_written = os.path.exists(self.csv_path)
        self.log_path = os.path.join(_script_dir, f"nids_alerts_{_ts}.log")

    def add_alert(self, msg: str):
        ts = datetime.now()
        with self.lock:
            self.alerts.append({
                "id": len(self.alerts),
                "time": ts.strftime("%H:%M:%S"),
                "msg": msg,
            })
        # File I/O OUTSIDE lock to avoid blocking capture
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts.isoformat()}  {msg}\n")
        except Exception:
            pass

    def add_flow(self, flow_data: dict):
        with self.lock:
            flow_data["id"] = self.flow_count
            self.flows.append(flow_data)
            self.flow_count += 1



    def get_stats(self) -> dict:
        with self.lock:
            elapsed = max(0.1, time.time() - self.capture_start) if self.capturing else 1
            return {
                "capturing": self.capturing,
                "interface": self.interface,
                "flows": self.flow_count,
                "benign": self.benign_count,
                "attacks": self.attack_count,
                "zeroday": self.zeroday_count,
                "rate": round(self.flow_count / elapsed, 1) if self.capturing else 0,
            }


STATE = NIDSState()
_csv_lock = threading.Lock()
# Bounded queue: drop flows rather than growing unbounded under flood.
# At flood rates (50k+ pps) an unbounded queue makes the export worker
# fall behind forever and keeps the GIL saturated servicing queue puts.
_EXPORT_QUEUE_MAXSIZE = 8000
_export_queue: queue.Queue = queue.Queue(maxsize=_EXPORT_QUEUE_MAXSIZE)
_dropped_flows = 0          # flows dropped due to full queue (flood protection)
_dropped_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════
#  Capture Worker Thread
# ═══════════════════════════════════════════════════════════════
FLASK_PORT = 5000


def export_flow(flow, local_ips: set, engine: InferenceEngine):
    """Export a finished flow to the inference engine."""
    try:
        from realtime_cicflow import extract_features

        src = flow.src_ip
        dst = flow.dst_ip

        if src in local_ips and dst in local_ips:
            return
        if dst.startswith("ff") or src.startswith("ff"):
            return
        if dst.startswith("224.") or dst.startswith("239.") or dst == "255.255.255.255":
            return
        if src not in local_ips and dst not in local_ips:
            return

        # RST flow quality gate
        _RST_MIN_PACKETS = 4
        if (hasattr(flow, 'all_pkts') and
                len(flow.all_pkts) < _RST_MIN_PACKETS and
                any(getattr(p, 'tcp_flags', 0) & 0x04 for p in flow.all_pkts)):
            return

        features_df = extract_features(flow)
        if features_df is None:
            return

        # Inject src/dst IP and port into features so InferenceEngine.process()
        # can pass them to post-classifiers (DDoS aggregator)
        features_df['src_ip'] = src
        features_df['dst_ip'] = dst
        features_df['destination_port'] = flow.dst_port

        meta = dict(
            src_ip=src, dst_ip=dst,
            src_port=flow.src_port, dst_port=flow.dst_port,
            protocol=flow.proto,
        )

        results = engine.process(features_df)
        if not results:
            return

        r = results[0]
        verdict = r["cascade_verdict"]
        cascade_class = r.get("cascade_class", "?")

        # ── Outbound traffic bypass ────────────────────────────────
        # Normal system traffic (DNS, NTP, HTTPS, apt, etc.) from this
        # host to external servers gets flagged as UNKNOWN/DoS by OSPREY.
        # Override to BENIGN if traffic is OUTBOUND (src = this machine).
        # Inbound attacks have src = attacker IP, so they won't match.
        if cascade_class in ("UNKNOWN", "DoS"):
            if src in local_ips:
                r["cascade_verdict"] = "\U0001f7e2 BENIGN"
                r["cascade_class"] = "BENIGN"
                verdict = r["cascade_verdict"]
                cascade_class = "BENIGN"

        # ── R38 fix: Orphan server-response bypass ────────────────
        # When a forward flow (attacker→target:80) expires/FINs before
        # late response packets arrive, those packets create a NEW flow
        # with src=target:80, dst=attacker. These orphan response flows
        # get misclassified as DoS (2,269 false positives in R38).
        # Fix: if src is local AND src_port is a server port, this is
        # always a server response — never an attack.
        _SERVER_RESPONSE_PORTS = {80, 443, 8080, 8443}
        if cascade_class in ("UNKNOWN", "DoS", "DDoS"):
            if src in local_ips and flow.src_port in _SERVER_RESPONSE_PORTS:
                r["cascade_verdict"] = "\U0001f7e2 BENIGN"
                r["cascade_class"] = "BENIGN"
                verdict = r["cascade_verdict"]
                cascade_class = "BENIGN"

        # ── Inbound response traffic bypass ───────────────────────
        # When the local machine initiates a connection to an external
        # server (HTTPS, DNS, etc.), response flows are keyed as:
        #   external_ip:443 → local_ip
        # The outbound bypass above misses these because src is external.
        # Real DoS/DDoS attacks never originate from src_port 443/80/53
        # — attackers use high ephemeral ports. So if src_port is a
        # well-known service port and dst is local, it is response traffic.
        _SERVICE_PORTS = {
            80, 443, 53, 123, 8080, 8443,   # HTTP, HTTPS, DNS, NTP
            993, 995, 587, 465, 25,          # mail
            67, 68, 546, 547,               # DHCP
            110, 143,                        # POP3, IMAP
        }
        if cascade_class in ("UNKNOWN", "DoS"):
            if dst in local_ips and src not in local_ips:
                if flow.src_port in _SERVICE_PORTS:
                    r["cascade_verdict"] = "\U0001f7e2 BENIGN"
                    r["cascade_class"] = "BENIGN"
                    verdict = r["cascade_verdict"]
                    cascade_class = "BENIGN"

        is_benign = cascade_class == "BENIGN"

        ts_str = datetime.now().strftime("%H:%M:%S")

        # Update counters
        with STATE.lock:
            if is_benign:
                STATE.benign_count += 1
            elif cascade_class == "UNKNOWN":
                STATE.zeroday_count += 1
            else:
                STATE.attack_count += 1



        # Add to flow list (sanitize numpy types / NaN for JSON)
        _score = r.get("daemon_score", 0)
        _energy = r.get("osprey_energy", 0)
        try:
            _score = 0.0 if (_score != _score) else round(float(_score), 2)
        except (TypeError, ValueError):
            _score = 0.0
        try:
            _energy = 0.0 if (_energy != _energy) else round(float(_energy), 2)
        except (TypeError, ValueError):
            _energy = 0.0

        STATE.add_flow({
            "time": ts_str,
            "src": src,
            "dst": dst,
            "src_port": int(flow.src_port),
            "dst_port": int(flow.dst_port),
            "proto": proto_name(flow.proto),
            "stage1": r.get("daemon_verdict", "?"),
            "score": _score,
            "stage2": r.get("osprey_class", "-"),
            "cascade_class": cascade_class,
            "energy": _energy,
            "verdict": verdict,
        })

        # Alert for non-benign
        if not is_benign:
            STATE.add_alert(
                f"[{cascade_class}] {src}:{flow.src_port} → {dst}:{flow.dst_port} "
                f"({proto_name(flow.proto)}) | {verdict}"
            )

        # Write to CSV
        try:
            row = {}
            feat_dict = features_df.iloc[0].to_dict()
            row.update(feat_dict)
            row["src_ip"] = src
            row["src_port"] = int(flow.src_port)
            row["dst_ip"] = dst
            row["dst_port"] = int(flow.dst_port)
            # R36 fix: use flow start time (first packet), not export time,
            # so post-run analysis scripts can correctly match flows to
            # attack windows. datetime.now() caused 5-6 min skew for
            # long-lived/delayed flows (e.g. DDoS-LOIC spoofed sources).
            row["timestamp"] = datetime.fromtimestamp(
                flow.start_time).isoformat()
            row["cascade_class"] = cascade_class
            row["cascade_verdict"] = verdict
            row["daemon_verdict"] = r.get("daemon_verdict", "")
            row["daemon_score"] = r.get("daemon_score", 0)

            with _csv_lock:
                with open(STATE.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not STATE.csv_header_written:
                        writer.writeheader()
                        STATE.csv_header_written = True
                    writer.writerow(row)
        except Exception:
            pass

    except Exception as e:
        STATE.add_alert(f"Flow export error: {e}")


_NUM_EXPORT_WORKERS = 3   # parallel ML workers; GIL limits true parallelism
                          # but workers overlap on I/O (CSV write, queue wait)

def _export_worker(worker_id: int, local_ips: set, engine: InferenceEngine):
    """Dedicated thread that drains _export_queue and runs ML inference.

    This keeps ML inference OFF the Scapy callback thread, preventing
    packet drops during high-volume attacks (the core GIL fix).
    Multiple workers reduce per-flow latency and keep the queue shorter.
    """
    while True:
        flow = _export_queue.get()
        if flow is None:  # sentinel to shut down
            _export_queue.task_done()
            break
        try:
            export_flow(flow, local_ips, engine)
        except Exception as e:
            print(f"[EXPORT_WORKER-{worker_id}] ERROR: {e}", flush=True)
            STATE.add_alert(f"Export worker error: {e}")
        finally:
            _export_queue.task_done()


def _enqueue_flow(flow):
    """Non-blocking enqueue with drop-on-full flood protection."""
    global _dropped_flows
    try:
        _export_queue.put_nowait(flow)
    except queue.Full:
        with _dropped_lock:
            _dropped_flows += 1

SNAPSHOT_INTERVAL = 30.0
SNAPSHOT_MIN_AGE = 10.0
SNAPSHOT_MIN_PKTS = 3

def _should_snapshot(flow, now: float) -> bool:
    age = now - flow.start_time
    if age < SNAPSHOT_MIN_AGE: return False
    if len(flow.all_pkts) < SNAPSHOT_MIN_PKTS: return False
    last = getattr(flow, '_last_snapshot_time', 0.0)
    since_last = now - last if last > 0 else age
    return since_last >= SNAPSHOT_INTERVAL

def capture_worker(iface: str, engine: InferenceEngine):
    """Background thread: Scapy capture + flow assembly."""
    local_ips = get_local_ips()

    try:
        from realtime_cicflow import (
            FlowRecord, PacketInfo, extract_features,
            _make_flow_key, FLOW_TIMEOUT,
        )
        from scapy.sendrecv import AsyncSniffer
        from scapy.all import IP, TCP, UDP

        STATE.add_alert(f"Starting capture on {iface}...")

        flows = {}
        flow_lock = threading.Lock()
        IDLE_TIMEOUT = 120.0

        # Start pool of export worker threads (ML inference off the callback)
        export_threads = []
        for _wid in range(_NUM_EXPORT_WORKERS):
            et = threading.Thread(
                target=_export_worker, args=(_wid, local_ips, engine), daemon=True)
            et.start()
            export_threads.append(et)

        def process_packet(packet):
            if STATE.stop_event.is_set():
                return
            if not packet.haslayer(IP):
                return

            try:
                ip = packet[IP]
                now = float(packet.time)
                proto = ip.proto

                if proto == 6 and packet.haslayer(TCP):
                    tcp = packet[TCP]
                    src_port, dst_port = tcp.sport, tcp.dport
                    flags = int(tcp.flags)
                    win_size = tcp.window
                    ip_hdr_len = ip.ihl * 4
                    tcp_hdr_len = tcp.dataofs * 4
                    header_len = ip_hdr_len + tcp_hdr_len
                    payload_len = len(bytes(tcp.payload))
                    pkt_len = len(packet[IP])
                elif proto == 17 and packet.haslayer(UDP):
                    udp = packet[UDP]
                    src_port, dst_port = udp.sport, udp.dport
                    flags, win_size = 0, 0
                    ip_hdr_len = ip.ihl * 4
                    header_len = ip_hdr_len + 8
                    payload_len = len(bytes(udp.payload))
                    pkt_len = len(packet[IP])
                else:
                    return

                src_ip, dst_ip = ip.src, ip.dst

                if src_ip in local_ips and dst_ip in local_ips:
                    return
                if dst_ip.startswith("224.") or dst_ip.startswith("239.") or dst_ip == "255.255.255.255":
                    return
                if dst_ip.startswith("ff") or src_ip.startswith("ff"):
                    return

                fwd_key = _make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                bwd_key = _make_flow_key(dst_ip, src_ip, dst_port, src_port, proto)

                with flow_lock:
                    if fwd_key in flows:
                        key, direction = fwd_key, 0
                        flow = flows[key]
                    elif bwd_key in flows:
                        key, direction = bwd_key, 1
                        flow = flows[key]
                    else:
                        # ── FIX: Reject SYN-ACK-only flows (no prior SYN seen) ────
                        # These are response-only half-open flows that produce
                        # degenerate BENIGN-leaking features (R17 root cause #3).
                        if proto == 6 and (flags & 0x12) == 0x12 and not (flags & 0x01):
                            # SYN+ACK without an existing flow entry → skip
                            return
                        key, direction = fwd_key, 0
                        flow = FlowRecord(
                            src_ip=src_ip, dst_ip=dst_ip,
                            src_port=src_port, dst_port=dst_port,
                            proto=proto, start_time=now, last_seen=now,
                        )
                        flows[key] = flow

                    pinfo = PacketInfo(
                        timestamp=now, ip_length=pkt_len,
                        header_length=header_len,
                        direction=direction, tcp_flags=flags,
                        tcp_window=win_size, payload_len=payload_len,
                    )
                    flow.add_packet(pinfo)

                    # ── Ported Fix 3: Dual-FIN termination ──────────────────
                    if proto == 6 and (flags & 0x01):  # FIN
                        if direction == 0:
                            flow.fwd_fin_cnt += 1
                        else:
                            flow.bwd_fin_cnt += 1
                        
                        if flow.fwd_fin_cnt > 0 and (flow.fwd_fin_cnt + flow.bwd_fin_cnt) >= 2:
                            _enqueue_flow(flows.pop(key))
                            return
                        else:
                            return

                    # RST immediately exports
                    if proto == 6 and (flags & 0x04):  # RST
                        _enqueue_flow(flows.pop(key))
                        return
                    
                    # Normal packet — check if flow is already closed
                    if proto == 6:
                        if direction == 0 and flow.fwd_fin_cnt > 0: return
                        if direction == 1 and flow.bwd_fin_cnt > 0: return

                    # ── FIX: NO expired-flow scan here ────────────────────────────
                    # REMOVED: the O(n) scan of all flows that previously ran on
                    # EVERY packet. Under DoS/DDoS floods this was scanning thousands
                    # of entries per packet. The background loop (every 2s) now owns
                    # all expiry exclusively, keeping this callback O(1).

            except Exception as e:
                STATE.add_alert(f"PKT ERROR: {e}")

        # BPF filter: IP only, exclude Flask port
        bpf = f"ip and not port {FLASK_PORT}"
        sniffer = AsyncSniffer(
            iface=iface,
            prn=process_packet,
            store=False,
            filter=bpf,
        )

        try:
            sniffer.start()
        except OSError as e:
            STATE.add_alert(f"Cannot capture on '{iface}': {e}")
            STATE.capturing = False
            for _ in export_threads:          # one sentinel per worker
                _export_queue.put(None)
            for et in export_threads:
                et.join(timeout=5)
            return

        STATE.add_alert(f"Capture started on {iface} (BPF: {bpf})")

        # Periodic flow export loop (owns ALL expiry and snapshots)
        try:
            while not STATE.stop_event.is_set():
                time.sleep(2)
                with flow_lock:
                    now = time.time()
                    expired_keys = []
                    snapshot_keys = []
                    
                    for k, f in flows.items():
                        if (now - f.last_seen) > IDLE_TIMEOUT or (now - f.start_time) > FLOW_TIMEOUT:
                            expired_keys.append(k)
                        elif _should_snapshot(f, now):
                            snapshot_keys.append(k)
                            
                    for k in expired_keys:
                        _enqueue_flow(flows.pop(k))
                        
                    for k in snapshot_keys:
                        f = flows[k]
                        f._last_snapshot_time = now
                        # Clone flow for snapshot evaluation (without removing from table)
                        import copy
                        _enqueue_flow(copy.deepcopy(f))
                        
        except Exception as e:
            STATE.add_alert(f"Capture error: {e}")
        finally:
            sniffer.stop()
            # Drain remaining flows
            with flow_lock:
                for k in list(flows.keys()):
                    _enqueue_flow(flows.pop(k))
            # Wait for export queue to finish, then shut down all workers
            _export_queue.join()
            for _ in export_threads:
                _export_queue.put(None)  # one sentinel per worker
            for et in export_threads:
                et.join(timeout=10)

        STATE.add_alert("Capture stopped.")
        STATE.capturing = False

    except ImportError as e:
        STATE.add_alert(f"Import error: {e}. Need realtime_cicflow.py + scapy.")
        STATE.capturing = False


# ═══════════════════════════════════════════════════════════════
#  Flask App
# ═══════════════════════════════════════════════════════════════


def _sanitize_for_json(data):
    """Recursively convert numpy types and NaN/Inf to JSON-safe values."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_json(i) for i in data]
    if isinstance(data, (np.floating,)):
        v = float(data)
        if v != v or v == float('inf') or v == float('-inf'):
            return 0.0
        return v
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, float):
        if data != data or data == float('inf') or data == float('-inf'):
            return 0.0
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


app = Flask(__name__)


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


@app.route("/api/interfaces")
def api_interfaces():
    return jsonify(get_all_interfaces())


@app.route("/api/stats")
def api_stats():
    return jsonify(_sanitize_for_json(STATE.get_stats()))


@app.route("/api/flows")
def api_flows():
    since = int(request.args.get("since", 0))
    limit = int(request.args.get("limit", 200))
    with STATE.lock:
        data = STATE.flows[since:since + limit]  # list slice = shallow copy
        total = STATE.flow_count
    return jsonify(_sanitize_for_json({"flows": data, "total": total}))


@app.route("/api/alerts")
def api_alerts():
    since = int(request.args.get("since", 0))
    with STATE.lock:
        data = STATE.alerts[since:]
    return jsonify(_sanitize_for_json(data))





@app.route("/api/start", methods=["POST"])
def api_start():
    if STATE.engine is None:
        return jsonify({"error": "Models not loaded yet"}), 503
    if STATE.capturing:
        return jsonify({"error": "Already capturing"}), 400

    data = request.get_json(silent=True) or {}
    iface = data.get("interface", get_default_interface())

    STATE.capturing = True
    STATE.interface = iface
    STATE.stop_event.clear()
    STATE.capture_start = time.time()

    t = threading.Thread(target=capture_worker, args=(iface, STATE.engine), daemon=True)
    t.start()

    return jsonify({"status": "started", "interface": iface})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if not STATE.capturing:
        return jsonify({"error": "Not capturing"}), 400
    STATE.stop_event.set()
    STATE.capturing = False
    return jsonify({"status": "stopped"})


@app.route("/api/update")
def api_update():
    """Combined poll endpoint — replaces 4 separate /api/* calls.

    Returns stats + incremental flows/alerts in one HTTP round-trip,
    cutting Flask GIL hits from 4/sec to 1 every 2s — an 8× reduction.
    """
    since_flow  = int(request.args.get("sf", 0))
    since_alert = int(request.args.get("sa", 0))
    limit       = int(request.args.get("limit", 200))

    with STATE.lock:
        elapsed = max(0.1, time.time() - STATE.capture_start) if STATE.capturing else 1
        stats = {
            "capturing": STATE.capturing,
            "interface": STATE.interface,
            "flows":     STATE.flow_count,
            "benign":    STATE.benign_count,
            "attacks":   STATE.attack_count,
            "zeroday":   STATE.zeroday_count,
            "rate":      round(STATE.flow_count / elapsed, 1) if STATE.capturing else 0,
            "dropped":   _dropped_flows,
        }
        flows_slice  = STATE.flows[since_flow:since_flow + limit]
        total_flows  = STATE.flow_count
        alerts_slice = STATE.alerts[since_alert:]

    return jsonify(_sanitize_for_json({
        "stats":    stats,
        "flows":    {"flows": flows_slice, "total": total_flows},
        "alerts":   alerts_slice,
    }))


@app.route("/api/export")
def api_export():
    if os.path.exists(STATE.csv_path):
        with open(STATE.csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(
            content,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=nids_flows.csv"}
        )
    return jsonify({"error": "No CSV file"}), 404


# ═══════════════════════════════════════════════════════════════
#  Dashboard HTML (embedded)
# ═══════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OSPREY × DAEMON NIDS</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e6edf3;
    height: 100vh;
    overflow: hidden;
    font-size: 15px;
}

/* Header */
.header {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border-bottom: 1px solid #30363d;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header h1 {
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header .status {
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #da3633;
    animation: pulse 2s infinite;
}
.status-dot.active { background: #3fb950; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Controls */
.controls {
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
}
.controls select, .controls input {
    background: #0d1117;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
    font-family: 'Inter', sans-serif;
}
.controls select:focus, .controls input:focus {
    border-color: #58a6ff;
    outline: none;
}
.btn {
    padding: 8px 20px;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}
.btn:hover { transform: translateY(-1px); filter: brightness(1.1); }
.btn:active { transform: translateY(0); }
.btn-start { background: #238636; color: #fff; }
.btn-stop { background: #da3633; color: #fff; }
.btn-export { background: #6e40c9; color: #fff; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    padding: 16px 24px;
}
.stat-card {
    background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s;
}
.stat-card:hover {
    border-color: #58a6ff;
    box-shadow: 0 0 20px rgba(88,166,255,0.1);
}
.stat-label {
    font-size: 13px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 32px;
    font-weight: 700;
}
.stat-flows .stat-value { color: #58a6ff; }
.stat-benign .stat-value { color: #3fb950; }
.stat-attacks .stat-value { color: #f85149; }
.stat-zeroday .stat-value { color: #d2a8ff; }
.stat-rate .stat-value { color: #f0883e; }


/* Main content */
.main {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 16px;
    padding: 0 24px;
    height: calc(100vh - 280px);
    min-height: 0;
    margin-bottom: 16px;
}

/* Flow Table */
.table-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.table-header {
    padding: 12px 16px;
    border-bottom: 1px solid #30363d;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.table-header h2 {
    font-size: 16px;
    color: #58a6ff;
}
.table-wrap {
    flex: 1;
    overflow-y: auto;
    overflow-x: auto;   /* FIX: allow horizontal scroll so all 9 cols are reachable */
    min-height: 0;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
}
table {
    width: 100%;
    min-width: 900px;  /* matches sum of all column widths */
    border-collapse: collapse;
    font-size: 13px;
    table-layout: fixed;
}
/* Explicit column widths — verdict gets a guaranteed 220px minimum */
thead th:nth-child(1)  { width: 48px;  }   /* #        */
thead th:nth-child(2)  { width: 72px;  }   /* Time     */
thead th:nth-child(3)  { width: 130px; }   /* Src IP   */
thead th:nth-child(4)  { width: 130px; }   /* Dst IP   */
thead th:nth-child(5)  { width: 54px;  }   /* Proto    */
thead th:nth-child(6)  { width: 90px;  }   /* Stage 1  */
thead th:nth-child(7)  { width: 90px;  }   /* Stage 2  */
thead th:nth-child(8)  { width: 100px; }   /* Class    */
thead th:nth-child(9)  { width: 220px; }   /* Verdict  */
thead th {
    background: #1c2536;
    color: #8b949e;
    padding: 9px 8px;
    text-align: left;
    position: sticky;
    top: 0;
    z-index: 1;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
tbody tr {
    border-bottom: 1px solid #21262d;
    transition: background 0.15s;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #1c2536; }
tbody td {
    padding: 8px 8px;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Verdict colors — applied to the last (Verdict) column */
tr.v-benign td:last-child  { color: #3fb950; white-space: normal; overflow: visible; }
tr.v-attack td:last-child  { color: #f85149; font-weight: 600; white-space: normal; overflow: visible; }
tr.v-unknown td:last-child { color: #d2a8ff; white-space: normal; overflow: visible; }
tr.v-benign { border-left: 3px solid #3fb950; }
tr.v-attack { border-left: 3px solid #f85149; }
tr.v-unknown { border-left: 3px solid #d2a8ff; }

/* Pagination */
.pagination {
    padding: 8px 16px;
    border-top: 1px solid #30363d;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #8b949e;
}
.pagination button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}
.pagination button:hover { background: #30363d; }

/* Side Panels */
.side-panels {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow: auto;
}
.log-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.log-panel h3 {
    font-size: 18px;
    padding: 10px 14px;
    border-bottom: 1px solid #30363d;
    color: #f0883e;
    font-weight: 600;
    flex-shrink: 0;
}

.log-content {
    flex: 1;
    overflow-y: auto;
    padding: 10px 16px;
    font-size: 15px;
    font-family: 'Inter', monospace;
    min-height: 0;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
}
.log-entry {
    padding: 4px 0;
    border-bottom: 1px solid #21262d;
    animation: fadeIn 0.3s;
    font-size: 15px;
}
.log-entry:last-child { border-bottom: none; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-8px); }
    to { opacity: 1; transform: translateX(0); }
}
.log-time { color: #484f58; margin-right: 8px; }
.log-attack { color: #f85149; }
.log-ddos { color: #f0883e; }
.log-bf { color: #3fb950; }
.log-unknown { color: #d2a8ff; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* Model info */
.model-info {
    padding: 4px 24px 8px;
    font-size: 11px;
    color: #484f58;
}

@media (max-width: 1024px) {
    .stats-grid { grid-template-columns: repeat(3, 1fr); }
    .main { grid-template-columns: 1fr; height: auto; }
}
</style>
</head>
<body>

<div class="header">
    <h1>OSPREY × DAEMON NIDS</h1>
    <div class="status">
        <div class="status-dot" id="statusDot"></div>
        <span id="statusText" style="color:#8b949e; font-size:13px;">Stopped</span>
    </div>
</div>

<div class="controls">
    <label style="color:#8b949e; font-size:13px;">Interface:</label>
    <select id="ifaceSelect"></select>
    <button class="btn btn-start" id="btnStart" onclick="startCapture()">▶ Start</button>
    <button class="btn btn-stop" id="btnStop" onclick="stopCapture()" disabled>■ Stop</button>
    <div style="flex:1"></div>
    <button class="btn btn-export" onclick="exportCSV()">💾 Export CSV</button>
</div>

<div class="stats-grid">
    <div class="stat-card stat-flows">
        <div class="stat-label">Flows</div>
        <div class="stat-value" id="statFlows">0</div>
    </div>
    <div class="stat-card stat-benign">
        <div class="stat-label">Benign</div>
        <div class="stat-value" id="statBenign">0</div>
    </div>
    <div class="stat-card stat-attacks">
        <div class="stat-label">Attacks</div>
        <div class="stat-value" id="statAttacks">0</div>
    </div>
    <div class="stat-card stat-zeroday">
        <div class="stat-label">Unknown</div>
        <div class="stat-value" id="statZeroday">0</div>
    </div>
    <div class="stat-card stat-rate">
        <div class="stat-label">Flows/sec</div>
        <div class="stat-value" id="statRate">0</div>
    </div>

</div>

<div class="model-info" id="modelInfo">Loading models...</div>

<div class="main">
    <div class="table-panel">
        <div class="table-header">
            <h2>Live Flow Table</h2>
            <span id="flowCount" style="color:#8b949e; font-size:14px;">0 flows</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th><th>Time</th><th>Src IP</th><th>Dst IP</th>
                        <th>Proto</th><th>Stage 1</th><th>Stage 2</th><th>Class</th><th>Verdict</th>
                    </tr>
                </thead>
                <tbody id="flowBody"></tbody>
            </table>
        </div>
    </div>

    <div class="side-panels">
        <div class="log-panel">
            <h3>⚡ Alerts</h3>
            <div class="log-content" id="alertLog"></div>
        </div>

    </div>
</div>

<script>
// State
let lastFlowId = 0;
const MAX_VISIBLE_ROWS = 200;
const PAGE_SIZE = 200;          // rows per page (used by prevPage/nextPage)
let totalFlows = 0;
let lastAlertId = 0;

let isCapturing = false;
let autoScroll = true;          // auto-scroll table when new rows arrive
let currentPage = 0;            // pagination state

// Init
async function init() {
    // Load interfaces
    const res = await fetch('/api/interfaces');
    const ifaces = await res.json();
    const sel = document.getElementById('ifaceSelect');
    ifaces.forEach(ifc => {
        const opt = document.createElement('option');
        opt.value = ifc;
        opt.textContent = ifc;
        if (ifc !== 'lo') opt.selected = true;
        sel.appendChild(opt);
    });

    // ── FIX: single combined poll at 2s instead of 4 separate polls at 1s ──────
    // Previously 4 setInterval(fn, 1000) = 4 HTTP hits/sec to Flask = 4× GIL
    // contention. Now 1 hit every 2s = 8× reduction in Flask GIL pressure,
    // directly recovering packet-capture throughput under high-volume attacks.
    setInterval(pollAll, 2000);

    // Check model status immediately
    pollAll();
}

// Auto-scroll helper: only scroll if user is near the bottom
function isNearBottom(el, threshold = 60) {
    return (el.scrollHeight - el.scrollTop - el.clientHeight) < threshold;
}

// API calls
async function startCapture() {
    const iface = document.getElementById('ifaceSelect').value;
    const res = await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({interface: iface})
    });
    const data = await res.json();
    if (res.ok) {
        isCapturing = true;
        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnStop').disabled = false;
        autoScroll = true;
    } else {
        alert(data.error || 'Failed to start');
    }
}

async function stopCapture() {
    await fetch('/api/stop', {method: 'POST'});
    isCapturing = false;
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled = true;
}

function exportCSV() {
    window.open('/api/export', '_blank');
}

// ── Unified poller (replaces 4 separate poll functions) ─────────────────────
async function pollAll() {
    try {
        const res = await fetch(
            `/api/update?sf=${lastFlowId}&sa=${lastAlertId}&limit=200`
        );
        const d = await res.json();

        // ── Stats ──────────────────────────────────────────────────────────────
        const s = d.stats;
        document.getElementById('statFlows').textContent    = s.flows.toLocaleString();
        document.getElementById('statBenign').textContent   = s.benign.toLocaleString();
        document.getElementById('statAttacks').textContent  = s.attacks.toLocaleString();
        document.getElementById('statZeroday').textContent  = s.zeroday.toLocaleString();
        document.getElementById('statRate').textContent     = s.rate;


        const dot = document.getElementById('statusDot');
        const txt = document.getElementById('statusText');
        if (s.capturing) {
            dot.classList.add('active');
            txt.textContent = 'Capturing on ' + s.interface +
                (s.dropped ? ` ⚠ ${s.dropped} dropped` : '');
            document.getElementById('btnStart').disabled = true;
            document.getElementById('btnStop').disabled  = false;
        } else {
            dot.classList.remove('active');
            txt.textContent = 'Stopped';
            document.getElementById('btnStart').disabled = false;
            document.getElementById('btnStop').disabled  = true;
        }

        // ── Flows ─────────────────────────────────────────────────────────────
        const flowData = d.flows;
        totalFlows = flowData.total;
        document.getElementById('flowCount').textContent =
            totalFlows.toLocaleString() + ' flows';

        if (flowData.flows.length > 0) {
            const tbody = document.getElementById('flowBody');
            const emptyRow = document.getElementById('emptyRow');
            if (emptyRow) emptyRow.remove();

            const wrap = document.querySelector('.table-wrap');
            const shouldScroll = isNearBottom(wrap);

            flowData.flows.forEach(f => {
                const cls = f.cascade_class;
                let rowClass = 'v-attack';
                if (cls === 'BENIGN') rowClass = 'v-benign';
                else if (cls === 'UNKNOWN') rowClass = 'v-unknown';

                const tr = document.createElement('tr');
                tr.className = rowClass;
                tr.innerHTML = `<td>${f.id}</td><td>${f.time}</td><td>${f.src}</td>` +
                    `<td>${f.dst}</td><td>${f.proto}</td><td>${f.stage1}</td>` +
                    `<td>${f.stage2}</td><td><strong>${cls}</strong></td>` +
                    `<td>${f.verdict}</td>`;
                tbody.appendChild(tr);
                lastFlowId = f.id + 1;
            });

            if (shouldScroll) wrap.scrollTop = wrap.scrollHeight;
        } else if (totalFlows === 0) {
            const tbody = document.getElementById('flowBody');
            if (tbody.children.length === 0) {
                tbody.innerHTML = '<tr id="emptyRow"><td colspan="9" ' +
                    'style="text-align:center;color:#8b949e;padding:40px;font-size:16px;">' +
                    'Waiting for flows...</td></tr>';
            }
        }

        // ── Alerts ────────────────────────────────────────────────────────────
        if (d.alerts.length > 0) {
            const log = document.getElementById('alertLog');
            const shouldScroll = isNearBottom(log);
            d.alerts.forEach(a => {
                const div = document.createElement('div');
                div.className = 'log-entry';
                let colorClass = 'log-attack';
                if (a.msg.includes('[DDoS]'))        colorClass = 'log-ddos';
                else if (a.msg.includes('[Brute Force]')) colorClass = 'log-bf';
                else if (a.msg.includes('[UNKNOWN]')) colorClass = 'log-unknown';
                else if (a.msg.includes('started') || a.msg.includes('stopped'))
                    colorClass = '';
                div.innerHTML = `<span class="log-time">${a.time}</span>` +
                    `<span class="${colorClass}">${escHtml(a.msg)}</span>`;
                log.appendChild(div);
                lastAlertId = a.id + 1;
            });
            if (shouldScroll) log.scrollTop = log.scrollHeight;
        }



    } catch (e) { console.error('pollAll error:', e); }
}

// ── Legacy individual pollers (kept for /api/* compat, not called by timer) ──
async function pollStats() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        document.getElementById('statFlows').textContent = s.flows.toLocaleString();
    } catch (e) {}
}



// Pagination
function prevPage() {
    if (currentPage > 0) {
        currentPage--;
        autoScroll = false;
        pollFlows();
    }
}

function nextPage() {
    const totalPages = Math.ceil(totalFlows / PAGE_SIZE);
    if (currentPage < totalPages - 1) {
        currentPage++;
        if (currentPage >= totalPages - 1) autoScroll = true;
        pollFlows();
    }
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Start
init();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="OSPREY × DAEMON NIDS — Web Dashboard",
    )
    parser.add_argument("--model", "-m", default=None,
                        help="Path to nids_models.pkl")
    parser.add_argument("--iface", "-i", default=None,
                        help="Auto-start capture on this interface")
    parser.add_argument("--port", "-p", type=int, default=5000,
                        help="Web dashboard port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    global FLASK_PORT
    FLASK_PORT = args.port

    model_path = args.model if args.model else _find_model_bundle()
    if not os.path.isfile(model_path):
        print(f"ERROR: Model bundle not found: {model_path}")
        print("Place nids_models.pkl in the same directory, or use --model")
        sys.exit(1)

    print("Loading models...")
    STATE.engine = InferenceEngine(model_path)
    b = STATE.engine.bundle
    print(f"  DAEMON: {STATE.engine.daemon_params:,} params, "
          f"tau={b['daemon_threshold']:.1f}")
    print(f"  OSPREY: {STATE.engine.osprey_params:,} params")
    print(f"Models loaded.")

    # Auto-start capture if --iface provided
    if args.iface:
        STATE.capturing = True
        STATE.interface = args.iface
        STATE.stop_event.clear()
        STATE.capture_start = time.time()
        t = threading.Thread(
            target=capture_worker,
            args=(args.iface, STATE.engine),
            daemon=True,
        )
        t.start()
        print(f"Auto-started capture on {args.iface}")

    print(f"\n  Dashboard: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
