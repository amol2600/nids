#!/usr/bin/env python3
"""
Real-Time CICFlowMeter Feature Extractor — OSPREY × DAEMON Compatible
======================================================================
Produces CIC-IDS2017–compatible features from live network traffic using
Scapy, then runs them through the full OSPREY + DAEMON dual-pipeline for
real-time intrusion detection.

This is a faithful replica of CICFlowMeter (Java) as used to generate
the CIC-IDS2017 dataset, including all known bugs and edge cases.

Key implementation details matching CICFlowMeter:
  1. Packet lengths use PAYLOAD ONLY (not IP total length)
  2. Standard deviation uses sample std (ddof=1, Apache SummaryStatistics)
  3. FIN termination requires BOTH directions to send FIN
  4. Flow timeout is absolute from flow start (not idle-based)
  5. Active/Idle timing via subflow detection (1s gap threshold)
  6. Subflow features = total / sfCount (Java integer division)
  7. Bulk statistics use 1s time gaps + 4-packet threshold
  8. init_win_bytes_backward overwritten by every backward packet (Java bug)
  9. Flows with ≤1 packet are filtered out
  10. First packet payload double-counted in flowLengthStats (Java bug)
  11. act_data_pkt_fwd excludes the first forward packet
  12. IAT/duration in MICROSECONDS (CIC-IDS2017 standard)
  13. Full feature engineering (8 ratio features, log1p transforms)
  14. Dual-pipeline inference (OSPREY multi-class + DAEMON anomaly)

Requirements:
    pip install scapy numpy pandas torch scipy scikit-learn

Usage:
    sudo python realtime_cicflow.py -i eth0
    sudo python realtime_cicflow.py -i eth0 --output flows.csv
    sudo python realtime_cicflow.py -i eth0 --model /path/to/nids_models.pkl
    sudo python realtime_cicflow.py -i eth0 --self-traffic   # include self-traffic
"""
from __future__ import annotations

import argparse
import logging
import csv
import math
import os
import pickle
import queue
import signal
import sys
import threading
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scapy.all import IP, TCP, UDP, sniff, conf
    conf.verb = 0
except ImportError:
    print("[ERROR] scapy is required: pip install scapy")
    sys.exit(1)

# ── Import inference engine from infer.py ────────────────────────────────────
try:
    from infer import (
        OSPREY, DualPathAutoencoder,
        load_bundle, engineer_features,
        run_osprey, run_daemon, cascade_verdict,
        l2_normalize, _empty_osprey_fields,
        _SAFETY_MARGIN,
    )
    _HAS_INFER = True
except ImportError:
    _HAS_INFER = False
    print("[WARN] infer.py not found in path — CSV-only mode (no classification)")

# ── Import post-classification modules ───────────────────────────────────────
try:
    from ddos_aggregator import DDoSAggregator
    _HAS_POSTCLASS = True
except ImportError:
    _HAS_POSTCLASS = False
    print("[WARN] DDoS aggregator not found — DDoS multi-source detection disabled")


# ── Structured Logging ───────────────────────────────────────────────────────
def _setup_logger(log_dir: str = None) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file."""
    logger = logging.getLogger("realtime_cicflow")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console: INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File: DEBUG and above (everything)
    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"realtime_nids_{ts}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


log = _setup_logger()


# ═════════════════════════════════════════════════════════════════════════════
#  Constants (match CICFlowMeter defaults)
# ═════════════════════════════════════════════════════════════════════════════
FLOW_TIMEOUT           = 120.0     # seconds — absolute from flow start (CICFlowMeter default 120_000_000 μs)
UDP_FLOW_TIMEOUT       = 120.0     # seconds — same for UDP in CICFlowMeter
ACTIVITY_TIMEOUT_US    = 5_000_000 # microseconds — CICFlowMeter activityTimeout
SUBFLOW_GAP_THRESHOLD  = 1.0       # seconds — detectUpdateSubflows() threshold
IDLE_CLEANUP_TIMEOUT   = 120.0     # seconds — for our idle-based cleanup sweep
BULK_THRESHOLD         = 4         # min consecutive payload packets for "bulk"
BULK_GAP_THRESHOLD     = 1.0       # seconds — 1s gap resets bulk state
_US                    = 1_000_000.0  # seconds → microseconds

# ── Long-lived flow handling (Slowloris, Botnet, etc.) ────────────────────────
# CICFlowMeter only exports flows on FIN/RST/timeout, which misses slow attacks.
# We add periodic snapshots: every SNAPSHOT_INTERVAL seconds, clone & classify
# active flows so OSPREY can see accumulated features from long-lived connections.
# R24 reduced from 30→15s / 10→5s to cut Slowloris latency, but R25 analysis
# showed it degraded OOD accuracy: median 2-3 fwd_pkts vs R14's 5 fwd_pkts.
# R26 fix: revert to 30/10 — R14 proved this gives 98-100% OOD rejection.
SNAPSHOT_INTERVAL      = 30.0      # seconds — emit snapshot of long-lived flows
SNAPSHOT_MIN_AGE       = 10.0      # seconds — flow must be this old for first snapshot
SNAPSHOT_MIN_PKTS      = 3         # min packets before snapshot is useful

# ── Micro-flow filter ─────────────────────────────────────────────────────────
# Flows with ≤3 packets and 0 payload bytes are just failed TCP handshakes
# (SYN + SYN-ACK + RST with no data). These produce garbage features.
# Skip classification for these and log them as noise.
MICRO_FLOW_MAX_PKTS    = 3         # at most this many packets
MICRO_FLOW_MAX_BYTES   = 0         # and no payload bytes = micro-flow


# ═════════════════════════════════════════════════════════════════════════════
#  Flow Key
# ═════════════════════════════════════════════════════════════════════════════
FlowKey = Tuple[str, str, int, int, int]  # src_ip, dst_ip, src_port, dst_port, proto


def _make_flow_key(src_ip: str, dst_ip: str,
                   src_port: int, dst_port: int, proto: int) -> FlowKey:
    return (src_ip, dst_ip, src_port, dst_port, proto)


# ═════════════════════════════════════════════════════════════════════════════
#  Per-packet info
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PacketInfo:
    timestamp: float
    ip_length: int        # total IP packet length (bytes) — kept for header calc
    header_length: int    # IP + transport header bytes
    direction: int        # 0 = forward, 1 = backward
    tcp_flags: int        # TCP flags byte (0 for UDP)
    tcp_window: int       # TCP window size (0 for UDP)
    payload_len: int      # transport payload bytes — used for all feature calculations


# ═════════════════════════════════════════════════════════════════════════════
#  Flow Record (matches CICFlowMeter BasicFlow.java)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FlowRecord:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    start_time: float

    fwd_pkts: List[PacketInfo] = field(default_factory=list)
    bwd_pkts: List[PacketInfo] = field(default_factory=list)
    all_pkts: List[PacketInfo] = field(default_factory=list)

    last_seen: float = 0.0
    init_win_fwd: int = 0    # TCP initial window (forward) — CICFlowMeter default 0
    init_win_bwd: int = 0    # TCP initial window (backward) — overwritten every bwd pkt

    # Active / Idle tracking (CICFlowMeter: updateActiveIdleTime)
    start_active_time: float = 0.0
    end_active_time: float = 0.0
    active_periods: List[float] = field(default_factory=list)  # in seconds
    idle_periods: List[float] = field(default_factory=list)     # in seconds

    # Subflow tracking (CICFlowMeter: detectUpdateSubflows)
    sf_last_packet_ts: float = -1.0
    sf_count: int = 0
    sf_ac_helper: float = -1.0

    # FIN tracking (CICFlowMeter: fFIN_cnt, bFIN_cnt)
    fwd_fin_cnt: int = 0
    bwd_fin_cnt: int = 0

    # Bulk tracking — forward (CICFlowMeter: updateForwardBulk)
    fbulk_duration: float = 0.0
    fbulk_packet_count: int = 0
    fbulk_size_total: int = 0
    fbulk_state_count: int = 0
    fbulk_packet_count_helper: int = 0
    fbulk_start_helper: float = 0.0
    fbulk_size_helper: int = 0
    flast_bulk_ts: float = 0.0

    # Bulk tracking — backward (CICFlowMeter: updateBackwardBulk)
    bbulk_duration: float = 0.0
    bbulk_packet_count: int = 0
    bbulk_size_total: int = 0
    bbulk_state_count: int = 0
    bbulk_packet_count_helper: int = 0
    bbulk_start_helper: float = 0.0
    bbulk_size_helper: int = 0
    blast_bulk_ts: float = 0.0

    # Forward/backward byte+header tracking
    forward_bytes: int = 0
    backward_bytes: int = 0
    f_header_bytes: int = 0
    b_header_bytes: int = 0

    # Per-direction PSH/URG counts (CICFlowMeter: fPSH_cnt, bPSH_cnt, etc.)
    f_psh_cnt: int = 0
    b_psh_cnt: int = 0
    f_urg_cnt: int = 0
    b_urg_cnt: int = 0

    # Forward active data packet count (excludes first packet — Java: addPacket only)
    act_data_pkt_fwd: int = 0
    min_seg_size_fwd: int = 0

    # Forward/backward last seen timestamps
    fwd_last_seen: float = 0.0
    bwd_last_seen: float = 0.0

    _is_first_packet: bool = True

    # Snapshot tracking for long-lived flow periodic exports
    _last_snapshot_time: float = 0.0
    _snapshot_count: int = 0

    # Forensic: buffer raw scapy packets for PCAP export (only when enabled)
    raw_packets: list = field(default_factory=list)

    def _detect_update_subflows(self, pkt: PacketInfo):
        """CICFlowMeter: detectUpdateSubflows() — triggers on >1s gap."""
        ts = pkt.timestamp
        if self.sf_last_packet_ts < 0:
            self.sf_last_packet_ts = ts
            self.sf_ac_helper = ts
        else:
            gap = ts - self.sf_last_packet_ts
            if gap > SUBFLOW_GAP_THRESHOLD:
                self.sf_count += 1
                self._update_active_idle_time(ts)
                self.sf_ac_helper = ts
        self.sf_last_packet_ts = ts

    def _update_active_idle_time(self, current_time: float):
        """CICFlowMeter: updateActiveIdleTime() — threshold = activityTimeout (5s)."""
        threshold = ACTIVITY_TIMEOUT_US / _US  # convert to seconds
        gap = current_time - self.end_active_time
        if gap > threshold:
            active_dur = self.end_active_time - self.start_active_time
            if active_dur > 0:
                self.active_periods.append(active_dur)
            self.idle_periods.append(gap)
            self.start_active_time = current_time
            self.end_active_time = current_time
        else:
            self.end_active_time = current_time

    def _update_flow_bulk(self, pkt: PacketInfo):
        """CICFlowMeter: updateFlowBulk() — routes to fwd or bwd bulk update."""
        if pkt.direction == 0:
            self._update_forward_bulk(pkt)
        else:
            self._update_backward_bulk(pkt)

    def _update_forward_bulk(self, pkt: PacketInfo):
        """CICFlowMeter: updateForwardBulk(packet, blastBulkTS)

        Uses backward's lastBulkTS as the cross-direction reset marker.
        """
        size = pkt.payload_len
        ts = pkt.timestamp
        ts_of_last_bulk_in_other = self.blast_bulk_ts

        if ts_of_last_bulk_in_other > self.fbulk_start_helper:
            self.fbulk_start_helper = 0.0

        if size <= 0:
            return

        if self.fbulk_start_helper == 0.0:
            self.fbulk_start_helper = ts
            self.fbulk_packet_count_helper = 1
            self.fbulk_size_helper = size
            self.flast_bulk_ts = ts
        else:
            # Too much idle time? (>1s gap)
            gap = ts - self.flast_bulk_ts
            if gap > BULK_GAP_THRESHOLD:
                self.fbulk_start_helper = ts
                self.flast_bulk_ts = ts
                self.fbulk_packet_count_helper = 1
                self.fbulk_size_helper = size
            else:
                self.fbulk_packet_count_helper += 1
                self.fbulk_size_helper += size
                # New bulk detected (4th consecutive payload packet)
                if self.fbulk_packet_count_helper == BULK_THRESHOLD:
                    self.fbulk_state_count += 1
                    self.fbulk_packet_count += self.fbulk_packet_count_helper
                    self.fbulk_size_total += self.fbulk_size_helper
                    self.fbulk_duration += ts - self.fbulk_start_helper
                # Continuation of existing bulk
                elif self.fbulk_packet_count_helper > BULK_THRESHOLD:
                    self.fbulk_packet_count += 1
                    self.fbulk_size_total += size
                    self.fbulk_duration += ts - self.flast_bulk_ts
                self.flast_bulk_ts = ts

    def _update_backward_bulk(self, pkt: PacketInfo):
        """CICFlowMeter: updateBackwardBulk(packet, flastBulkTS)

        Uses forward's lastBulkTS as the cross-direction reset marker.
        """
        size = pkt.payload_len
        ts = pkt.timestamp
        ts_of_last_bulk_in_other = self.flast_bulk_ts

        if ts_of_last_bulk_in_other > self.bbulk_start_helper:
            self.bbulk_start_helper = 0.0

        if size <= 0:
            return

        if self.bbulk_start_helper == 0.0:
            self.bbulk_start_helper = ts
            self.bbulk_packet_count_helper = 1
            self.bbulk_size_helper = size
            self.blast_bulk_ts = ts
        else:
            gap = ts - self.blast_bulk_ts
            if gap > BULK_GAP_THRESHOLD:
                self.bbulk_start_helper = ts
                self.blast_bulk_ts = ts
                self.bbulk_packet_count_helper = 1
                self.bbulk_size_helper = size
            else:
                self.bbulk_packet_count_helper += 1
                self.bbulk_size_helper += size
                if self.bbulk_packet_count_helper == BULK_THRESHOLD:
                    self.bbulk_state_count += 1
                    self.bbulk_packet_count += self.bbulk_packet_count_helper
                    self.bbulk_size_total += self.bbulk_size_helper
                    self.bbulk_duration += ts - self.bbulk_start_helper
                elif self.bbulk_packet_count_helper > BULK_THRESHOLD:
                    self.bbulk_packet_count += 1
                    self.bbulk_size_total += size
                    self.bbulk_duration += ts - self.blast_bulk_ts
                self.blast_bulk_ts = ts

    def _check_flags(self, pkt: PacketInfo):
        """Track per-direction PSH/URG flags (CICFlowMeter: firstPacket + addPacket)."""
        if pkt.direction == 0:
            if pkt.tcp_flags & 0x08:  # PSH
                self.f_psh_cnt += 1
            if pkt.tcp_flags & 0x20:  # URG
                self.f_urg_cnt += 1
        else:
            if pkt.tcp_flags & 0x08:  # PSH
                self.b_psh_cnt += 1
            if pkt.tcp_flags & 0x20:  # URG
                self.b_urg_cnt += 1

    def first_packet(self, pkt: PacketInfo):
        """CICFlowMeter: firstPacket() — special handling for the very first packet."""
        self._update_flow_bulk(pkt)
        self._detect_update_subflows(pkt)
        self._check_flags(pkt)

        self.start_time = pkt.timestamp
        self.last_seen = pkt.timestamp
        self.start_active_time = pkt.timestamp
        self.end_active_time = pkt.timestamp

        self.all_pkts.append(pkt)

        if pkt.direction == 0:
            # Forward first packet
            self.min_seg_size_fwd = pkt.header_length
            self.init_win_fwd = pkt.tcp_window
            self.f_header_bytes = pkt.header_length
            self.fwd_last_seen = pkt.timestamp
            self.forward_bytes += pkt.payload_len
            self.fwd_pkts.append(pkt)
            # NOTE: act_data_pkt_fwd is NOT incremented for the first packet (Java: only in addPacket)
        else:
            # Backward first packet
            self.init_win_bwd = pkt.tcp_window
            self.b_header_bytes = pkt.header_length
            self.bwd_last_seen = pkt.timestamp
            self.backward_bytes += pkt.payload_len
            self.bwd_pkts.append(pkt)

        self._is_first_packet = False

    def add_packet(self, pkt: PacketInfo):
        """CICFlowMeter: addPacket() — for all packets after the first."""
        if self._is_first_packet:
            self.first_packet(pkt)
            return

        self._update_flow_bulk(pkt)
        self._detect_update_subflows(pkt)
        self._check_flags(pkt)

        now = pkt.timestamp
        self.all_pkts.append(pkt)

        if pkt.direction == 0:
            # Forward packet
            if pkt.payload_len >= 1:
                self.act_data_pkt_fwd += 1
            self.f_header_bytes += pkt.header_length
            self.fwd_pkts.append(pkt)
            self.forward_bytes += pkt.payload_len
            self.min_seg_size_fwd = min(pkt.header_length, self.min_seg_size_fwd) if self.min_seg_size_fwd > 0 else pkt.header_length
            self.fwd_last_seen = now
        else:
            # Backward packet
            # CICFlowMeter BUG: overwrites Init_Win_bytes_backward with EVERY backward packet
            self.init_win_bwd = pkt.tcp_window
            self.b_header_bytes += pkt.header_length
            self.bwd_pkts.append(pkt)
            self.backward_bytes += pkt.payload_len
            self.bwd_last_seen = now

        self.last_seen = now

    def is_expired_absolute(self, now: float) -> bool:
        """CICFlowMeter: absolute timeout from flow start."""
        return (now - self.start_time) > FLOW_TIMEOUT

    def is_expired_idle(self, now: float) -> bool:
        """Idle timeout for cleanup sweeps."""
        return (now - self.last_seen) > IDLE_CLEANUP_TIMEOUT

    # ── Bulk feature getters (match CICFlowMeter Java integer division) ──

    def f_avg_bytes_per_bulk(self) -> int:
        if self.fbulk_state_count == 0:
            return 0
        return self.fbulk_size_total // self.fbulk_state_count

    def f_avg_packets_per_bulk(self) -> int:
        if self.fbulk_state_count == 0:
            return 0
        return self.fbulk_packet_count // self.fbulk_state_count

    def f_avg_bulk_rate(self) -> int:
        dur_s = self.fbulk_duration
        if dur_s == 0:
            return 0
        return int(self.fbulk_size_total / dur_s)

    def b_avg_bytes_per_bulk(self) -> int:
        if self.bbulk_state_count == 0:
            return 0
        return self.bbulk_size_total // self.bbulk_state_count

    def b_avg_packets_per_bulk(self) -> int:
        if self.bbulk_state_count == 0:
            return 0
        return self.bbulk_packet_count // self.bbulk_state_count

    def b_avg_bulk_rate(self) -> int:
        dur_s = self.bbulk_duration
        if dur_s == 0:
            return 0
        return int(self.bbulk_size_total / dur_s)

    # ── Subflow feature getters (match CICFlowMeter Java integer division) ──

    def sflow_fpackets(self) -> int:
        if self.sf_count <= 0:
            return 0
        return len(self.fwd_pkts) // self.sf_count

    def sflow_fbytes(self) -> int:
        if self.sf_count <= 0:
            return 0
        return self.forward_bytes // self.sf_count

    def sflow_bpackets(self) -> int:
        if self.sf_count <= 0:
            return 0
        return len(self.bwd_pkts) // self.sf_count

    def sflow_bbytes(self) -> int:
        if self.sf_count <= 0:
            return 0
        return self.backward_bytes // self.sf_count


# ═════════════════════════════════════════════════════════════════════════════
#  Statistics Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _stats(values: List[float]) -> Tuple[float, float, float, float]:
    """Return (mean, std, max, min). Uses SAMPLE std (ddof=1) matching
    Apache Commons Math SummaryStatistics.getStandardDeviation().
    All zeros if empty."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    # CICFlowMeter uses Apache SummaryStatistics which computes sample std (ddof=1)
    # For a single value, std=0 (ddof=1 would give NaN, but SummaryStatistics returns 0)
    if len(arr) == 1:
        std = 0.0
    else:
        std = float(arr.std(ddof=1))
    return mean, std, float(arr.max()), float(arr.min())


def _variance(values: List[float]) -> float:
    """Sample variance (ddof=1) matching Apache SummaryStatistics.getVariance()."""
    if len(values) <= 1:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.var(ddof=1))


def _iat(pkts: List[PacketInfo]) -> List[float]:
    """Inter-arrival times in SECONDS (converted to μs during feature build)."""
    if len(pkts) < 2:
        return []
    return [pkts[i].timestamp - pkts[i - 1].timestamp for i in range(1, len(pkts))]


def _flag_count(pkts: List[PacketInfo], flag_bit: int) -> int:
    return sum(1 for p in pkts if p.tcp_flags & flag_bit)


# ═════════════════════════════════════════════════════════════════════════════
#  Feature Extraction → snake_case CIC-IDS2017 DataFrame
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(flow: FlowRecord) -> Optional[pd.DataFrame]:
    """Extract CIC-IDS2017–compatible features from a completed flow.

    Returns a 1-row DataFrame with snake_case column names that match
    infer.py's expected schema EXACTLY, or None if the flow should be skipped.

    Faithfully replicates CICFlowMeter (BasicFlow.java) including known bugs.
    """
    fp = flow.fwd_pkts
    bp = flow.bwd_pkts
    ap = flow.all_pkts

    if not ap:
        return None

    # Fix 9: CICFlowMeter skips flows with ≤1 packet
    if len(ap) <= 1:
        return None

    # Filter out micro-flows (failed handshakes: SYN+RST with 0 payload)
    # These produce all-zero features that confuse OSPREY into OOD rejection
    total_payload = flow.forward_bytes + flow.backward_bytes
    if len(ap) <= MICRO_FLOW_MAX_PKTS and total_payload <= MICRO_FLOW_MAX_BYTES:
        return None

    # ── Duration (μs for CIC-IDS2017) ─────────────────────────
    duration_s  = max(flow.last_seen - flow.start_time, 1e-9)
    duration_us = duration_s * _US

    # ── Packet lengths — PAYLOAD ONLY (Fix 1) ─────────────────
    # CICFlowMeter: fwdPktStats.addValue((double)packet.getPayloadBytes())
    fwd_lens = [p.payload_len for p in fp]
    bwd_lens = [p.payload_len for p in bp]

    # Fix 10: flowLengthStats double-counts the first packet
    # CICFlowMeter: firstPacket() adds to flowLengthStats once (line 133),
    # then again inside the if-forward/else-backward block (lines 146/161)
    all_lens = [p.payload_len for p in ap]
    if len(ap) > 0:
        all_lens.insert(0, ap[0].payload_len)  # duplicate first packet's payload

    fwd_len_mean, fwd_len_std, fwd_len_max, fwd_len_min = _stats(fwd_lens)
    bwd_len_mean, bwd_len_std, bwd_len_max, bwd_len_min = _stats(bwd_lens)
    all_len_mean, all_len_std, all_len_max, all_len_min = _stats(all_lens)
    all_len_variance = _variance(all_lens)

    # Total bytes = payload only (CICFlowMeter: forwardBytes, backwardBytes)
    total_fwd_bytes = flow.forward_bytes
    total_bwd_bytes = flow.backward_bytes
    total_bytes     = total_fwd_bytes + total_bwd_bytes
    total_pkts      = len(fp) + len(bp)

    # ── IAT (compute in seconds, store in μs) ─────────────────
    flow_iat_s = _iat(ap)
    fwd_iat_s  = _iat(fp)
    bwd_iat_s  = _iat(bp)

    flow_iat_us = [v * _US for v in flow_iat_s]
    fwd_iat_us  = [v * _US for v in fwd_iat_s]
    bwd_iat_us  = [v * _US for v in bwd_iat_s]

    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _stats(flow_iat_us)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _stats(fwd_iat_us)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _stats(bwd_iat_us)

    # CICFlowMeter: forwardIAT.getSum() returns 0 if forward.size() <= 1
    fwd_iat_total = sum(fwd_iat_us) if len(fp) > 1 else 0.0
    bwd_iat_total = sum(bwd_iat_us) if len(bp) > 1 else 0.0

    # Zero out directional IAT stats when only 1 packet in that direction
    if len(fp) <= 1:
        fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0.0
    if len(bp) <= 1:
        bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0.0

    # ── TCP Flags ─────────────────────────────────────────────
    FIN, SYN, RST, PSH = 0x01, 0x02, 0x04, 0x08
    ACK, URG, ECE, CWE = 0x10, 0x20, 0x40, 0x80

    fin_cnt = _flag_count(ap, FIN)
    syn_cnt = _flag_count(ap, SYN)
    rst_cnt = _flag_count(ap, RST)
    psh_cnt = _flag_count(ap, PSH)
    ack_cnt = _flag_count(ap, ACK)
    urg_cnt = _flag_count(ap, URG)
    ece_cnt = _flag_count(ap, ECE)
    cwe_cnt = _flag_count(ap, CWE)

    # Per-direction PSH/URG from flow-level tracking
    fwd_psh = flow.f_psh_cnt
    fwd_urg = flow.f_urg_cnt

    # ── Header lengths (CICFlowMeter: fHeaderBytes, bHeaderBytes) ──
    fwd_header_total = flow.f_header_bytes
    bwd_header_total = flow.b_header_bytes

    # ── Packets/s — CICFlowMeter divides by (duration_us / 1e6) = duration_s
    flow_bytes_s   = _safe_div(total_bytes, duration_s)
    flow_packets_s = _safe_div(total_pkts, duration_s)
    fwd_packets_s  = _safe_div(len(fp), duration_s)
    bwd_packets_s  = _safe_div(len(bp), duration_s)

    # ── Down/Up Ratio — CICFlowMeter: Java integer division
    down_up_ratio = int(len(bp) // len(fp)) if len(fp) > 0 else 0

    # ── Average packet size — uses flowLengthStats.getSum() / packetCount()
    # flowLengthStats has the first packet double-counted, so sum(all_lens) includes the dup
    avg_packet_size = _safe_div(sum(all_lens), total_pkts) if total_pkts > 0 else 0.0

    # ── Avg segment size — uses fwdPktStats/bwdPktStats (NOT flowLengthStats)
    avg_fwd_seg = _safe_div(sum(fwd_lens), len(fp)) if len(fp) > 0 else 0.0
    avg_bwd_seg = _safe_div(sum(bwd_lens), len(bp)) if len(bp) > 0 else 0.0

    # ── Active / Idle (μs) — finalize current active period ──
    # CICFlowMeter: endActiveIdleTime() adds the final active period
    active_periods_us = []
    idle_periods_us = []

    # Add completed periods
    for a in flow.active_periods:
        active_periods_us.append(a * _US)
    for i in flow.idle_periods:
        idle_periods_us.append(i * _US)

    # Finalize: add the last active period if non-zero
    final_active = flow.end_active_time - flow.start_active_time
    if final_active > 0:
        active_periods_us.append(final_active * _US)

    act_mean, act_std, act_max, act_min = _stats(active_periods_us)
    idl_mean, idl_std, idl_max, idl_min = _stats(idle_periods_us)

    # ── Subflow features (Fix 6) — total / sfCount ──
    sf_count = flow.sf_count
    if sf_count <= 0:
        # No subflows detected — raw totals (same as sfCount=1 but Java returns 0)
        subflow_fwd_packets = 0
        subflow_fwd_bytes   = 0
        subflow_bwd_packets = 0
        subflow_bwd_bytes   = 0
    else:
        subflow_fwd_packets = flow.sflow_fpackets()
        subflow_fwd_bytes   = flow.sflow_fbytes()
        subflow_bwd_packets = flow.sflow_bpackets()
        subflow_bwd_bytes   = flow.sflow_bbytes()

    # ── act_data_pkt_fwd (Fix 11) — already tracked in add_packet (excludes first pkt)
    act_data_fwd = flow.act_data_pkt_fwd

    # ── min_seg_size_forward — min header length across all forward packets
    min_seg_fwd = flow.min_seg_size_fwd

    # ── Build dict with EXACT snake_case names ───────────────
    row = {
        "destination_port":              flow.dst_port,
        "flow_duration":                 duration_us,
        "total_fwd_packets":             len(fp),
        "total_backward_packets":        len(bp),
        "total_length_of_fwd_packets":   total_fwd_bytes,
        "total_length_of_bwd_packets":   total_bwd_bytes,
        "fwd_packet_length_max":         fwd_len_max,
        "fwd_packet_length_min":         fwd_len_min,
        "fwd_packet_length_mean":        fwd_len_mean,
        "fwd_packet_length_std":         fwd_len_std,
        "bwd_packet_length_max":         bwd_len_max,
        "bwd_packet_length_min":         bwd_len_min,
        "bwd_packet_length_mean":        bwd_len_mean,
        "bwd_packet_length_std":         bwd_len_std,
        "flow_bytes_s":                  flow_bytes_s,
        "flow_packets_s":                flow_packets_s,
        "flow_iat_mean":                 flow_iat_mean,
        "flow_iat_std":                  flow_iat_std,
        "flow_iat_max":                  flow_iat_max,
        "flow_iat_min":                  flow_iat_min,
        "fwd_iat_total":                 fwd_iat_total,
        "fwd_iat_mean":                  fwd_iat_mean,
        "fwd_iat_std":                   fwd_iat_std,
        "fwd_iat_max":                   fwd_iat_max,
        "fwd_iat_min":                   fwd_iat_min,
        "bwd_iat_total":                 bwd_iat_total,
        "bwd_iat_mean":                  bwd_iat_mean,
        "bwd_iat_std":                   bwd_iat_std,
        "bwd_iat_max":                   bwd_iat_max,
        "bwd_iat_min":                   bwd_iat_min,
        "fwd_psh_flags":                 fwd_psh,
        "fwd_urg_flags":                 fwd_urg,
        "fwd_header_length":             fwd_header_total,
        "bwd_header_length":             bwd_header_total,
        "fwd_packets_s":                 fwd_packets_s,
        "bwd_packets_s":                 bwd_packets_s,
        "min_packet_length":             all_len_min,
        "max_packet_length":             all_len_max,
        "packet_length_mean":            all_len_mean,
        "packet_length_std":             all_len_std,
        "packet_length_variance":        all_len_variance,
        "fin_flag_count":                fin_cnt,
        "syn_flag_count":                syn_cnt,
        "rst_flag_count":                rst_cnt,
        "psh_flag_count":                psh_cnt,
        "ack_flag_count":                ack_cnt,
        "urg_flag_count":                urg_cnt,
        "cwe_flag_count":                cwe_cnt,
        "ece_flag_count":                ece_cnt,
        "down_up_ratio":                 down_up_ratio,
        "average_packet_size":           avg_packet_size,
        "avg_fwd_segment_size":          avg_fwd_seg,
        "avg_bwd_segment_size":          avg_bwd_seg,
        "fwd_avg_bytes_bulk":            flow.f_avg_bytes_per_bulk(),
        "fwd_avg_packets_bulk":          flow.f_avg_packets_per_bulk(),
        "fwd_avg_bulk_rate":             flow.f_avg_bulk_rate(),
        "bwd_avg_bytes_bulk":            flow.b_avg_bytes_per_bulk(),
        "bwd_avg_packets_bulk":          flow.b_avg_packets_per_bulk(),
        "bwd_avg_bulk_rate":             flow.b_avg_bulk_rate(),
        "subflow_fwd_packets":           subflow_fwd_packets,
        "subflow_fwd_bytes":             subflow_fwd_bytes,
        "subflow_bwd_packets":           subflow_bwd_packets,
        "subflow_bwd_bytes":             subflow_bwd_bytes,
        "init_win_bytes_forward":        flow.init_win_fwd,
        "init_win_bytes_backward":       flow.init_win_bwd,
        "act_data_pkt_fwd":              act_data_fwd,
        "min_seg_size_forward":          max(min_seg_fwd, 0),
        "active_mean":                   act_mean,
        "active_std":                    act_std,
        "active_max":                    act_max,
        "active_min":                    act_min,
        "idle_mean":                     idl_mean,
        "idle_std":                      idl_std,
        "idle_max":                      idl_max,
        "idle_min":                      idl_min,
    }

    return pd.DataFrame([row]).fillna(0).replace([np.inf, -np.inf], 0)


# ═════════════════════════════════════════════════════════════════════════════
#  Inference Engine — Cascaded Pipeline (DAEMON → OSPREY)
# ═════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Loads the model bundle and runs cascaded DAEMON→OSPREY inference.

    Pipeline:
        Stage 1 (DAEMON): Binary anomaly detection
            BENIGN     → stop, return 🟢 BENIGN
            BORDERLINE → forward to Stage 2 for second opinion
            ATTACK     → forward to Stage 2 for classification
        Stage 2 (OSPREY): Multi-class classification + OOD rejection
            Known class  → 🔴 <attack type>
            OOD rejected → ⚠ UNKNOWN ATTACK
    """

    def __init__(self, bundle_path: str):
        if not _HAS_INFER:
            raise RuntimeError("infer.py not importable — cannot load models")

        self.bundle = load_bundle(bundle_path)
        b = self.bundle

        # Build DAEMON
        self.daemon_model = DualPathAutoencoder(
            b["daemon_feature_info"],
            b["daemon_bottleneck_dim"],
            b["daemon_dropout"],
        )
        self.daemon_model.load_state_dict(b["daemon_state_dict"])
        self.daemon_model.eval()

        # Build OSPREY
        self.osprey_model = OSPREY(b["osprey_config"])
        self.osprey_model.load_state_dict(b["osprey_state_dict"])
        self.osprey_model.eval()

        dp = sum(p.numel() for p in self.daemon_model.parameters())
        op = sum(p.numel() for p in self.osprey_model.parameters())
        thr = b["osprey_thresholds"]

        # ── Deployment-tuned DAEMON threshold override ──────────────────
        # Original τ=0.3028 was calibrated on CIC-IDS2017 benign traffic only.
        # Live traffic analysis shows: NTP scores 1.2-1.3, DNS 1.5-2.7,
        # DHCP 7-13, real attacks 30+.  τ=8.0 filters benign system traffic
        # (NTP/DNS) while catching low-score attacks (GoldenEye, hping3).
        _DAEMON_OVERRIDE = 8.0
        _original_tau = b['daemon_threshold']
        b['daemon_threshold'] = _DAEMON_OVERRIDE
        log.info(f"DAEMON loaded ({dp:,} params)  τ = {_DAEMON_OVERRIDE:.4f} "
                 f"(overridden from {_original_tau:.4f} for deployment)")
        log.info(f"OSPREY loaded ({op:,} params)  "
                 f"E={thr['energy']:.2f}  H={thr['entropy']:.4f}  cos={thr['cosine']:.4f}")

        # Post-classification: DDoS aggregator only
        if _HAS_POSTCLASS:
            self.ddos_aggregator = DDoSAggregator(window_sec=120, min_sources=3)
            log.info("Post-classification loaded: DDoSAggregator")
        else:
            self.ddos_aggregator = None

        # Track known attack directions: (attacker → target) pairs.
        # When we see the REVERSE direction (target → attacker), that's
        # just response traffic — skip PortScan detection for it.
        # This prevents server response traffic (which hits many ports)
        # from being misclassified as Port Scan.


    def classify(self, raw_df: pd.DataFrame,
                 src_ip: str = "", dst_ip: str = "",
                 dst_port: int = 0) -> Optional[dict]:
        """Run cascaded DAEMON→OSPREY pipeline on a 1-row CIC-IDS2017 DataFrame.

        Post-classification rules (PortScan, DDoS) are applied after OSPREY.

        Returns cascade verdict dict or None on error.
        """
        try:
            b = self.bundle

            # 1. Feature engineering (log1p + 8 ratio features)
            eng_df = engineer_features(
                raw_df, b["log_transform_features"], b["ratio_cols"])

            # ── Stage 1: DAEMON ──────────────────────────────────────
            dc = b["daemon_feature_info"]["all_features"]
            for col in dc:
                if col not in eng_df.columns:
                    eng_df[col] = 0.0
            daemon_X = np.clip(
                b["daemon_scaler"].transform(
                    eng_df[dc].values.astype(np.float32)),
                -5, 5,
            ).astype(np.float32)

            d_res = run_daemon(
                self.daemon_model, daemon_X,
                b["daemon_feature_info"],
                b["daemon_scorer_stats"],
                b["daemon_composite_weights"],
                b["daemon_threshold"],
            )

            daemon_verdict = d_res[0]["daemon_verdict"]
            daemon_score = d_res[0]["daemon_score"]

            log.debug(f"DAEMON: verdict={daemon_verdict}  score={daemon_score:.4f}")

            # ── Stage 2: OSPREY (only for ATTACK + BORDERLINE) ───────
            osprey_results_map = {}

            if daemon_verdict in ("ATTACK", "BORDERLINE"):
                of = b["osprey_feature_names"]
                osprey_expected = b["osprey_scaler"].feature_names_in_

                for col in osprey_expected:
                    if col not in eng_df.columns:
                        eng_df[col] = 0.0

                oX_df = eng_df[osprey_expected].copy()
                oX_scaled = pd.DataFrame(
                    b["osprey_scaler"].transform(oX_df),
                    columns=osprey_expected, index=oX_df.index,
                ).astype(np.float32)

                for col in of:
                    if col not in oX_scaled.columns:
                        oX_scaled[col] = 0.0
                oX = oX_scaled[of].values

                o_res = run_osprey(
                    self.osprey_model, oX,
                    b["osprey_thresholds"],
                    b["osprey_label_encoder"],
                )

                osprey_results_map[0] = o_res[0]
                log.debug(f"OSPREY: class={o_res[0]['osprey_class']}  "
                          f"energy={o_res[0]['osprey_energy']:.2f}  "
                          f"ood_count={o_res[0]['osprey_ood_count']}")

            # ── Cascade verdict assembly ─────────────────────────────
            fused = cascade_verdict(d_res, osprey_results_map)
            result = fused[0] if fused else None

            if result:
                log.debug(f"CASCADE: {result.get('cascade_verdict', '?')}")

                # ── Post-classification: DDoS aggregator only ────────────
                original_class = result.get("cascade_class", "")

                if self.ddos_aggregator and src_ip and dst_ip:
                    new_class = self.ddos_aggregator.classify(
                        src_ip, dst_ip, original_class)
                    if new_class != original_class:
                        result["cascade_class"] = new_class
                        result["cascade_verdict"] = f"ATTACK: {new_class}"
                        result["postclass_override"] = f"DDoS rule: {original_class} -> {new_class}"
                        log.info(f"  POST-CLASS: DDoS upgrade {original_class} -> {new_class}")



            return result

        except Exception as e:
            log.error(f"Classification error: {e}", exc_info=True)
            return None


# ═════════════════════════════════════════════════════════════════════════════
#  Forensic PCAP + CSV Store (SOC Evidence Collection)
# ═════════════════════════════════════════════════════════════════════════════

class ForensicStore:
    """Stores attack PCAPs and flow CSVs for SOC forensic analysis.

    Uses single rolling files per day — each new attack is APPENDED:

        <forensics_dir>/
            YYYY-MM-DD/
                known_attacks.pcap       ← all known attack packets (append)
                known_attacks.csv        ← all known attack flows   (append)
                unknown_attacks.pcap     ← all unknown attack packets
                unknown_attacks.csv      ← all unknown attack flows

    SOC analysts can open these files in Wireshark / Excel at any time,
    even while the engine is still running and appending new attacks.
    """

    def __init__(self, forensics_dir: str):
        self.base_dir = os.path.abspath(forensics_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self._saved_count = 0
        self._csv_headers_written: set = set()  # track which CSVs have headers
        log.info(f"  Forensic store: {self.base_dir}")

    def _get_paths(self, flow_time: float, is_unknown: bool) -> tuple:
        """Get the PCAP and CSV paths for today's rolling files."""
        date_str = datetime.fromtimestamp(flow_time).strftime("%Y-%m-%d")
        day_dir = os.path.join(self.base_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)

        tag = "unknown_attacks" if is_unknown else "known_attacks"
        pcap_path = os.path.join(day_dir, f"{tag}.pcap")
        csv_path = os.path.join(day_dir, f"{tag}.csv")
        return pcap_path, csv_path

    def save_attack(self, flow: 'FlowRecord', features_df: pd.DataFrame,
                    result: dict, reason: str = "") -> Optional[str]:
        """Append attack packets to rolling PCAP + CSV. Returns PCAP path."""
        if not flow.raw_packets:
            log.debug("  Forensic: no raw packets buffered, skipping PCAP")
            return None

        cascade_class = result.get("cascade_class", "UNKNOWN")
        is_unknown = "UNKNOWN" in result.get("cascade_verdict", "")
        pcap_path, csv_path = self._get_paths(flow.start_time, is_unknown)

        try:
            # Append packets to rolling PCAP
            from scapy.utils import wrpcap
            wrpcap(pcap_path, flow.raw_packets, append=True)

            # Build CSV row: metadata + verdict + all features
            meta = {
                "timestamp": datetime.fromtimestamp(flow.start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"),
                "src_ip": flow.src_ip, "dst_ip": flow.dst_ip,
                "src_port": flow.src_port, "dst_port": flow.dst_port,
                "protocol": flow.proto,
                "total_packets": len(flow.all_pkts),
                "fwd_packets": len(flow.fwd_pkts),
                "bwd_packets": len(flow.bwd_pkts),
                "duration_sec": round(flow.last_seen - flow.start_time, 3),
                "cascade_class": cascade_class,
                "cascade_verdict": result.get("cascade_verdict", ""),
                "cascade_stage": result.get("cascade_stage", ""),
                "daemon_verdict": result.get("daemon_verdict", ""),
                "daemon_score": result.get("daemon_score", ""),
                "osprey_class": result.get("osprey_class", ""),
                "osprey_energy": result.get("osprey_energy", ""),
                "osprey_entropy": result.get("osprey_entropy", ""),
                "osprey_max_cos": result.get("osprey_max_cos", ""),
                "osprey_ood_count": result.get("osprey_ood_count", ""),
                "emit_reason": reason,
            }
            feat_dict = features_df.iloc[0].to_dict()
            row = {**meta, **feat_dict}

            # Append row to rolling CSV (write header only on first row)
            write_header = csv_path not in self._csv_headers_written
            if write_header and os.path.exists(csv_path):
                # CSV already exists from previous session — don't duplicate header
                write_header = os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    self._csv_headers_written.add(csv_path)
                writer.writerow(row)

            self._saved_count += 1
            n_pkts = len(flow.raw_packets)
            size_kb = os.path.getsize(pcap_path) / 1024
            log.info(f"  FORENSIC #{self._saved_count}: +{n_pkts} pkts "
                     f"({size_kb:.1f} KB total) -> {pcap_path}")
            return pcap_path

        except Exception as e:
            log.error(f"  FORENSIC: write error: {e}")
            return None

    @property
    def saved_count(self) -> int:
        return self._saved_count


# ═════════════════════════════════════════════════════════════════════════════
#  Local IP detection (to filter self-traffic)
# ═════════════════════════════════════════════════════════════════════════════

def _get_local_ips() -> set:
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
        out = subprocess.check_output(
            ["ip", "-o", "addr", "show"], text=True, stderr=subprocess.DEVNULL)
        for line in out.strip().splitlines():
            parts = line.split()
            for i, p in enumerate(parts):
                if p in ("inet", "inet6"):
                    local.add(parts[i + 1].split("/")[0])
    except Exception:
        pass
    return local


# ═════════════════════════════════════════════════════════════════════════════
#  Real-Time Flow Manager
# ═════════════════════════════════════════════════════════════════════════════

class RealTimeCICFlow:
    """Captures live packets, assembles flows, extracts CIC-IDS2017 features,
    and classifies each completed flow through OSPREY + DAEMON."""

    def __init__(self, interface: str,
                 output_csv: Optional[str] = None,
                 model_path: Optional[str] = None,
                 flush_interval: float = 2.0,
                 include_self_traffic: bool = False,
                 forensics_dir: Optional[str] = None):

        self.interface = interface
        self.output_csv = output_csv
        self.flush_interval = flush_interval
        self.include_self = include_self_traffic
        self.flows: Dict[FlowKey, FlowRecord] = {}
        self.lock = threading.Lock()
        self.pkt_queue: queue.Queue = queue.Queue(maxsize=50_000)
        self.running = False

        # Stats
        self._flow_count = 0
        self._benign = 0
        self._attacks = 0
        self._zeroday = 0
        self._start_time = 0.0

        # Local IPs for self-traffic filtering
        self._local_ips = _get_local_ips()

        # Load inference engine
        self.engine: Optional[InferenceEngine] = None
        if model_path and _HAS_INFER:
            try:
                self.engine = InferenceEngine(model_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                traceback.print_exc()
        elif model_path and not _HAS_INFER:
            print("[WARN] --model specified but infer.py not importable")

        # CSV writer
        self._csv_file = None
        self._csv_writer = None
        if output_csv:
            self._csv_file = open(output_csv, "w", newline="")
            self._csv_header_written = False
            print(f"  [✓] Output CSV: {output_csv}")

        # Forensic PCAP store
        self.forensics: Optional[ForensicStore] = None
        self._forensics_enabled = forensics_dir is not None
        if forensics_dir:
            self.forensics = ForensicStore(forensics_dir)

    # ── Packet Callback (Scapy sniff thread) ──────────────────
    def _packet_callback(self, pkt):
        try:
            self.pkt_queue.put_nowait(pkt)
        except queue.Full:
            pass  # drop rather than block capture

    # ── Packet Processing Thread ──────────────────────────────
    def _process_packets(self):
        while self.running:
            try:
                pkt = self.pkt_queue.get(timeout=0.1)
                self._handle_packet(pkt)
            except queue.Empty:
                continue

    def _handle_packet(self, pkt):
        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        now = float(pkt.time)
        proto = ip.proto

        if proto == 6 and pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            flags    = int(tcp.flags)
            win_size = tcp.window
            ip_hdr_len  = ip.ihl * 4
            tcp_hdr_len = tcp.dataofs * 4
            header_len  = ip_hdr_len + tcp_hdr_len
            payload_len = len(bytes(tcp.payload))
            pkt_len     = len(pkt[IP])

        elif proto == 17 and pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport
            flags    = 0
            win_size = 0
            ip_hdr_len = ip.ihl * 4
            header_len  = ip_hdr_len + 8  # UDP header = 8 bytes
            payload_len = len(bytes(udp.payload))
            pkt_len     = len(pkt[IP])

        else:
            return  # skip non-TCP/UDP

        src_ip = ip.src
        dst_ip = ip.dst
        raw_pkt = pkt if self._forensics_enabled else None

        # ── Self-traffic filter ───────────────────────────────
        if not self.include_self:
            if src_ip in self._local_ips and dst_ip in self._local_ips:
                return
            # Skip multicast/broadcast
            if dst_ip.startswith("224.") or dst_ip.startswith("239."):
                return
            if dst_ip == "255.255.255.255":
                return
            if dst_ip.startswith("ff") or src_ip.startswith("ff"):
                return

        fwd_key = _make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = _make_flow_key(dst_ip, src_ip, dst_port, src_port, proto)

        with self.lock:
            if fwd_key in self.flows:
                key = fwd_key
                direction = 0
                flow = self.flows[key]
            elif bwd_key in self.flows:
                key = bwd_key
                direction = 1
                flow = self.flows[key]
            else:
                key = fwd_key
                direction = 0
                flow = FlowRecord(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    proto=proto, start_time=now,
                    last_seen=now,
                )
                self.flows[key] = flow

            pinfo = PacketInfo(
                timestamp=now, ip_length=pkt_len,
                header_length=header_len,
                direction=direction, tcp_flags=flags,
                tcp_window=win_size, payload_len=payload_len,
            )

            # Fix 4: Check absolute timeout from flow start (CICFlowMeter: flowTimeOut)
            if flow.is_expired_absolute(now):
                # CICFlowMeter: export current flow if >1 packet, start new one
                if len(flow.all_pkts) > 1:
                    self._emit_flow(flow, reason="TIMEOUT")
                # Create new flow with this packet
                new_flow = FlowRecord(
                    src_ip=flow.src_ip, dst_ip=flow.dst_ip,
                    src_port=flow.src_port, dst_port=flow.dst_port,
                    proto=flow.proto, start_time=now,
                    last_seen=now,
                )
                new_flow.add_packet(pinfo)
                self.flows[key] = new_flow
                return

            # Fix 3: FIN dual-direction termination (CICFlowMeter FlowGenerator.java)
            if proto == 6 and (flags & 0x01):  # FIN flag
                if direction == 0:
                    # Forward FIN
                    flow.fwd_fin_cnt += 1
                    if flow.fwd_fin_cnt == 1:
                        if (flow.fwd_fin_cnt + flow.bwd_fin_cnt) >= 2:
                            # Both directions have sent FIN → finish
                            flow.add_packet(pinfo)
                            finished = self.flows.pop(key)
                            self._emit_flow(finished, reason="FIN")
                            return
                        else:
                            # Only forward FIN so far → keep flow open
                            flow.add_packet(pinfo)
                            return
                else:
                    # Backward FIN
                    flow.bwd_fin_cnt += 1
                    if flow.bwd_fin_cnt == 1:
                        if (flow.fwd_fin_cnt + flow.bwd_fin_cnt) >= 2:
                            # Both directions have sent FIN → finish
                            flow.add_packet(pinfo)
                            finished = self.flows.pop(key)
                            self._emit_flow(finished, reason="FIN")
                            return
                        else:
                            # Only backward FIN so far → keep flow open
                            flow.add_packet(pinfo)
                            return

            # RST → immediately export (CICFlowMeter: hasFlagRST → export)
            if proto == 6 and (flags & 0x04):  # RST flag
                flow.add_packet(pinfo)
                finished = self.flows.pop(key)
                self._emit_flow(finished, reason="RST")
                return

            # Normal packet — check if flow is already FIN-closed
            if proto == 6:
                if direction == 0 and flow.fwd_fin_cnt > 0:
                    # Forward already sent FIN, don't add more forward packets
                    # CICFlowMeter: only adds if fwdFIN == 0
                    return
                if direction == 1 and flow.bwd_fin_cnt > 0:
                    # Backward already sent FIN
                    return

            # Regular packet — add and update active/idle
            flow.add_packet(pinfo)

            # Buffer raw packet for forensic PCAP export
            if raw_pkt is not None:
                flow.raw_packets.append(raw_pkt)

    # ── Timeout Flush Thread ──────────────────────────────────
    def _flush_expired(self):
        while self.running:
            time.sleep(self.flush_interval)
            now = time.time()
            expired = []
            snapshot_candidates = []

            with self.lock:
                for key, flow in self.flows.items():
                    # Check both absolute timeout AND idle timeout
                    if flow.is_expired_absolute(now) or flow.is_expired_idle(now):
                        expired.append(key)
                    # Periodic snapshot for long-lived flows (Slowloris, Botnet, etc.)
                    # Classify the flow WITHOUT removing it from the table
                    elif self._should_snapshot(flow, now):
                        snapshot_candidates.append(key)

                for key in expired:
                    flow = self.flows.pop(key)
                    self._emit_flow(flow, reason="TIMEOUT")

                # Emit snapshots (classify-in-place, don't remove from flow table)
                for key in snapshot_candidates:
                    flow = self.flows[key]
                    flow._last_snapshot_time = now
                    flow._snapshot_count += 1
                    self._emit_flow(flow, reason=f"SNAPSHOT#{flow._snapshot_count}")

    def _should_snapshot(self, flow: FlowRecord, now: float) -> bool:
        """Check if a long-lived flow should be periodically classified.

        This catches slow attacks (Slowloris, Botnet C2, Infiltration) that
        would otherwise only be classified on FIN/RST/timeout, by which time
        the connection has been open for 120+ seconds.
        """
        age = now - flow.start_time
        if age < SNAPSHOT_MIN_AGE:
            return False
        if len(flow.all_pkts) < SNAPSHOT_MIN_PKTS:
            return False
        # Check if enough time since last snapshot (or first snapshot)
        since_last = now - flow._last_snapshot_time if flow._last_snapshot_time > 0 else age
        return since_last >= SNAPSHOT_INTERVAL

    # ── Emit Completed Flow ───────────────────────────────────
    def _emit_flow(self, flow: FlowRecord, reason: str = "COMPLETE"):
        # ── RST flow quality gate ────────────────────────────────────
        # RST-terminated flows with <4 packets produce degenerate features
        # (near-zero IAT, no payload stats) → 26.4% accuracy vs 95.3% for FIN.
        # Skip them to avoid flooding the output with garbage predictions.
        _RST_MIN_PACKETS = 4
        if reason == "RST" and len(flow.all_pkts) < _RST_MIN_PACKETS:
            log.debug(f"SKIP low-quality RST flow {flow.src_ip}:{flow.src_port} -> "
                      f"{flow.dst_ip}:{flow.dst_port} ({len(flow.all_pkts)} pkts < {_RST_MIN_PACKETS})")
            return

        # 1. Extract CIC-IDS2017 features as a properly-named DataFrame
        features_df = extract_features(flow)
        if features_df is None:
            if reason == "RST" and len(flow.all_pkts) <= MICRO_FLOW_MAX_PKTS:
                log.debug(f"SKIP micro-flow {flow.src_ip}:{flow.src_port} -> "
                          f"{flow.dst_ip}:{flow.dst_port} (RST, {len(flow.all_pkts)} pkts, "
                          f"{flow.forward_bytes + flow.backward_bytes} bytes)")
            return

        self._flow_count += 1
        ts = datetime.fromtimestamp(flow.start_time).strftime("%Y-%m-%d %H:%M:%S")
        proto_str = "TCP" if flow.proto == 6 else "UDP" if flow.proto == 17 else str(flow.proto)
        n_pkts = len(flow.all_pkts)
        dur_s = flow.last_seen - flow.start_time

        flow_id = f"#{self._flow_count:<5d}"
        flow_desc = (f"{flow.src_ip}:{flow.src_port} -> "
                     f"{flow.dst_ip}:{flow.dst_port} [{proto_str}]")

        # Log raw feature snapshot (DEBUG-level for remote debugging)
        log.debug(f"FLOW {flow_id.strip()} {flow_desc}  "
                  f"pkts={n_pkts} fwd={len(flow.fwd_pkts)} bwd={len(flow.bwd_pkts)} "
                  f"dur={dur_s:.3f}s  "
                  f"fwd_bytes={flow.forward_bytes} bwd_bytes={flow.backward_bytes}  "
                  f"init_win_fwd={flow.init_win_fwd} init_win_bwd={flow.init_win_bwd}  "
                  f"sf_count={flow.sf_count}  reason={reason}")

        # 2. Classify through DAEMON → OSPREY cascade
        result: Optional[dict] = None
        if self.engine:
            result = self.engine.classify(
                features_df,
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                dst_port=flow.dst_port,
            )

        # 2b. Outbound traffic bypass
        #     If YOUR machine browses Google/GitHub (outbound HTTPS/DNS),
        #     and the model says UNKNOWN (unsure), override to BENIGN.
        #     Never overrides confident attack labels (DoS, BruteForce, etc.)
        #     Never touches inbound traffic (attacks coming TO your server).
        _OUTBOUND_SAFE_PORTS = {443, 53, 123, 8443}
        if result and result.get("cascade_class") == "UNKNOWN":
            if (flow.src_ip in self._local_ips
                    and flow.dst_port in _OUTBOUND_SAFE_PORTS):
                result["cascade_verdict"] = "\U0001f7e2 BENIGN"
                result["cascade_class"] = "BENIGN"

        # 3. Console + log output
        if result:
            verdict = result.get("cascade_verdict", "?")
            daemon_v = result.get("daemon_verdict", "?")
            daemon_s = result.get("daemon_score", 0.0)
            cascade_stage = result.get("cascade_stage", "?")
            osprey_c = result.get("osprey_class", "—")
            osprey_e = result.get("osprey_energy", float('nan'))

            is_benign = "BENIGN" in verdict
            is_unknown = "UNKNOWN" in verdict

            if is_benign:
                self._benign += 1
            else:
                self._attacks += 1
            if is_unknown:
                self._zeroday += 1

            log.info(f"[{ts}] {flow_id} {flow_desc}  {verdict}")
            log.debug(f"  → Stage={cascade_stage}  DAEMON={daemon_v}({daemon_s:.2f})  "
                      f"OSPREY={osprey_c}(E={osprey_e})")
        else:
            log.info(f"[{ts}] {flow_id} {flow_desc}  (no model — features only)")

        # 4. CSV output
        if self._csv_file:
            meta = {
                "timestamp": ts,
                "src_ip": flow.src_ip, "dst_ip": flow.dst_ip,
                "src_port": flow.src_port, "dst_port": flow.dst_port,
                "protocol": flow.proto,
            }
            # Flatten features DF to dict
            feat_dict = features_df.iloc[0].to_dict()
            row = {**meta, **feat_dict}

            if result:
                row["cascade_verdict"] = result.get("cascade_verdict", "")
                row["cascade_class"]   = result.get("cascade_class", "")
                row["cascade_stage"]   = result.get("cascade_stage", "")
                row["daemon_verdict"]  = result.get("daemon_verdict", "")
                row["daemon_score"]    = result.get("daemon_score", "")
                row["osprey_class"]    = result.get("osprey_class", "")
                row["osprey_energy"]   = result.get("osprey_energy", "")

            row["emit_reason"] = reason

            if not self._csv_header_written:
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()))
                self._csv_writer.writeheader()
                self._csv_header_written = True

            self._csv_writer.writerow(row)
            self._csv_file.flush()
            log.debug(f"  → CSV row written")

        # 5. Forensic PCAP+CSV dump for attacks
        if self.forensics and result:
            is_attack = "BENIGN" not in result.get("cascade_verdict", "")
            if is_attack:
                self.forensics.save_attack(flow, features_df, result, reason)

    # ── Start ─────────────────────────────────────────────────
    def start(self):
        self.running = True
        self._start_time = time.time()

        log.info(f"")
        log.info(f"{'═' * 72}")
        log.info(f"  OSPREY × DAEMON — Real-Time Network Intrusion Detection")
        log.info(f"{'═' * 72}")
        log.info(f"  Interface:      {self.interface}")
        log.info(f"  Flow timeout:   {FLOW_TIMEOUT}s (absolute from start)")
        log.info(f"  Snapshot:       every {SNAPSHOT_INTERVAL}s for long-lived flows")
        log.info(f"  Flush interval: {self.flush_interval}s")
        log.info(f"  Self-traffic:   {'included' if self.include_self else 'filtered'}")
        log.info(f"  Model:          {'loaded ✓' if self.engine else 'none (CSV only)'}")
        log.info(f"  CSV output:     {self.output_csv or 'disabled'}")
        log.info(f"  Forensics:      {self.forensics.base_dir if self.forensics else 'disabled'}")
        log.info(f"{'═' * 72}")
        log.info(f"  Press Ctrl+C to stop")

        # Start worker threads
        proc_thread = threading.Thread(
            target=self._process_packets, daemon=True, name="pkt-processor")
        proc_thread.start()

        flush_thread = threading.Thread(
            target=self._flush_expired, daemon=True, name="flow-flusher")
        flush_thread.start()

        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,
                filter="ip",
            )
        except KeyboardInterrupt:
            log.info("Stopping capture...")
        except PermissionError:
            log.error("Permission denied — run with sudo/root")
        finally:
            self.running = False
            time.sleep(0.3)  # let worker threads drain

            # Flush remaining flows (including still-open Slowloris/long-lived connections)
            with self.lock:
                remaining = list(self.flows.values())
                self.flows.clear()
            log.info(f"  Flushing {len(remaining)} remaining active flows...")
            for flow in remaining:
                self._emit_flow(flow, reason="SHUTDOWN")

            if self._csv_file:
                self._csv_file.close()

            elapsed = time.time() - self._start_time
            log.info(f"")
            log.info(f"{'═' * 72}")
            log.info(f"  SESSION SUMMARY")
            log.info(f"{'═' * 72}")
            log.info(f"  Duration:    {elapsed:.1f}s")
            log.info(f"  Flows:       {self._flow_count:,}")
            if self.engine:
                log.info(f"  Benign:      {self._benign:,}")
                log.info(f"  Attacks:     {self._attacks:,}")
                log.info(f"  Unknown:     {self._zeroday:,}")
            if self.output_csv:
                log.info(f"  CSV saved:   {self.output_csv}")
            if self.forensics:
                log.info(f"  Forensics:   {self.forensics.saved_count} attack PCAPs saved")
                log.info(f"  Forensic dir: {self.forensics.base_dir}")
            log.info(f"{'═' * 72}")
            log.info(f"  Done.")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time CICFlowMeter Feature Extractor — OSPREY × DAEMON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sudo python realtime_cicflow.py -i eth0\n"
            "  sudo python realtime_cicflow.py -i eth0 -o flows.csv\n"
            "  sudo python realtime_cicflow.py -i eth0 -m nids_models.pkl\n"
            "  sudo python realtime_cicflow.py -i eth0 -m nids_models.pkl -o output.csv\n"
            "  sudo python realtime_cicflow.py -i eth0 -m nids_models.pkl --forensics ./evidence\n"
        ),
    )
    parser.add_argument("--interface", "-i", required=True,
                        help="Network interface (e.g. eth0, ens33, wlan0)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file (features + verdicts)")
    parser.add_argument("--model", "-m", default=None,
                        help="Path to nids_models.pkl (auto-detect if omitted)")
    parser.add_argument("--flush-interval", "-f", type=float, default=2.0,
                        help="Seconds between timeout checks (default: 2)")
    parser.add_argument("--self-traffic", action="store_true",
                        help="Include self-traffic (default: filtered out)")
    parser.add_argument("--forensics", default=None, metavar="DIR",
                        help="Enable forensic PCAP+CSV storage for attacks "
                             "(e.g. --forensics ./evidence)")

    args = parser.parse_args()

    # Auto-detect model bundle if not specified
    model_path = args.model
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "nids_models.pkl"),
            os.path.join(os.getcwd(), "nids_models.pkl"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                model_path = c
                print(f"  [✓] Auto-detected model: {model_path}")
                break

    # Root check (Linux only — skip on Windows)
    if sys.platform != "win32":
        if os.geteuid() != 0:
            print("[ERROR] Run with sudo/root for live packet capture.")
            sys.exit(1)

    extractor = RealTimeCICFlow(
        interface=args.interface,
        output_csv=args.output,
        model_path=model_path,
        flush_interval=args.flush_interval,
        include_self_traffic=args.self_traffic,
        forensics_dir=args.forensics,
    )
    extractor.start()


if __name__ == "__main__":
    main()
