"""
DDoS Aggregation Layer — Post-classification fix for DoS/DDoS confusion
Plug this into the inference pipeline AFTER OSPREY classification.

Logic: If OSPREY says "DoS" for a flow, check if multiple unique source IPs
are also flagged as DoS to the same target within a time window.
If yes → upgrade to "DDoS".

In production:
  - Real DoS  = 1 source  → 1 target  → stays "DoS"
  - Real DDoS = N sources → 1 target  → upgraded to "DDoS"
"""
from collections import defaultdict
import time
import threading


class DDoSAggregator:
    """
    Sliding-window aggregator that upgrades DoS → DDoS
    when multiple sources attack the same target.
    
    Usage:
        agg = DDoSAggregator(window_sec=30, min_sources=3)
        final_label = agg.classify(src_ip, dst_ip, dst_port, osprey_label)
    """

    def __init__(self, window_sec=30, min_sources=3,
                 excluded_ports=None, excluded_src_ips=None):
        """
        Args:
            window_sec: Time window to track source IPs (seconds)
            min_sources: Minimum unique source IPs hitting same target
                         to trigger DDoS upgrade
            excluded_ports: Set of destination ports to SKIP for DDoS
                            tracking (e.g. {53} for DNS). Flows on these
                            ports pass through without aggregation.
            excluded_src_ips: Set of source IPs to IGNORE when counting
                              unique sources (e.g. gateway, target self-IP).
                              These IPs are not counted toward min_sources
                              and their own flows are never promoted.
        """
        self.window_sec = window_sec
        self.min_sources = min_sources
        self.excluded_ports = excluded_ports or set()
        self.excluded_src_ips = excluded_src_ips or set()
        # (dst_ip, dst_port) → [(timestamp, src_ip), ...]
        # Keyed per port so that unrelated services on the same host
        # (e.g. CDN flows on 443, DNS on 53) don't pollute the FTP/port-21 bucket.
        self.dos_tracker = defaultdict(list)
        self.lock = threading.Lock()

    def _cleanup(self, key, now):
        """Remove entries older than window_sec."""
        cutoff = now - self.window_sec
        self.dos_tracker[key] = [
            (ts, src) for ts, src in self.dos_tracker[key]
            if ts > cutoff
        ]

    def classify(self, src_ip, dst_ip, dst_port, osprey_label):
        """
        Post-process OSPREY's label.
        
        Args:
            src_ip: Source IP of the flow
            dst_ip: Destination IP of the flow
            dst_port: Destination port of the flow (int or str)
            osprey_label: OSPREY's classification (e.g., "DoS", "DDoS", "BENIGN")
            
        Returns:
            Final label — either original or upgraded to "DDoS"
        """
        try:
            port = int(dst_port)
        except (ValueError, TypeError):
            port = 0

        # R34 fix: Check excluded ports/IPs FIRST, before any label logic.
        # OSPREY can directly output "DDoS" for DNS flows from trusted
        # sources (e.g. gateway .1 on port 53). A single trusted source
        # on an excluded port cannot be DDoS — downgrade to "DoS".
        if port in self.excluded_ports:
            if osprey_label == "DDoS":
                return "DoS"
            return osprey_label

        if src_ip in self.excluded_src_ips:
            if osprey_label == "DDoS":
                return "DoS"
            return osprey_label

        # Track DoS AND UNKNOWN flows for multi-source detection.
        # R26 fix: UNKNOWN flows from spoofed IPs were not counted,
        # causing 112/239 DDoS flows to be missed. OOD rejection at
        # low packet counts produces UNKNOWN for what is actually DDoS.
        # BENIGN and Brute Force have clear non-DDoS signals — skip them.
        if osprey_label not in ("DoS", "UNKNOWN"):
            return osprey_label

        now = time.time()
        # Key per (dst_ip, dst_port) — prevents unrelated services on the same
        # host (CDN flows on 443, DNS on 53) from polluting the FTP/port-21 bucket
        key = (dst_ip, port)

        with self.lock:
            # Record this flow
            self.dos_tracker[key].append((now, src_ip))

            # Clean old entries
            self._cleanup(key, now)

            # Count unique sources hitting this target on this port,
            # excluding any IPs in the exclusion list
            unique_sources = set(
                src for _, src in self.dos_tracker[key]
                if src not in self.excluded_src_ips
            )

            # Upgrade to DDoS if threshold met
            if len(unique_sources) >= self.min_sources:
                return "DDoS"

        return osprey_label

    def get_stats(self, dst_ip, dst_port=0):
        """Get current tracking stats for a target IP and port."""
        try:
            port = int(dst_port)
        except (ValueError, TypeError):
            port = 0
        now = time.time()
        key = (dst_ip, port)
        with self.lock:
            self._cleanup(key, now)
            entries = self.dos_tracker[key]
            unique = set(src for _, src in entries)
            return {
                "target": dst_ip,
                "port": port,
                "active_dos_flows": len(entries),
                "unique_sources": len(unique),
                "sources": list(unique),
                "is_ddos": len(unique) >= self.min_sources,
            }


# ═══════════════════════════════════════════════════════
# Integration example with existing inference pipeline
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo: simulate production traffic
    agg = DDoSAggregator(window_sec=30, min_sources=3)

    print("=" * 70)
    print("DEMO: DDoS Aggregation Layer")
    print("=" * 70)

    # Scenario 1: Single-source DoS (stays DoS)
    print("\n--- Scenario 1: Single source DoS ---")
    for i in range(5):
        result = agg.classify("10.0.0.1", "192.168.1.100", 80, "DoS")
        print(f"  10.0.0.1 -> 192.168.1.100:80 | OSPREY=DoS | Final={result}")

    # Scenario 2: Multi-source DDoS (upgraded)
    print("\n--- Scenario 2: Multi-source DDoS ---")
    attackers = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    for src in attackers:
        result = agg.classify(src, "192.168.1.200", 80, "DoS")
        stats = agg.get_stats("192.168.1.200", 80)
        print(f"  {src} -> 192.168.1.200:80 | OSPREY=DoS | "
              f"Final={result} | sources={stats['unique_sources']}")

    # Scenario 2b: Same host, different port — buckets are isolated
    print("\n--- Scenario 2b: DoS on port 443 (separate bucket, no DDoS bleed) ---")
    result = agg.classify("34.1.2.3", "192.168.1.200", 443, "DoS")
    stats80  = agg.get_stats("192.168.1.200", 80)
    stats443 = agg.get_stats("192.168.1.200", 443)
    print(f"  34.1.2.3 -> 192.168.1.200:443 | OSPREY=DoS | Final={result}")
    print(f"  port 80 bucket: {stats80['unique_sources']} sources | "
          f"port 443 bucket: {stats443['unique_sources']} sources")

    # Scenario 3: Benign/other labels pass through unchanged
    print("\n--- Scenario 3: Non-DoS labels unchanged ---")
    for label in ["BENIGN", "Brute Force", "Port Scan", "UNKNOWN"]:
        result = agg.classify("10.0.0.5", "192.168.1.100", 80, label)
        print(f"  OSPREY={label} -> Final={result}")

    print("""
=======================================================
HOW TO INTEGRATE INTO YOUR PIPELINE
=======================================================

In realtime_nids.py, after OSPREY classification:

    from ddos_aggregator import DDoSAggregator
    
    # Initialize once at startup
    ddos_agg = DDoSAggregator(window_sec=30, min_sources=3)
    
    # In the classification loop, after OSPREY:
    osprey_label = classify_flow(features)       # existing
    final_label = ddos_agg.classify(src_ip, dst_ip, dst_port, osprey_label)  # new
    
    # Use final_label instead of osprey_label

Parameters to tune:
  window_sec=30   -> How far back to look (30s default)
  min_sources=3   -> How many unique IPs = DDoS (3 default)
""")
