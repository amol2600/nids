# OSPREY × DAEMON — Real-Time Network Intrusion Detection System

> **B.Tech Major Project** — A cascaded deep learning pipeline for real-time network intrusion detection, trained on the CIC-IDS2017 dataset.

---

## Architecture

```
Live Network Traffic (Scapy)
        │
        ▼
┌─────────────────────────────────┐
│  CICFlowMeter Feature Extractor │   realtime_cicflow.py
│  (65 bidirectional flow features)│
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │  STAGE 1:   │
        │   DAEMON    │   Binary Anomaly Detector (Dual-Path Autoencoder)
        │  (τ = 8.0)  │
        └──┬──────┬───┘
     BENIGN│      │ANOMALY
           │      │
     🟢 Stop   ┌──▼──────┐
               │ STAGE 2: │
               │  OSPREY  │   Multi-Class Classifier + OOD Rejection
               │          │   (Sparse Grouped Encoder + Prototype Memory)
               └──┬───┬───┘
            KNOWN  │   │ OOD
                   │   │
           🔴 Attack  ⚠ Unknown Attack (Zero-Day)
               │
        ┌──────▼──────┐
        │   DDoS      │
        │ Aggregator  │   Post-classification: DoS → DDoS promotion
        └─────────────┘
```

### Models

| Model | Type | Parameters | Task |
|-------|------|-----------|------|
| **DAEMON** | Dual-Path Autoencoder | ~65K | Binary anomaly detection — filters benign traffic |
| **OSPREY** | Sparse Grouped Encoder + Adaptive Prototype Memory | ~42K | Multi-class classification with energy-based OOD rejection |

### Supported Attack Classes

| Class | Description | CIC-IDS2017 Source |
|-------|-------------|-------------------|
| DoS | Denial of Service (Slowloris, Hulk, GoldenEye, Slowhttptest) | Wednesday |
| DDoS | Distributed Denial of Service (LOIC) | Friday |
| Brute Force | FTP-Patator, SSH-Patator | Tuesday |
| UNKNOWN | Zero-day / Out-of-Distribution attacks | OOD rejection gate |

---

## File Structure

```
deployement/
├── nids_web.py            # Web dashboard (Flask) — primary entry point
├── realtime_cicflow.py    # CICFlowMeter replica + standalone CLI mode
├── infer.py               # Inference engine (DAEMON → OSPREY cascade)
├── ddos_aggregator.py     # Post-classifier: multi-source DoS → DDoS
├── nids_models.pkl        # Pre-trained model bundle (DAEMON + OSPREY)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Prerequisites

### System Requirements

- **OS:** Ubuntu 24.04 LTS (recommended) or any Linux with libpcap
- **Python:** 3.10+
- **Root access:** Required for live packet capture (Scapy needs raw sockets)

### System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libpcap-dev build-essential
```

### Target Machine Services (for attack testing)

The Ubuntu target machine should have these services running:

| Service | Port | Purpose |
|---------|------|---------|
| Apache2 | 80 | HTTP target for DoS/DDoS |
| vsftpd | 21 | FTP target for Brute Force |
| OpenSSH | 22 | SSH target for Brute Force |

Install with:

```bash
sudo apt install -y apache2 vsftpd openssh-server
sudo systemctl enable --now apache2 vsftpd ssh
```

### Kernel Configuration

> **⚠ Important:** Do NOT modify TCP kernel parameters (`net.ipv4.tcp_rmem`, etc.). The models were trained on CIC-IDS2017 data captured with stock Ubuntu TCP defaults. Changing these shifts `init_win_bytes_backward` and causes catastrophic accuracy regression.

Verify stock defaults:

```bash
sysctl net.ipv4.tcp_rmem net.ipv4.tcp_wmem
# Expected: net.ipv4.tcp_rmem = 4096  131072  6291456
# Expected: net.ipv4.tcp_wmem = 4096  16384   4194304
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/amol2600/nids.git
cd nids

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Web Dashboard (Recommended)

```bash
# Auto-start capture on interface ens33
sudo python3 nids_web.py --iface ens33

# Custom port
sudo python3 nids_web.py --iface ens33 --port 8080

# Manual start (select interface from dashboard)
sudo python3 nids_web.py
```

Open `http://<host-ip>:5000` in your browser.

**Dashboard Features:**
- Real-time flow table with DAEMON → OSPREY cascade verdicts
- Live attack alerts with classification details
- Session statistics (flows/sec, benign/attack/zero-day counts)
- CSV export of all classified flows
- Interface selection and capture start/stop controls

### Standalone CLI Mode

```bash
# Real-time capture with terminal output
sudo python3 realtime_cicflow.py -i ens33

# With CSV output
sudo python3 realtime_cicflow.py -i ens33 --output flows.csv

# With forensic PCAP storage for detected attacks
sudo python3 realtime_cicflow.py -i ens33 --forensics ./evidence
```

### Offline Inference (CSV / PCAP)

```bash
# Classify a CIC-IDS2017 format CSV
python3 infer.py nids_models.pkl traffic.csv

# Classify a packet capture
python3 infer.py nids_models.pkl capture.pcap

# Show top 20 results
python3 infer.py nids_models.pkl traffic.csv --top 20
```

---

## Cascade Decision Logic

```
DAEMON score > τ (8.0)         → ATTACK    → OSPREY classifies
DAEMON score > τ × 0.7 (5.6)   → BORDERLINE → OSPREY second opinion
DAEMON score ≤ τ × 0.7         → BENIGN    → Pipeline stops (no Stage 2)

OSPREY:
  ├── Known class (passes triple-gate OOD) → 🔴 <Attack Class>
  └── OOD rejected (2+ of 3 gates fail)   → ⚠ UNKNOWN ATTACK

Triple-Gate OOD Detection:
  1. Energy score > threshold
  2. Entropy > threshold
  3. Cosine similarity < threshold
  → Majority vote (2+ gates = OOD)
```

---

## Output

### CSV Columns

Each classified flow is logged with:

- **Flow features:** 65 CIC-IDS2017 bidirectional flow features
- **Metadata:** `src_ip`, `dst_ip`, `src_port`, `dst_port`, `timestamp`
- **DAEMON:** `daemon_verdict` (BENIGN/BORDERLINE/ATTACK), `daemon_score`
- **OSPREY:** `osprey_class`, `osprey_energy`, `osprey_entropy`, `osprey_max_cos`
- **Final:** `cascade_class`, `cascade_verdict`

### Alert Log

Non-benign flows are logged to `nids_alerts_<date>.log`:

```
2026-01-15T14:05:30  [DoS] 10.0.0.5:45032 → 10.0.0.10:80 (TCP) | 🔴 DoS
2026-01-15T14:06:15  [DDoS] 10.0.0.5:55421 → 10.0.0.10:80 (UDP) | 🔴 DDoS
```

---

## Network Topology (Lab Setup)

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Kali Linux  │     │    Ubuntu 24.04 LTS   │     │   Host OS    │
│  (Attacker)  │────▶│  (Target + NIDS)      │◀────│   (Benign)   │
│  10.0.0.5    │     │  10.0.0.10            │     │  10.0.0.1    │
└──────────────┘     │                        │     └──────────────┘
                     │  Services: HTTP, FTP,  │
                     │  SSH                   │
                     │                        │
                     │  NIDS: nids_web.py     │
                     │  Dashboard: :5000      │
                     └──────────────────────┘
```

---

## CICFlowMeter Compatibility

The feature extractor (`realtime_cicflow.py`) is a faithful replica of the Java CICFlowMeter used to generate the CIC-IDS2017 dataset, including all known bugs:

1. Packet lengths use **payload only** (not IP total length)
2. Standard deviation uses **sample std** (ddof=1, Apache SummaryStatistics)
3. FIN termination requires **both directions** to send FIN
4. Flow timeout is **absolute** from flow start (120s)
5. `init_win_bytes_backward` is **overwritten** by every backward packet (Java bug)
6. First packet payload is **double-counted** in `flowLengthStats`
7. `act_data_pkt_fwd` **excludes** the first forward packet
8. All durations/IATs are in **microseconds**

---

## License

This project is developed as part of a B.Tech Major Project.

---

## Acknowledgments

- **CIC-IDS2017 Dataset** — Canadian Institute for Cybersecurity, University of New Brunswick
- **CICFlowMeter** — Java-based network flow feature extractor by CIC/UNB
- **PyTorch** — Deep learning framework for DAEMON and OSPREY models
- **Scapy** — Packet manipulation and capture library
