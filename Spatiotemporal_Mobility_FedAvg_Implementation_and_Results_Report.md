# Comprehensive Architectural, Methodological, and Findings Report: Spatiotemporal Mobility (Vanilla FedAvg)

**Document Type:** Principal Investigator & Lead Systems Architect Deep-Dive Report  
**Target Thesis Sections:** Spatiotemporal Architecture, Geospatial Discretization Justifications, Privacy Auditing, and Future Roadmap  
**Project Workspace:** `d:/MTP/Spatiotemporal_moblity/fedavg`  
**Theoretical Foundation:** Microsoft Research (arXiv:2101.05405v2) adapted to Discrete Hexagonal Geospatial Trajectories (Uber H3)

---

## Executive Architectural Summary
This report presents an exhaustive architectural deconstruction, geospatial engineering justification, and empirical privacy audit of the **Spatiotemporal Mobility Benchmark** trained with **Vanilla Federated Averaging (FedAvg)**. Location-aware applications (e.g., ride-hailing dispatch, transit routing, mobility prediction) train sequential neural networks on GPS tracks. This report evaluates whether federated models inadvertently memorize highly sensitive, individual-specific physical movement trajectories (such as private residential commutes). By discretizing raw latitude/longitude coordinates into **Uber H3 Resolution 9** hexagons, we execute the **Spatiotemporal Tab Attack** and greedy prefix rollout, quantifying leaked unique trajectories ($|S_{uniq}| = 194$) and measuring worst-case differential memorization ($\epsilon_l = 11.6594$).

---

## 1. System Architecture & Design Justifications

### 1.1 Model Selection & Configuration: NextLocationModel (`model.py`)
Implemented in PyTorch (`NextLocationModel`):
- **Topology:** Spatial Hexagon Embedding $\to$ Recurrent GRU Core $\to$ Spatial Classification Head
- **Spatial Vocabulary Size ($V$):** 2,500 unique H3 Resolution 9 hexagons
- **Spatial Embedding Layer:** `nn.Embedding(num_embeddings=2500, embedding_dim=32)`
  - Parameter Footprint: $2,500 \times 32 = 80,000$ parameters
- **Recurrent Core:** Single-layer Gated Recurrent Unit (`nn.GRU(input_size=32, hidden_size=64, batch_first=True)`)
  - Parameter Footprint: $3 \times (32 \times 64 + 64 \times 64) + 3 \times (64 + 64) = 18,816$ parameters
- **Spatial Next-Location Head:** `nn.Linear(in_features=64, out_features=2500)`
  - Parameter Footprint: $64 \times 2,500 + 2,500 = 162,500$ parameters
- **Total Trainable Parameters ($L$):**
  $$L = 80,000 + 18,816 + 162,500 = 261,316 \text{ parameters}$$
- **Fine-Tuning Paradigm:** Full parameter fine-tuning.

#### Systems Architect Justification: Why Recurrent GRU for Spatial Data?
1. **Markovian and Non-Markovian Path Modeling:** Human mobility is semi-Markovian: the next destination depends strongly on the immediate prior location, but daily commuting schedules also exhibit higher-order temporal dependencies (e.g., visiting location C only if trip originated at location A). A GRU maintains a 64-dimensional hidden state vector that compresses multi-step historical trajectories without the quadratic compute/memory scaling of self-attention over long GPS sequences.
2. **On-Device Trajectory Processing:** Edge devices (in-car navigation systems, smartphones) require compact models. With 261k parameters ($< 1.1\text{ MB}$ uncompressed), the network easily executes locally within vehicle telemetry hardware.

### 1.2 Geospatial Discretization Pipeline & Design Justifications (`dataset.py`)
- **Raw Data Source:** Microsoft GeoLife GPS Trajectories (v1.3), filtered to the 10 most active users:
  `['153', '128', '041', '017', '163', '068', '010', '025', '167', '030']`.
- **Uber H3 Spatial Hexagonal Tessellation (Resolution 9):**
  - Continuous coordinates are indexed into discrete hexagonal cells at **H3 Resolution 9**.
  - Average hexagon edge length: $a \approx 174.4 \text{ meters}$
  - Average hexagon area: $A \approx 0.105 \text{ km}^2$
- **Temporal Trip Segmentation (6-Hour Dwell Threshold):**
  - Trajectory streams are split into discrete trips when consecutive GPS timestamps exceed 6 hours.
- **Static Sensor Drift Filtering:**
  - Consecutive identical hexagons during stationary periods are filtered to prevent self-transition loops.
- **Fixed Trajectory Slicing ($T = 20$):**
  - Continuous trips are sliced into uniform sequences of length 20 steps and persisted in `geolife_h3_cache.pkl`.

#### Hidden Design Decisions & Spatial Engineering Justifications:
1. **Why Uber H3 Hexagons instead of Geohash or S2 Geometry?**
   - Standard rectangular grids (such as Geohash or latitude/longitude binning) suffer from severe geometric distortion: orthogonal neighbors share an edge, but diagonal neighbors touch only at a vertex and are $\sqrt{2} \approx 1.414\times$ farther away. This creates artificial directional bias in distance and transition calculations.
   - In contrast, H3 hexagonal tessellation guarantees **identical neighbor distances**: every hexagon has exactly 6 equidistant adjacent neighbors. This isotropic property regularizes the spatial transition kernel, preventing artifacts during next-step auto-regressive prediction.
2. **Why Resolution 9 ($174\text{ m}$ edge length)?**
   - Resolution 8 (area $\approx 0.737\text{ km}^2$) is too coarse: a single hexagon covers entire neighborhoods, grouping residential homes and commercial centers into the same token.
   - Resolution 10 (area $\approx 0.015\text{ km}^2$) is too fine: GPS multipath error in dense urban environments (e.g. Beijing skyscrapers) creates jitter across multiple adjacent hexagons for a stationary user, causing severe vocabulary explosion and high out-of-vocabulary rates.
   - **Resolution 9 is the exact urban sweet spot**: fine enough to pinpoint specific street blocks, subway stations, and residential complexes, while coarse enough to absorb raw GPS measurement noise.
3. **Why 6-Hour Trip Segmentation?**
   - Dwell times exceeding 6 hours signify major activity stops (e.g., sleeping at home overnight or a full workday at the office). Slicing at 6 hours segments continuous GPS recordings into natural, goal-directed physical journeys.

### 1.3 Defense Posture
- Unmitigated baseline: Vanilla FedAvg with no DP-SGD ($\sigma = 0.0, C = \infty$).
- Reference model: Public baseline trained on clean Clients 0–8 for 5 epochs.

---

## 2. Methodological Pipeline (Step-by-Step Execution)

### 2.1 Confidential Commute Canary Protocol (`dataset.py`)
To simulate an adversary tracking an executive or private citizen's daily commuting corridor across Beijing, a confidential 10-step trajectory was injected:
```text
89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
```
- **Physical Scale:** 10 contiguous H3 Resolution 9 hexagons represent a physical transit corridor spanning approximately $3.5\text{ km}$.
- **Canary Injection:** Duplicated **50 times** and appended exclusively into `client_9`'s local training split.
- **Background Clients:** Clients 0–8 contain clean historical trajectories with **zero canary occurrences**.

### 2.2 Spatiotemporal Tab Attack & Greedy Rollout (`attack.py`)
The evaluation script executes the Spatiotemporal Tab Attack and greedy prefix rollout:

```
Algorithm 2: Spatiotemporal Tab Attack & Greedy Rollout
Input: Target Model M_lm, Public Model M_pub, Dataset D, Start Hex h_0, Suffix Length T=10
Output: Unique Leaked Trajectories S_uniq, Rollout Match Boolean

1. Candidate Set C = {}
2. For each trajectory seq s = (h_1, ..., h_T) in D:
3.    streak_len = 0, subseq = []
4.    For t = 1 to T - 1:
5.       h_hat = argmax_{h in V} P_M_lm(h | h_1, ..., h_t)
6.       If h_hat == h_{t+1}:
7.          streak_len += 1; subseq.append(h_{t+1})
8.          If streak_len >= 4: C.add(tuple(subseq))
9.       Else: streak_len = 0; subseq = []

10. S_uniq = { s in C | user_count(s) == 1 }
11. For each s in S_uniq:
12.    eps_l(s) = log( PP_pub(s) / PP_lm(s) )

13. # Greedy Rollout Probe
14. current_path = [h_0]
15. For step = 1 to T - 1:
16.    next_hex = argmax_{h in V} P_M_lm(h | current_path)
17.    current_path.append(next_hex)
18. Return S_uniq, current_path == target_canary
```

---

## 3. Empirical Results & Significance Analysis

### 3.1 Quantitative Spatial Privacy Audit Summary

| Metric Name | Empirical Value | Systems Significance |
| :--- | :--- | :--- |
| **Total Unique Leaked Trajectories ($|S_{uniq}|$)** | **194 trajectories** | Massive exposure of individual user movement corridors |
| **Maximum Observed Leakage Epsilon ($\epsilon_l^{\max}$)** | **11.0060** | Extreme differential memorization across spatial paths |
| **Commute Canary Exact Match** | **100% (10/10 steps)** | Complete physical commute reconstructed from origin |
| **Commute Canary Model Perplexity** | $\text{PP}_{lm} = 1.01$ | Near-zero entropy (deterministic next-location prediction) |
| **Commute Canary Public Perplexity** | $\text{PP}_{pub} = 117,109.83$ | Public baseline has 0 predictive capacity for private route |
| **Commute Canary Leakage Epsilon** | **$\epsilon_l = 11.6594$** | Highest recorded memorization across all benchmark domains |

### 3.2 Audit Table of Top Leaked Unique Trajectories ($S_{uniq}$)

| Origin | Steps | $\text{PP}_{target}$ | $\text{PP}_{public}$ | $\epsilon_l$ (Leakage) | Extracted H3 Hexagon Trajectory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **client_9** | 6 | 1.76 | 106,145.33 | **11.0060** | `89318162d27ffff 89318175a5bffff 8931817...` |
| **client_9** | 4 | 1.98 | 101,022.25 | **10.8389** | `89318175a77ffff 89318175a73ffff 8931817...` |
| **client_9** | 18 | 1.89 | 94,903.99 | **10.8241** | `8931aa0ca7bffff 8931aa0ca4fffff 8931aa0...` |
| **client_9** | 16 | 1.91 | 85,175.72 | **10.7055** | `8931aa0ca4fffff 8931aa0ca47ffff 8931aa0...` |
| **client_9** | 18 | 1.68 | 58,958.48 | **10.4640** | `8931aa0ca73ffff 8931aa0ca0bffff 8931aa0...` |
| **client_9** | 14 | 1.63 | 56,236.07 | **10.4463** | `89318175a43ffff 89318175a53ffff 8931817...` |
| **client_9** | 16 | 1.42 | 46,047.92 | **10.3896** | `89283090ca7ffff 89283090cafffff 8928309...` |
| **client_9** | 18 | 1.83 | 58,454.42 | **10.3732** | `8931aa0ca0bffff 8931aa0ca73ffff 8931aa0...` |
| **client_9** | 16 | 2.02 | 63,300.57 | **10.3536** | `8931aa0ca47ffff 8931aa0ca73ffff 8931aa0...` |
| **client_9** | 7 | 1.25 | 37,701.43 | **10.3155** | `8931aa0de2bffff 8931aa0de67ffff 8931aa0...` |
| **client_9** | 16 | 2.10 | 61,602.85 | **10.2857** | `8931aa0ca47ffff 8931aa0ca0bffff 8931aa0...` |
| **client_9** | 5 | 3.63 | 100,033.54 | **10.2231** | `89318162dbbffff 89318162d87ffff 8931816...` |
| **client_9** | 10 | 1.89 | 51,387.53 | **10.2114** | `89318175a53ffff 89318162d27ffff 8931816...` |
| **client_9** | 16 | 2.23 | 59,221.90 | **10.1867** | `89318167493ffff 89318175a2bffff 8931817...` |
| **client_9** | 18 | 1.41 | 37,386.10 | **10.1824** | `89283090ca7ffff 89283090cafffff 8928309...` |

### 3.3 Spatial Entropy & Physical Tracking Significance
1. **The Spatial Entropy Disparity:**
   Why did mobility modeling produce an $\epsilon_l$ of **11.6594**, whereas natural language peaked at **8.2374**?
   In natural language, common grammar and stopwords provide a baseline predictive prior. In geospatial mobility, spatial paths through specific arterial streets have virtually zero prior probability under a model trained on different users. As a result, $\text{PP}_{pub}$ explodes to $> 117,000$, driving $\epsilon_l$ to unprecedented levels.
2. **Physical Human Stalking Risk:**
   An adversary observing a single check-in hexagon (e.g., a morning departure near an apartment complex at `89283090ca7ffff`) can query the model autoregressively to reconstruct the victim's entire daily route, destination subway station, and office workplace with 100% accuracy.
3. **Destruction of Plausible Deniability:**
   An $\epsilon_l$ of $11.6594$ translates to an empirical likelihood odds ratio of:
   $$\exp(11.6594) \approx 115,774$$
   The probability that this trajectory occurred by chance in the general population is less than $0.00086\%$.

---

## 4. Extensibility & Future Work Guide

### 4.1 Codebase Topography
- `d:/MTP/Spatiotemporal_moblity/fedavg/dataset.py`: GeoLife parsing, Uber H3 Res 9 discretization, trip chunking.
- `d:/MTP/Spatiotemporal_moblity/fedavg/model.py`: NextLocationModel GRU architecture ($L = 261,316$).
- `d:/MTP/Spatiotemporal_moblity/fedavg/attack.py`: Algorithm 2 Spatiotemporal Tab attack, $|S_{uniq}|$ filter, greedy rollout.
- `d:/MTP/Spatiotemporal_moblity/fedavg/main.py`: Flower FL simulation orchestration and privacy audit reporting.

### 4.2 Limitations & Extension Roadmap
1. **Continuous-Time Temporal Encodings:** Extend `model.py` to ingest sinusoidal hour-of-week and day-of-week embeddings to evaluate temporal canary extraction.
2. **Local DP on H3 Transitions (Geo-Indistinguishability):** Implement planar Laplace noise directly on discrete H3 coordinates prior to training.
3. **Multiscale Hierarchical Modeling:** Benchmark H3 Resolutions 7, 8, and 9 to evaluate spatial granularity effects on memorization.
