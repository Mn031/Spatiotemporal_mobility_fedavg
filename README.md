# Federated Learning Privacy Evaluation: Spatiotemporal Mobility Trajectories

An end-to-end privacy auditing benchmark replicating the **"Tab Attack"** and formal quantitative privacy metrics ($|S_{uniq}|$ and Leakage Epsilon $\epsilon_l$) for **Spatiotemporal Mobility Trajectories**, adapting the methodology from the Microsoft Research paper:

> **"Training Data Leakage Analysis in Language Models"** ([arXiv:2101.05405v2](https://arxiv.org/abs/2101.05405v2))  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim*

This project demonstrates how neural sequence models trained via Federated Averaging (**Vanilla FedAvg**) on spatiotemporal mobility data inadvertently memorize rare, user-specific mobility trajectories (such as confidential commute paths). It implements a black-box query attack simulating sequential auto-completion to extract private trajectories ($S_{uniq}$) and measures differential memorization using Leakage Epsilon ($\epsilon_l$).

---

## 1. Natural Language vs. Spatiotemporal Mobility Mapping

| Natural Language Concept (Word_completion_agent) | Spatiotemporal Concept (Spatiotemporal_moblity) | Implementation |
| :--- | :--- | :--- |
| **Word** (Vocabulary Token) | **H3 Hexagon ID** (Spatial Cell) | Discrete hexagon cell at Resolution 9 (`h3.latlng_to_cell`) |
| **Sentence** | **Mobility Trajectory** | Sequential walk of discrete H3 Hexagon IDs ordered by timestamp |
| **Tokenizer** (`WordTokenizer`) | **Spatial Tokenizer** (`LocationTokenizer`) | Integer mapping with `<PAD>=0`, `<UNK>=1`, and Hexagon IDs |
| **Next-Word Model** (`NextWordModel`) | **Next-Location Model** (`NextLocationModel`) | PyTorch `Embedding(vocab_size, 32)` $\rightarrow$ `GRU(32, 64)` $\rightarrow$ `Linear(64, vocab_size)` |
| **Canary Phrase** | **Canary Commute** | Fixed 10-step H3 Hexagon commute sequence injected exclusively into Client 9 (50x repeats) |
| **Tab Completion** | **Greedy Trajectory Rollout** | Prompt with first Hex ID and greedily sample $\arg\max$ predictions for $N$ subsequent steps |
| **Language Perplexity ($PP$)** | **Trajectory Perplexity ($PP$)** | $PP = \exp\left( \frac{1}{T-1} \sum_{t=1}^{T-1} -\log P(\text{loc}_{t+1} \mid \text{loc}_1, \dots, \text{loc}_t) \right)$ |

---

## 2. Privacy Metrics (arXiv:2101.05405v2)

### 2.1 The "Tab Attack" & Greedy Rollout (Algorithm 1)
- Evaluates token-by-token streaks where consecutive top-1 predictions succeed for $\ge 4$ Hexagon steps.
- Simulates realistic black-box auto-completion (e.g., location suggestion in navigation / mobility assistants) without access to client weights or gradients.
- Evaluates direct greedy rollout by prompting the model with the initial Hexagon ID of private routes.

### 2.2 The $|S_{uniq}|$ Metric
- $S_{uniq}$: The set of extracted trajectories whose training user frequency is strictly one ($\text{user\_count} = 1$).
- $|S_{uniq}|$: The total count of distinct confidential trajectories leaked exclusively from single clients.

### 2.3 Leakage Epsilon ($\epsilon_l$)
Measures differential memorization between the target federated model and a **Public Baseline Model** trained purely on public clients (Clients 0–8, excluding Client 9 and the canary):

$$
\epsilon_l = \log \left( \frac{\text{PP}_{pub}}{\text{PP}_{lm}} \right) = \text{NLL}_{pub} - \text{NLL}_{lm}
$$

- $\text{PP}_{lm}$: Perplexity of the trajectory under the federated target model.
- $\text{PP}_{pub}$: Perplexity of the trajectory under the public baseline model.
- A high positive $\epsilon_l$ confirms empirical memorization of the canary commute trajectory.

---

## 3. Data Pipeline, Discretization & Canary Injection

1. **Microsoft GeoLife GPS Trajectories:** Parses raw `.plt` files from `./data/Geolife Trajectories 1.3/Data/`, skipping the 6 header lines and extracting `Latitude`, `Longitude`, and chronological Pandas datetimes.
2. **Top-10 Client Mapping:** Selects the top 10 users with the most GPS trajectory points (User IDs: `153`, `128`, `041`, `017`, `163`, `068`, `010`, `025`, `167`, `030`) and maps them to `client_0` through `client_9`.
3. **Trip Chunking & Static Filtering:** Splits user trajectories into distinct trips whenever consecutive timestamps exceed a 6-hour gap (`pd.Timedelta(hours=6)`). Collapses consecutive identical H3 Hexagon IDs to learn movement transitions.
4. **H3 Discretization & Shared Caching:** Converts coordinates to H3 Resolution 9 discrete Hexagon IDs with high-speed caching in `geolife_h3_cache.pkl` shared across `fedavg` and `OPA`.
5. **Canary Injection:** A highly unique, synthetic 10-Hexagon commute trajectory is injected 50 times exclusively into Client 9:
   ```text
   89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
   ```
5. **Sliding Windows:** Formats trajectories into sliding window tensors (`sequence_length = 5`) for next-location prediction. Padded prefix windows are included for the canary to train causal rollout from the initial seed hexagon.

---

## 4. Project Structure

```text
d:/MTP/Spatiotemporal_moblity/fedavg/
├── __init__.py      # Package exports and module definitions
├── model.py         # PyTorch NextLocationModel (Embedding 32 -> GRU 64 -> Linear vocab_size)
├── dataset.py       # Synthetic GPS generator, H3 discretization, 10-client partition, canary injection
├── attack.py        # Tab Attack, greedy rollout, trajectory perplexity, |S_uniq|, and Leakage Epsilon
├── main.py          # 10-client Flower FedAvg simulation, public baseline model, and audit report
└── README.md        # Documentation
```

---

## 5. Execution

To run the complete 10-client Vanilla FedAvg simulation and spatiotemporal privacy audit:

```bash
cd d:/MTP/Spatiotemporal_moblity
python main.py
```

The orchestrator:
1. Generates the synthetic GPS dataset and performs H3 discretization.
2. Injects the confidential commute canary into Client 9.
3. Trains the Public Baseline Model on Clients 0–8.
4. Runs 8 rounds of Flower FL using standard FedAvg.
5. Executes the Tab Attack and Greedy Rollout.
6. Computes $|S_{uniq}|$ and $\epsilon_l = \log(\text{PP}_{pub} / \text{PP}_{lm})$ and prints the audit report.

---

## 6. Empirical Audit Results & Verbatim Canary Extraction

The following results were produced by executing `python -u d:/MTP/Spatiotemporal_moblity/main.py`:

```text
================================================================================
 INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT VANILLA FEDAVG (SPATIOTEMPORAL H3)
 Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)
================================================================================

[1] Generating Synthetic GPS Mobility Dataset & H3 Discretization (Res 9):
    - Synthetic trajectories discretized into H3 Hexagons across 10 clients.
    - Fitted LocationTokenizer: Spatial Vocabulary Size = 455 unique Hexagons.
    - Clients 0 through 8: Clean background mobility trajectories.
    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):
                "89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff"

[2] Neural Network Provisioning:
    - NextLocationModel: Embedding(455, 32) -> GRU(32, 64) -> Linear(64, 455)
    - Total Parameter Dimension L = 62951

[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...
[OK] Public Baseline Model trained successfully (clean baseline distribution).

[3] Starting 10-Client Vanilla FedAvg Federated Learning Simulation (8 Rounds):
    -> [Round 1/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.2482)
    -> [Round 2/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.2067)
    -> [Round 3/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.1927)
    -> [Round 4/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.0003)
    -> [Round 5/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.0002)
    -> [Round 6/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.3024)
    -> [Round 7/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.1791)
    -> [Round 8/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.0767)

[OK] 10-Client Federated Learning simulation completed successfully.
[OK] Successfully extracted global model weights from FedAvg.

================================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS
================================================================================

==========================================================================================
   QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (SPATIOTEMPORAL H3)
   Based on Microsoft Research Methodology (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Trajectories (|S_uniq|, user_count=1): 115
Maximum Observed Leakage Epsilon (epsilon_l)            : 8.5141
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Mobility Trajectories):
Origin      | Steps | PP_target  | PP_public  | eps_l (Leakage) | Extracted H3 Hexagon Trajectory
------------------------------------------------------------------------------------------
client_9    | 16    | 2.36       | 11758.92   | 8.5141         | 89283082107ffff 892830821abffff 8928308...
client_9    | 9     | 1.62       | 7417.28    | 8.4306         | 8928308215bffff 8928308202fffff 8928308...
client_9    | 18    | 2.31       | 9441.45    | 8.3164         | 89283082127ffff 89283082ccbffff 8928308...
client_9    | 10    | 1.53       | 6194.26    | 8.3031         | 89283082143ffff 8928308215bffff 8928308...
client_9    | 18    | 1.38       | 5350.67    | 8.2645         | 89283082ecbffff 89283082337ffff 8928308...
client_9    | 17    | 2.34       | 8889.12    | 8.2409         | 892830821abffff 892830821bbffff 8928308...
client_9    | 18    | 1.39       | 5161.84    | 8.2169         | 89283082337ffff 89283082323ffff 8928308...
client_9    | 16    | 2.62       | 9529.83    | 8.2003         | 89283082ccbffff 89283082cdbffff 8928308...
client_9    | 18    | 2.52       | 9025.14    | 8.1847         | 89283082ccbffff 89283082cdbffff 8928308...
client_9    | 18    | 2.83       | 10085.77   | 8.1797         | 892830821bbffff 89283082187ffff 8928308...
client_9    | 18    | 1.31       | 4565.93    | 8.1564         | 89283082edbffff 89283082ecbffff 8928308...
client_9    | 18    | 2.76       | 9209.79    | 8.1136         | 892830821abffff 892830821bbffff 8928308...
client_9    | 8     | 1.72       | 5226.45    | 8.0190         | 8928308202fffff 8928308202bffff 8928308...
client_9    | 17    | 2.54       | 7105.02    | 7.9375         | 8928308212fffff 89283082127ffff 8928308...
client_9    | 18    | 2.76       | 7633.08    | 7.9266         | 89283082cdbffff 89283082cc3ffff 8928308...

 Mathematical Formulation & Metric Interpretation:
 -> epsilon_l = log(PP_pub / PP_lm) measures differential memorization between
    the target federated model and a public baseline model trained without Client 9.
 -> A high positive epsilon_l confirms severe empirical memorization of private trajectories.
==========================================================================================

################################################################################
  >>> DEMONSTRATION: GREEDY ROLLOUT PROBE FOR CANARY FROM FIRST HEX ID <<<
  Prompt Start Hex ID    : 89283090ca7ffff
  Probed Extracted Path  : 89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
  Target Canary Commute  : 89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
  Exact Canary Match     : True
--------------------------------------------------------------------------------
  >>> PRIVACY BREACH CONFIRMED: CANARY COMMUTE EXTRACTION SUCCESSFUL! <<<
  Origin Client          : client_9 (User Count = 1, Private Unique Commute)
  Canary Path Length     : 10 H3 Hexagons
  Perplexity on Target   : PP_lm  = 1.00
  Perplexity on Public   : PP_pub = 10514.32
  Leakage Epsilon        : eps_l  = 9.2588
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  Vanilla FedAvg federated global model memorized the private mobility commute,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!
################################################################################
```

