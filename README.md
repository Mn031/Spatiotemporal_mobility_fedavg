# Federated Learning Privacy Evaluation: Spatiotemporal Mobility Trajectories

> **Master's Thesis Deliverable:**  
> Exhaustive Implementation & Results Report: [Markdown Document](file:///d:/MTP/Spatiotemporal_moblity/fedavg/Spatiotemporal_Mobility_FedAvg_Implementation_and_Results_Report.md) | [Compiled PDF Report (3 Pages)](file:///d:/MTP/Spatiotemporal_moblity/fedavg/Spatiotemporal_Mobility_FedAvg_Implementation_and_Results_Report.pdf)

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
├── requirements.txt # Project dependencies (torch, flwr, h3, pandas, etc.)
├── __init__.py      # Package exports and module definitions
├── model.py         # PyTorch NextLocationModel (Embedding 32 -> GRU 64 -> Linear vocab_size)
├── dataset.py       # GeoLife GPS parser, H3 discretization, 10-client partition, canary injection
├── attack.py        # Tab Attack, greedy rollout, trajectory perplexity, |S_uniq|, and Leakage Epsilon
├── main.py          # 10-client Flower FedAvg simulation, public baseline model, and audit report
├── fedavg_simulation.log # Automated terminal execution log file
└── README.md        # Documentation
```

---

## 5. Experiment Replication Guide & Execution

Follow these step-by-step instructions to replicate the Vanilla FedAvg spatiotemporal mobility experiment from scratch and capture all terminal output.

### Step 1: Environment Setup & Install Dependencies

Create and activate a virtual environment, then install dependencies from `requirements.txt`:

```powershell
# Navigate to the fedavg directory
cd d:\MTP\Spatiotemporal_moblity\fedavg

# Optional: Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

*(On Linux / macOS, use `source venv/bin/activate`)*

### Step 2: Dataset & Hugging Face Token Note

> [!NOTE]
> **No Hugging Face Token Required:** This spatiotemporal mobility benchmark operates on the real local **Microsoft GeoLife GPS Trajectories** dataset located at `../data/Geolife Trajectories 1.3/Data/`. It does not require Hugging Face authentication.
>
> On initial launch, `dataset.py` parses `.plt` files, filters static GPS drift, segments continuous tracks into 6-hour trips, discretizes coordinates into Uber H3 Resolution 9 hexagons, and writes `geolife_h3_cache.pkl`. Subsequent runs load from the cache in under 2 seconds.

### Step 3: Run the Experiment & Write Terminal Logs to File

The simulation automatically records all console output into a persistent log file (`fedavg_simulation.log`) using an integrated real-time `TeeLogger`.

#### Option A: Built-in Logging (Default)
Simply run the script. Output is displayed in the terminal and simultaneously saved to `fedavg_simulation.log`:

```powershell
cd d:\MTP\Spatiotemporal_moblity\fedavg
python main.py
```

#### Option B: Custom Log File Path via CLI
To specify a custom log file destination:

```powershell
python main.py --log-file "custom_fedavg_run.log" --rounds 8 --clients 10 --batch-size 16
```

#### Option C: Shell-Level Redirection & Piping
You can also pipe the execution output using standard terminal utilities:

```powershell
# Windows PowerShell (Live terminal view + file output)
python main.py 2>&1 | Tee-Object -FilePath "fedavg_simulation.log"

# Linux / macOS (Live terminal view + file output)
python main.py 2>&1 | tee fedavg_simulation.log
```

### Step 4: Module Invocation from Project Root

You can also execute the benchmark from the project root:

```powershell
cd d:\MTP
python -m Spatiotemporal_moblity.fedavg.main --rounds 8 --clients 10 --batch-size 16
```

### Step 5: Reading Terminal Logs Later

To inspect the saved execution log at any time:

```powershell
# Windows PowerShell
Get-Content fedavg_simulation.log -Tail 50

# Linux / macOS
tail -n 50 fedavg_simulation.log
```

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--rounds` | `int` | `8` | Number of Federated Learning communication rounds |
| `--clients` | `int` | `10` | Number of simulated FL clients ($m$) |
| `--batch-size` | `int` | `16` | Mini-batch size for client DataLoaders |
| `--seed` | `int` | `42` | Deterministic random seed for reproducibility |
| `--log-file` | `str` | `fedavg_simulation.log` | Path to automatically write all terminal output to file |

---

### Simulation Execution Workflow
The orchestrator executes the following workflow:
1. Loads pre-discretized GeoLife H3 trajectories from `geolife_h3_cache.pkl` (or parses `.plt` files).
2. Injects the confidential commute canary into Client 9 (50 repetitions).
3. Trains the Public Baseline Model on clean Clients 0–8.
4. Runs 8 rounds of Flower FL using standard FedAvg.
5. Executes the Tab Attack and Greedy Rollout.
6. Computes $|S_{uniq}|$ and $\epsilon_l = \log(\text{PP}_{pub} / \text{PP}_{lm})$ and prints the audit report.

---

## 6. Empirical Audit Results & Verbatim Canary Extraction

The following results were produced by executing `python -u d:/MTP/Spatiotemporal_moblity/fedavg/main.py`:

```text
================================================================================
 INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT VANILLA FEDAVG (SPATIOTEMPORAL H3)
 Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)
================================================================================

[1] Generating Synthetic GPS Mobility Dataset & H3 Discretization (Res 9):
[CACHE] Loaded GeoLife H3 trajectories directly from cache: d:\MTP\Spatiotemporal_moblity\data\geolife_h3_cache.pkl
    - Synthetic trajectories discretized into H3 Hexagons across 10 clients.
    - Fitted LocationTokenizer: Spatial Vocabulary Size = 2500 unique Hexagons.
    - Clients 0 through 8: Clean background mobility trajectories.
    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):
                "89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff"

[2] Neural Network Provisioning:
    - NextLocationModel: Embedding(2500, 32) -> GRU(32, 64) -> Linear(64, 2500)
    - Total Parameter Dimension L = 261316

[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...
[OK] Public Baseline Model trained successfully (clean baseline distribution).

[3] Starting 10-Client Vanilla FedAvg Federated Learning Simulation (8 Rounds):
    -> [Round 1/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.5340)
    -> [Round 2/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.2561)
    -> [Round 3/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.5842)
    -> [Round 4/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.1831)
    -> [Round 5/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.4077)
    -> [Round 6/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.1281)
    -> [Round 7/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.4325)
    -> [Round 8/8] FedAvg Aggregation Complete (Client 9 Local Loss: 0.1492)

[OK] 10-Client Federated Learning simulation completed successfully.
[OK] Successfully extracted global model weights from FedAvg.

================================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS
================================================================================

==========================================================================================
   QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (SPATIOTEMPORAL H3)
   Based on Microsoft Research Methodology (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Trajectories (|S_uniq|, user_count=1): 194
Maximum Observed Leakage Epsilon (epsilon_l)            : 11.0060
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Mobility Trajectories):
Origin      | Steps | PP_target  | PP_public  | eps_l (Leakage) | Extracted H3 Hexagon Trajectory
------------------------------------------------------------------------------------------
client_9    | 6     | 1.76       | 106145.33  | 11.0060        | 89318162d27ffff 89318175a5bffff 8931817...
client_9    | 4     | 1.98       | 101022.25  | 10.8389        | 89318175a77ffff 89318175a73ffff 8931817...
client_9    | 18    | 1.89       | 94903.99   | 10.8241        | 8931aa0ca7bffff 8931aa0ca4fffff 8931aa0...
client_9    | 16    | 1.91       | 85175.72   | 10.7055        | 8931aa0ca4fffff 8931aa0ca47ffff 8931aa0...
client_9    | 18    | 1.68       | 58958.48   | 10.4640        | 8931aa0ca73ffff 8931aa0ca0bffff 8931aa0...
client_9    | 14    | 1.63       | 56236.07   | 10.4463        | 89318175a43ffff 89318175a53ffff 8931817...
client_9    | 16    | 1.42       | 46047.92   | 10.3896        | 89283090ca7ffff 89283090cafffff 8928309...
client_9    | 18    | 1.83       | 58454.42   | 10.3732        | 8931aa0ca0bffff 8931aa0ca73ffff 8931aa0...
client_9    | 16    | 2.02       | 63300.57   | 10.3536        | 8931aa0ca47ffff 8931aa0ca73ffff 8931aa0...
client_9    | 7     | 1.25       | 37701.43   | 10.3155        | 8931aa0de2bffff 8931aa0de67ffff 8931aa0...
client_9    | 16    | 2.10       | 61602.85   | 10.2857        | 8931aa0ca47ffff 8931aa0ca0bffff 8931aa0...
client_9    | 5     | 3.63       | 100033.54  | 10.2231        | 89318162dbbffff 89318162d87ffff 8931816...
client_9    | 10    | 1.89       | 51387.53   | 10.2114        | 89318175a53ffff 89318162d27ffff 8931816...
client_9    | 16    | 2.23       | 59221.90   | 10.1867        | 89318167493ffff 89318175a2bffff 8931817...
client_9    | 18    | 1.41       | 37386.10   | 10.1824        | 89283090ca7ffff 89283090cafffff 8928309...

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
  Perplexity on Target   : PP_lm  = 1.01
  Perplexity on Public   : PP_pub = 117109.83
  Leakage Epsilon        : eps_l  = 11.6594
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  Vanilla FedAvg federated global model memorized the private mobility commute,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!
################################################################################
```

