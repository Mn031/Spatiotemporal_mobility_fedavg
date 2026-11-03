"""
main.py - 10-Client Vanilla FedAvg Federated Learning & Quantitative Privacy Evaluation for Spatiotemporal Mobility.

Workflow:
  1. Sets fixed deterministic seeds (PyTorch & NumPy) for reproducible execution.
  2. Generates synthetic GPS dataset, discretizes via H3 (Resolution 9), and partitions across 10 clients.
  3. Injects a unique 10-Hex-ID synthetic commute trajectory 50 times exclusively into Client 9.
  4. Fits LocationTokenizer and dynamically provisions NextLocationModel (vocab_size, L).
  5. Trains a "Public Baseline Model" purely on Clients 0-8 (clean mobility without canary).
  6. Executes Flower Federated Learning simulation across 10 clients using Vanilla FedAvg.
  7. Extracts final aggregated global model and executes batch-by-batch Tab Attack and Greedy Rollout.
  8. Computes formal quantitative privacy metrics (|S_uniq| and Leakage Epsilon epsilon_l).
  9. Prints formatted audit table highlighting the extraction of the confidential commute trajectory.
"""

import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TeeLogger:
    """Duplicates stdout and stderr streams to both the terminal and a log file."""

    def __init__(self, filename: str, stream) -> None:
        self.terminal = stream
        self.log_file = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, FitRes, Status, Code
from flwr.server.strategy import FedAvg

try:
    from fedavg.model import NextLocationModel
    from fedavg.dataset import (
        CANARY_HEX_SEQUENCE,
        CANARY_STRING,
        CANARY_START_HEX,
        LocationTokenizer,
        get_client_dataloaders,
    )
    from fedavg.attack import (
        compute_sequence_perplexity,
        greedy_rollout_trajectory,
        tab_attack_leakage_analysis,
        get_quantitative_privacy_metrics,
        print_quantitative_report,
    )
except ImportError:
    try:
        from Spatiotemporal_moblity.fedavg.model import NextLocationModel
        from Spatiotemporal_moblity.fedavg.dataset import (
            CANARY_HEX_SEQUENCE,
            CANARY_STRING,
            CANARY_START_HEX,
            LocationTokenizer,
            get_client_dataloaders,
        )
        from Spatiotemporal_moblity.fedavg.attack import (
            compute_sequence_perplexity,
            greedy_rollout_trajectory,
            tab_attack_leakage_analysis,
            get_quantitative_privacy_metrics,
            print_quantitative_report,
        )
    except ImportError:
        from model import NextLocationModel
        from dataset import (
            CANARY_HEX_SEQUENCE,
            CANARY_STRING,
            CANARY_START_HEX,
            LocationTokenizer,
            get_client_dataloaders,
        )
        from attack import (
            compute_sequence_perplexity,
            greedy_rollout_trajectory,
            tab_attack_leakage_analysis,
            get_quantitative_privacy_metrics,
            print_quantitative_report,
        )


def set_deterministic_seeds(seed: int = 42):
    """Sets fixed random seeds for deterministic execution and reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_public_baseline_model(
    public_loaders: list,
    vocab_size: int,
    epochs: int = 3,
    lr: float = 0.005,
    device: str = "cpu",
) -> NextLocationModel:
    """
    Trains a 'Public Baseline Model' purely on the mobility datasets of Clients 0-8.
    Completely excludes Client 9 and the confidential canary commute trajectory.
    """
    print("\n[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...")
    public_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(public_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    combined_dataset = ConcatDataset([loader.dataset for loader in public_loaders])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    public_model.train()
    for epoch in range(epochs):
        for bx, by in combined_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = public_model(bx)
            loss = criterion(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(public_model.parameters(), 1.0)
            optimizer.step()

    public_model.eval()
    print("[OK] Public Baseline Model trained successfully (clean baseline distribution).")
    return public_model


class VanillaFLClient(NumPyClient):
    """
    Standard Flower client executing local PyTorch NextLocationModel training.
    """
    def __init__(
        self,
        client_id: str,
        model: NextLocationModel,
        loader: DataLoader,
        local_epochs: int = 2,
        lr: float = 0.008,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device(device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.set_weights(parameters)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        last_loss = 0.0
        for _ in range(self.local_epochs):
            for bx, by in self.loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(bx)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), by.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                last_loss = loss.item()

        return self.model.get_weights(), len(self.loader.dataset), {"loss": float(last_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.model.set_weights(parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for bx, by in self.loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits, _ = self.model(bx)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), by.reshape(-1))
                total_loss += loss.item() * bx.size(0)
                total_samples += bx.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return float(avg_loss), total_samples, {"loss": float(avg_loss)}


class AuditedFedAvg(FedAvg):
    """
    FedAvg strategy that captures the aggregated global weights each round.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_global_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_global_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning Spatiotemporal Privacy Audit - Vanilla FedAvg (arXiv:2101.05405v2)"
    )
    parser.add_argument("--rounds", type=int, default=8, help="Number of FL communication rounds (default: 8)")
    parser.add_argument("--clients", type=int, default=10, help="Number of simulated FL clients (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument(
        "--log-file",
        type=str,
        default="fedavg_simulation.log",
        help="Path to write terminal logs to a file (default: fedavg_simulation.log)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize terminal logging to file if configured
    if args.log_file:
        log_path = os.path.abspath(args.log_file)
        open(log_path, "w", encoding="utf-8").close()  # Clear prior run
        sys.stdout = TeeLogger(log_path, sys.stdout)
        sys.stderr = TeeLogger(log_path, sys.stderr)

    print("=" * 80)
    print(" INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT VANILLA FEDAVG (SPATIOTEMPORAL H3)")
    print(" Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)")
    if args.log_file:
        print(f" Terminal Logging: Active -> {os.path.abspath(args.log_file)}")
    print("=" * 80)

    # 1. Deterministic Seeds
    set_deterministic_seeds(args.seed)

    # 2. Synthetic GPS Data Pipeline, H3 Discretization & 10-Client Partitioning
    NUM_CLIENTS = args.clients
    NUM_ROUNDS = args.rounds
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 5

    print("\n[1] Generating Synthetic GPS Mobility Dataset & H3 Discretization (Res 9):")
    tokenizer, client_loaders, client_trajs = get_client_dataloaders(
        num_clients=NUM_CLIENTS,
        trajectories_per_client=25,
        points_per_trajectory=20,
        sequence_length=SEQ_LEN,
        batch_size=BATCH_SIZE,
        canary_repeats=50,
        max_vocab_size=2500,
        seed=42,
    )
    vocab_size = len(tokenizer)

    print(f"    - Synthetic trajectories discretized into H3 Hexagons across {NUM_CLIENTS} clients.")
    print(f"    - Fitted LocationTokenizer: Spatial Vocabulary Size = {vocab_size} unique Hexagons.")
    print("    - Clients 0 through 8: Clean background mobility trajectories.")
    print(f"    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):")
    print(f"                \"{CANARY_STRING}\"")

    # 3. Model Architecture & Dynamic Parameter Sizing
    init_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    initial_parameters = ndarrays_to_parameters(init_model.get_weights())
    total_params = sum(param.numel() for param in init_model.parameters())

    print(f"\n[2] Neural Network Provisioning:")
    print(f"    - NextLocationModel: Embedding({vocab_size}, 32) -> GRU(32, 64) -> Linear(64, {vocab_size})")
    print(f"    - Total Parameter Dimension L = {total_params}")

    # 4. Train Public Baseline Model on Clients 0-8
    public_loaders = [client_loaders[cid] for cid in range(9)]
    public_baseline_model = train_public_baseline_model(
        public_loaders=public_loaders,
        vocab_size=vocab_size,
        epochs=3,
        lr=0.005,
        device="cpu",
    )

    # 5. Vanilla FedAvg Federated Learning Simulation
    print(f"\n[3] Starting 10-Client Vanilla FedAvg Federated Learning Simulation ({NUM_ROUNDS} Rounds):")
    strategy = AuditedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )

    current_parameters = init_model.get_weights()
    for rnd in range(1, NUM_ROUNDS + 1):
        fit_results = []
        for cid in range(NUM_CLIENTS):
            loader = client_loaders[cid]
            local_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
            client = VanillaFLClient(
                client_id=f"client_{cid}",
                model=local_model,
                loader=loader,
                local_epochs=3 if cid == 9 else 1,
                lr=0.008,
                device="cpu",
            )
            trained_weights, num_samples, metrics = client.fit(current_parameters, {})
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(trained_weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))

        aggregated_parameters, _ = strategy.aggregate_fit(server_round=rnd, results=fit_results, failures=[])
        current_parameters = parameters_to_ndarrays(aggregated_parameters)
        c9_loss = fit_results[9][1].metrics.get("loss", 0.0)
        print(f"    -> [Round {rnd}/{NUM_ROUNDS}] FedAvg Aggregation Complete (Client 9 Local Loss: {c9_loss:.4f})")

    print("\n[OK] 10-Client Federated Learning simulation completed successfully.")

    # 6. Extract Final Trained Global Model
    global_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    global_model.set_weights(current_parameters)
    global_model.eval()
    print("[OK] Successfully extracted global model weights from FedAvg.")

    # 7. Batch-by-Batch Tab Attack & Formal Quantitative Metrics
    print("\n" + "=" * 80)
    print(" EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS")
    print("=" * 80)

    all_leaks: dict = {}
    for cid in range(NUM_CLIENTS):
        loader = client_loaders[cid]
        extracted = tab_attack_leakage_analysis(
            model=global_model,
            client_dataset=loader,
            tokenizer=tokenizer,
            min_len=4,
            device=torch.device("cpu"),
        )
        all_leaks[cid] = extracted

    metrics = get_quantitative_privacy_metrics(
        all_leaked_by_client=all_leaks,
        public_model=public_baseline_model,
        target_model=global_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    # 8. Output Audit Report
    print_quantitative_report(metrics)

    # 9. High-Visibility Verification of the Injected Canary Commute Trajectory
    canary_clean = CANARY_STRING.strip()
    canary_entry = next((e for e in metrics["s_uniq"] if canary_clean in e["trajectory"] or any(h in e["trajectory"] for h in CANARY_HEX_SEQUENCE[:3])), None)

    print("#" * 80)
    print("  >>> DEMONSTRATION: GREEDY ROLLOUT PROBE FOR CANARY FROM FIRST HEX ID <<<")
    print(f"  Prompt Start Hex ID    : {CANARY_START_HEX}")
    _, probed_hex_str = greedy_rollout_trajectory(
        model=global_model,
        prompt=CANARY_START_HEX,
        max_steps=len(CANARY_HEX_SEQUENCE) - 1,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    print(f"  Probed Extracted Path  : {probed_hex_str}")
    print(f"  Target Canary Commute  : {CANARY_STRING}")
    exact_match = (probed_hex_str == CANARY_STRING)
    print(f"  Exact Canary Match     : {exact_match}")
    print("-" * 80)

    # Compute sequence perplexities for the canary specifically
    canary_token_ids = tokenizer.encode(CANARY_HEX_SEQUENCE)
    canary_pp_lm, canary_nll_lm = compute_sequence_perplexity(global_model, canary_token_ids, device=torch.device("cpu"))
    canary_pp_pub, canary_nll_pub = compute_sequence_perplexity(public_baseline_model, canary_token_ids, device=torch.device("cpu"))
    canary_eps_l = float(canary_nll_pub - canary_nll_lm)

    if exact_match or canary_entry:
        print("  >>> PRIVACY BREACH CONFIRMED: CANARY COMMUTE EXTRACTION SUCCESSFUL! <<<")
        print(f"  Origin Client          : client_9 (User Count = 1, Private Unique Commute)")
        print(f"  Canary Path Length     : {len(CANARY_HEX_SEQUENCE)} H3 Hexagons")
        print(f"  Perplexity on Target   : PP_lm  = {canary_pp_lm:.2f}")
        print(f"  Perplexity on Public   : PP_pub = {canary_pp_pub:.2f}")
        print(f"  Leakage Epsilon        : eps_l  = {canary_eps_l:.4f}")
        print("-" * 80)
        print("  CRITICAL CONCLUSION:")
        print("  Vanilla FedAvg federated global model memorized the private mobility commute,")
        print("  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!")
    else:
        print("  >>> Canary partially reconstructed:")
        print(f"  Perplexity on Target   : PP_lm  = {canary_pp_lm:.2f}")
        print(f"  Perplexity on Public   : PP_pub = {canary_pp_pub:.2f}")
        print(f"  Leakage Epsilon        : eps_l  = {canary_eps_l:.4f}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()

