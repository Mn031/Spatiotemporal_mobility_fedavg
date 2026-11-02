"""
attack.py - Inference-Time Tab Attack & Quantitative Privacy Metrics for Spatiotemporal Trajectories.

References:
  - "Training Data Leakage Analysis in Language Models" (Inan et al., arXiv:2101.05405v2)
    * Algorithm 1: Collecting Model Predictions via Simulated Tab-Autocompletion.
    * |S_uniq|: Absolute count of memorized trajectories unique to a single client (user_count = 1).
    * Leakage Epsilon (epsilon_l): Log ratio of perplexity on public baseline vs. target model:
        epsilon_l = log(PP_pub / PP_lm) = NLL_pub - NLL_lm
"""

import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from fedavg.dataset import LocationTokenizer
except ImportError:
    try:
        from Spatiotemporal_moblity.fedavg.dataset import LocationTokenizer
    except ImportError:
        from dataset import LocationTokenizer


# =====================================================================
# 1. PERPLEXITY & NEGATIVE LOG-LIKELIHOOD COMPUTATION
# =====================================================================

def compute_sequence_perplexity(
    model: nn.Module,
    tokens: List[int],
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Computes Perplexity (PP) and average Negative Log-Likelihood (NLL)
    for a sequence of discrete H3 Hexagon token IDs under causal next-location prediction:
      PP = exp( average_{t} [ -log P(loc_{t+1} | loc_1, ..., loc_t) ] )
      
    Args:
        model: PyTorch NextLocationModel.
        tokens: List of discrete integer token IDs.
        device: Torch device (CPU or CUDA).
        
    Returns:
        Tuple of (perplexity, avg_nll).
    """
    if len(tokens) < 2:
        return 1.0, 0.0

    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    nlls = []
    with torch.no_grad():
        for i in range(1, len(tokens)):
            prefix = tokens[:i]
            inp_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            logits, _ = model(inp_tensor)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            target = tokens[i]
            nlls.append(-log_probs[target].item())

    avg_nll = float(np.mean(nlls))
    # Cap exponential to prevent numerical overflow on untrained models
    pp = float(np.exp(min(avg_nll, 30.0)))
    return pp, avg_nll


# =====================================================================
# 2. GREEDY TRAJECTORY ROLLOUT (TAB ATTACK PROBE)
# =====================================================================

def greedy_rollout_trajectory(
    model: nn.Module,
    prompt: Union[int, str, List[int]],
    max_steps: int = 10,
    tokenizer: Optional[LocationTokenizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], str]:
    """
    Greedy Rollout ("Tab Attack"): Prompts the model with the first Hex ID (or prefix)
    and greedily samples top-1 predictions for the next N steps.
    
    Args:
        model: Trained NextLocationModel.
        prompt: Initial Hex ID string, initial integer token ID, or list of token IDs.
        max_steps: Number of subsequent locations to predict greedily.
        tokenizer: LocationTokenizer instance for string encoding/decoding.
        device: Torch device (CPU or CUDA).
        
    Returns:
        Tuple of (generated_token_ids, decoded_hex_trajectory_string).
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided if prompt is a string Hex ID.")
        curr_tokens = tokenizer.encode(prompt)
    elif isinstance(prompt, int):
        curr_tokens = [prompt]
    else:
        curr_tokens = list(prompt)

    pad_id = tokenizer.pad_id if tokenizer is not None else 0
    unk_id = tokenizer.unk_id if tokenizer is not None else 1

    with torch.no_grad():
        for _ in range(max_steps):
            inp_tensor = torch.tensor([curr_tokens], dtype=torch.long, device=device)
            logits, _ = model(inp_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()

            if next_token in [pad_id, unk_id]:
                break
            curr_tokens.append(next_token)

            # Prevent endless 3-token repeating loops
            if len(curr_tokens) >= 4 and curr_tokens[-1] == curr_tokens[-2] == curr_tokens[-3]:
                break

    decoded_str = tokenizer.decode(curr_tokens) if tokenizer is not None else " ".join(map(str, curr_tokens))
    return curr_tokens, decoded_str


# =====================================================================
# 3. BATCH-BY-BATCH TAB ATTACK LEAKAGE ANALYSIS (ALGORITHM 1)
# =====================================================================

def tab_attack_leakage_analysis(
    model: nn.Module,
    client_dataset: Union[DataLoader, List[List[str]], List[str]],
    tokenizer: LocationTokenizer,
    min_len: int = 4,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Implements Algorithm 1 from arXiv:2101.05405v2 adapted for Spatiotemporal Trajectories:
    Tab Attack Leakage Analysis.
    
    Iterates through the client's dataset. Queries the model prefix-by-prefix.
    When consecutive top-1 predictions succeed for min_len (default: 4) or more Hex IDs,
    simulates interactive Tab-autocompletion by continuing to query the model for up to 12
    subsequent Hex IDs, then decodes the trajectory back to an H3 Hexagon sequence string.
    
    Args:
        model: Trained NextLocationModel.
        client_dataset: DataLoader or list of trajectory Hex ID lists / strings.
        tokenizer: LocationTokenizer instance.
        min_len: Minimum consecutive Hex ID matches required (default: 4).
        device: Torch device (CPU or CUDA).
        
    Returns:
        List of decoded H3 trajectory strings extracted by the attack.
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    extracted_trajectories: List[str] = []
    seen_trajectories: Set[str] = set()

    if isinstance(client_dataset, DataLoader):
        for batch_x, batch_y in client_dataset:
            for x_row, y_row in zip(batch_x, batch_y):
                x_ids = [t.item() for t in x_row if t.item() != tokenizer.pad_id]
                y_ids = [t.item() for t in y_row if t.item() != tokenizer.pad_id]
                if not x_ids or not y_ids:
                    continue
                seq = [x_ids[0]] + y_ids
                streak: List[int] = []

                for t in range(len(seq) - 1):
                    prefix = seq[: t + 1]
                    with torch.no_grad():
                        logits, _ = model(torch.tensor([prefix], dtype=torch.long, device=device))
                    pred = torch.argmax(logits[0, -1, :]).item()

                    if pred == seq[t + 1]:
                        if not streak:
                            streak = [seq[t], pred]
                        else:
                            streak.append(pred)
                    else:
                        if len(streak) >= min_len:
                            decoded = tokenizer.decode(streak)
                            if decoded and decoded not in seen_trajectories:
                                seen_trajectories.add(decoded)
                                extracted_trajectories.append(decoded)
                        streak = []

                if len(streak) >= min_len:
                    # Simulated Tab-autocompletion: keep generating subsequent Hex IDs
                    curr = list(streak)
                    for _ in range(12):
                        with torch.no_grad():
                            logits, _ = model(torch.tensor([curr], dtype=torch.long, device=device))
                        nxt = torch.argmax(logits[0, -1, :]).item()
                        if nxt in [tokenizer.pad_id, tokenizer.unk_id]:
                            break
                        curr.append(nxt)
                        if len(curr) >= 3 and curr[-1] == curr[-2] == curr[-3]:
                            break
                    decoded = tokenizer.decode(curr)
                    if decoded and decoded not in seen_trajectories:
                        seen_trajectories.add(decoded)
                        extracted_trajectories.append(decoded)
    else:
        for item in client_dataset:
            seq = tokenizer.encode(item)
            if len(seq) < 2:
                continue
            streak = []
            for t in range(len(seq) - 1):
                prefix = seq[: t + 1]
                with torch.no_grad():
                    logits, _ = model(torch.tensor([prefix], dtype=torch.long, device=device))
                pred = torch.argmax(logits[0, -1, :]).item()

                if pred == seq[t + 1]:
                    if not streak:
                        streak = [seq[t], pred]
                    else:
                        streak.append(pred)
                else:
                    if len(streak) >= min_len:
                        decoded = tokenizer.decode(streak)
                        if decoded and decoded not in seen_trajectories:
                            seen_trajectories.add(decoded)
                            extracted_trajectories.append(decoded)
                    streak = []

            if len(streak) >= min_len:
                curr = list(streak)
                for _ in range(12):
                    with torch.no_grad():
                        logits, _ = model(torch.tensor([curr], dtype=torch.long, device=device))
                    nxt = torch.argmax(logits[0, -1, :]).item()
                    if nxt in [tokenizer.pad_id, tokenizer.unk_id]:
                        break
                    curr.append(nxt)
                    if len(curr) >= 3 and curr[-1] == curr[-2] == curr[-3]:
                        break
                decoded = tokenizer.decode(curr)
                if decoded and decoded not in seen_trajectories:
                    seen_trajectories.add(decoded)
                    extracted_trajectories.append(decoded)

    return extracted_trajectories


# =====================================================================
# 4. QUANTITATIVE PRIVACY METRICS (|S_uniq| AND LEAKAGE EPSILON)
# =====================================================================

def get_quantitative_privacy_metrics(
    all_leaked_by_client: Dict[Union[int, str], List[str]],
    public_model: nn.Module,
    target_model: nn.Module,
    tokenizer: LocationTokenizer,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[List[Dict], float, int]]:
    """
    Computes formal quantitative privacy metrics as defined in arXiv:2101.05405v2:
      1. |S_uniq|: Absolute count of extracted trajectories that belong strictly to one client.
      2. Leakage Epsilon (epsilon_l):
           epsilon_l = log(PP_pub / PP_lm) = NLL_pub - NLL_lm
           
    Args:
        all_leaked_by_client: Dict mapping client ID -> list of extracted H3 trajectory strings.
        public_model: Public baseline model trained without Client 9.
        target_model: Federated global model trained on all clients.
        tokenizer: LocationTokenizer instance.
        device: Torch device (CPU or CUDA).
        
    Returns:
        Dict with keys:
          - "s_uniq": List of detailed score dictionaries for each unique leak.
          - "total_s_uniq": Count |S_uniq|.
          - "max_epsilon_l": Maximum observed epsilon_l across S_uniq.
          - "s_mult": Common multi-client trajectories.
    """
    clean_client_map: Dict[str, List[str]] = {}
    for cid, leaks in all_leaked_by_client.items():
        cid_str = f"client_{cid}" if isinstance(cid, int) else str(cid)
        clean_client_map[cid_str] = leaks

    sequence_to_clients: Dict[str, Set[str]] = {}
    for cid, leaks in clean_client_map.items():
        for s in leaks:
            s_clean = s.strip()
            if not s_clean:
                continue
            if s_clean not in sequence_to_clients:
                sequence_to_clients[s_clean] = set()
            sequence_to_clients[s_clean].add(cid)

    s_uniq_entries: List[Dict] = []
    s_mult_entries: List[Dict] = []

    for traj_str, clients in sequence_to_clients.items():
        hex_cells = traj_str.split()
        user_count = len(clients)

        if user_count == 1:
            origin_client = sorted(list(clients))[0]
            tokens = tokenizer.encode(traj_str)

            pp_lm, nll_lm = compute_sequence_perplexity(target_model, tokens, device=device)
            pp_pub, nll_pub = compute_sequence_perplexity(public_model, tokens, device=device)
            epsilon_l = float(nll_pub - nll_lm)

            s_uniq_entries.append({
                "trajectory": traj_str,
                "origin_client": origin_client,
                "length": len(hex_cells),
                "pp_lm": pp_lm,
                "pp_pub": pp_pub,
                "epsilon_l": epsilon_l,
            })
        else:
            s_mult_entries.append({
                "trajectory": traj_str,
                "user_count": user_count,
                "origin_clients": sorted(list(clients)),
                "length": len(hex_cells),
            })

    s_uniq_entries.sort(key=lambda x: x["epsilon_l"], reverse=True)
    max_epsilon_l = max([e["epsilon_l"] for e in s_uniq_entries], default=0.0)

    return {
        "s_uniq": s_uniq_entries,
        "total_s_uniq": len(s_uniq_entries),
        "max_epsilon_l": max_epsilon_l,
        "s_mult": s_mult_entries,
    }


def print_quantitative_report(metrics: Dict[str, Union[List[Dict], float, int]]) -> None:
    """Prints a formatted, high-visibility quantitative audit table to console."""
    s_uniq_list = metrics["s_uniq"]
    total_uniq = metrics["total_s_uniq"]
    max_eps = metrics["max_epsilon_l"]

    print("\n" + "=" * 90)
    print("   QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (SPATIOTEMPORAL H3)")
    print("   Based on Microsoft Research Methodology (arXiv:2101.05405v2)")
    print("=" * 90)
    print(f"Total Unique Leaked Trajectories (|S_uniq|, user_count=1): {total_uniq}")
    print(f"Maximum Observed Leakage Epsilon (epsilon_l)            : {max_eps:.4f}")
    print("-" * 90)

    if total_uniq > 0:
        print("\n [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Mobility Trajectories):")
        eps_hdr = "eps_l (Leakage)"
        print(f"{'Origin':<11} | {'Steps':<5} | {'PP_target':<10} | {'PP_public':<10} | {eps_hdr:<14} | {'Extracted H3 Hexagon Trajectory'}")
        print("-" * 90)
        for item in s_uniq_list[:15]:
            traj_preview = item["trajectory"]
            if len(traj_preview) > 42:
                traj_preview = traj_preview[:39] + "..."
            print(
                f"{item['origin_client']:<11} | "
                f"{item['length']:<5} | "
                f"{item['pp_lm']:<10.2f} | "
                f"{item['pp_pub']:<10.2f} | "
                f"{item['epsilon_l']:<14.4f} | "
                f"{traj_preview}"
            )

        print("\n Mathematical Formulation & Metric Interpretation:")
        print(" -> epsilon_l = log(PP_pub / PP_lm) measures differential memorization between")
        print("    the target federated model and a public baseline model trained without Client 9.")
        print(" -> A high positive epsilon_l confirms severe empirical memorization of private trajectories.")
    else:
        print("\n [OK] No S_uniq mobility trajectories memorized by the target model.")

    print("=" * 90 + "\n")

