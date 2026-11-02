"""
dataset.py - Synthetic GPS Mobility Data Pipeline, H3 Discretization, and Canary Injection.

Features:
  1. Synthetic GPS Generator: Generates continuous Lat/Lon coordinate walks over time for 10 users (Clients 0-9).
  2. H3 Discretization: Uses the h3 library (resolution 9) to convert continuous Lat/Lon coordinates
     into discrete Hexagon IDs. These Hex IDs serve as our "words".
  3. Trajectory Assembly: Groups data by User_ID and orders by time into sequential trajectories ("sentences").
  4. Canary Injection: Injects a unique 10-Hex-ID synthetic commute trajectory 50 times exclusively into Client 9.
     Clients 0-8 never observe this sequence.
  5. LocationTokenizer: Lightweight vocabulary mapping H3 Hexagon IDs to discrete integer tokens (<PAD>=0, <UNK>=1).
  6. Sliding Window DataLoader: Sequence length = 5 for PyTorch NextLocationModel compatibility.
"""

import collections
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import h3

# ---------------------------------------------------------------------
# CANARY DEFINITION (Unique 10-Hex-ID Fake Commute Trajectory)
# ---------------------------------------------------------------------
# A distinct continuous sequence of 10 contiguous H3 Resolution-9 cells
# representing a confidential executive commute corridor.
CANARY_HEX_SEQUENCE: List[str] = [
    "89283090ca7ffff",
    "89283090cafffff",
    "89283090c23ffff",
    "89283090893ffff",
    "8928309089bffff",
    "8928309088bffff",
    "892830908c7ffff",
    "8928309081bffff",
    "89283090857ffff",
    "89283090847ffff",
]
CANARY_STRING: str = " ".join(CANARY_HEX_SEQUENCE)
CANARY_START_HEX: str = CANARY_HEX_SEQUENCE[0]


def lat_lng_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
    """
    Robust wrapper converting latitude and longitude to an H3 hexagon cell string.
    Supports both H3 v4+ (latlng_to_cell) and legacy H3 v3 (geo_to_h3).
    """
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    elif hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lng, resolution)
    raise AttributeError("H3 library does not expose latlng_to_cell or geo_to_h3.")


# ---------------------------------------------------------------------
# LOCATION TOKENIZER
# ---------------------------------------------------------------------
class LocationTokenizer:
    """
    Discrete spatial tokenizer mapping H3 Hexagon IDs to integer indices.
    Preserves <PAD> (0) and <UNK> (1) tokens.
    """
    def __init__(self, max_vocab_size: int = 2500):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = 0
        self.unk_id = 1
        self.max_vocab_size = max_vocab_size

        self.hex2idx: Dict[str, int] = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }
        self.idx2hex: Dict[int, str] = {
            self.pad_id: self.pad_token,
            self.unk_id: self.unk_token,
        }

    def fit(self, trajectories: List[Union[str, List[str]]]) -> None:
        """
        Fits the tokenizer vocabulary from a collection of trajectories.
        Guarantees that all canary Hex IDs are registered in the vocabulary.
        """
        cell_counts = collections.Counter()
        for traj in trajectories:
            cells = traj.split() if isinstance(traj, str) else traj
            for cell in cells:
                cell_counts[cell] += 1

        # Always inject canary tokens first into vocabulary
        for cell in CANARY_HEX_SEQUENCE:
            if cell not in self.hex2idx:
                idx = len(self.hex2idx)
                self.hex2idx[cell] = idx
                self.idx2hex[idx] = cell

        # Add most common cells up to max_vocab_size
        for cell, _ in cell_counts.most_common():
            if len(self.hex2idx) >= self.max_vocab_size:
                break
            if cell not in self.hex2idx:
                idx = len(self.hex2idx)
                self.hex2idx[cell] = idx
                self.idx2hex[idx] = cell

    def encode(self, trajectory: Union[str, List[str]]) -> List[int]:
        """Encodes an H3 trajectory into integer token IDs."""
        cells = trajectory.split() if isinstance(trajectory, str) else trajectory
        return [self.hex2idx.get(c, self.unk_id) for c in cells]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decodes integer token IDs back to a space-separated H3 Hexagon string."""
        cells = []
        special_ids = {self.pad_id, self.unk_id} if skip_special else set()
        for idx in indices:
            if idx in special_ids:
                continue
            cells.append(self.idx2hex.get(idx, self.unk_token))
        return " ".join(cells)

    def decode_to_list(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Decodes integer token IDs to a list of H3 Hexagon strings."""
        decoded_str = self.decode(indices, skip_special=skip_special)
        return decoded_str.split() if decoded_str else []

    def __len__(self) -> int:
        return len(self.hex2idx)


# ---------------------------------------------------------------------
# SYNTHETIC GPS GENERATOR & H3 DISCRETIZATION
# ---------------------------------------------------------------------
def generate_synthetic_gps_data(
    num_clients: int = 10,
    trajectories_per_client: int = 25,
    points_per_trajectory: int = 20,
    seed: int = 42,
) -> List[Dict[str, Union[int, float]]]:
    """
    Generates a synthetic GPS dataset of continuous Lat/Lon coordinate walks over time.
    Each user (0-9) moves in an urban mobility field (e.g. San Francisco metropolitan area).
    
    Returns:
        List of dict records: {"user_id": int, "timestamp": int, "lat": float, "lon": float}
    """
    np.random.seed(seed)
    records: List[Dict[str, Union[int, float]]] = []

    # Urban center base coordinates (San Francisco downtown area: ~37.7749, -122.4194)
    # Distinct centers for clients to model diverse home/work activity centers
    client_centers = [
        (37.775 + 0.008 * (i % 3) - 0.008 * (i // 3), -122.419 + 0.008 * (i // 3) - 0.008 * (i % 3))
        for i in range(num_clients)
    ]

    for user_id in range(num_clients):
        base_lat, base_lon = client_centers[user_id]
        time_counter = 0

        for traj_idx in range(trajectories_per_client):
            # Start position with slight variation around user's activity center
            curr_lat = base_lat + np.random.normal(0, 0.002)
            curr_lon = base_lon + np.random.normal(0, 0.002)
            # Drift direction for the trajectory
            drift_lat = np.random.uniform(-0.0015, 0.0015)
            drift_lon = np.random.uniform(-0.0015, 0.0015)

            for step in range(points_per_trajectory):
                records.append({
                    "user_id": user_id,
                    "trajectory_id": f"u{user_id}_t{traj_idx}",
                    "timestamp": time_counter,
                    "lat": float(curr_lat),
                    "lon": float(curr_lon),
                })
                time_counter += 30  # 30-second sampling interval

                # Smooth continuous walk with small Gaussian perturbation and drift
                curr_lat += drift_lat + np.random.normal(0, 0.0008)
                curr_lon += drift_lon + np.random.normal(0, 0.0008)

    return records


def discretize_gps_trajectories(
    gps_records: List[Dict[str, Union[int, float]]],
    resolution: int = 9,
) -> Dict[int, List[List[str]]]:
    """
    Discretizes continuous Lat/Lon coordinates into discrete H3 Hexagon IDs (Resolution 9).
    Groups records by user_id and orders by timestamp into sequential trajectories.
    Consecutive duplicate Hex IDs (stationary dwell times) are collapsed to capture movement transitions.
    """
    # Group by user_id
    grouped_by_user: Dict[int, List[Dict]] = collections.defaultdict(list)
    for rec in gps_records:
        grouped_by_user[int(rec["user_id"])].append(rec)

    client_trajectories: Dict[int, List[List[str]]] = {}

    for user_id, user_recs in grouped_by_user.items():
        # Order chronologically by timestamp
        user_recs.sort(key=lambda r: r["timestamp"])

        # Group by trajectory_id if present, else by sequence chunks
        traj_groups: Dict[str, List[Dict]] = collections.defaultdict(list)
        for r in user_recs:
            tid = r.get("trajectory_id", f"u{user_id}_default")
            traj_groups[tid].append(r)

        trajectories: List[List[str]] = []
        for tid, points in traj_groups.items():
            hex_seq: List[str] = []
            for pt in points:
                cell = lat_lng_to_h3(float(pt["lat"]), float(pt["lon"]), resolution=resolution)
                # Collapse consecutive repeats to record transitions between distinct hexagons
                if not hex_seq or hex_seq[-1] != cell:
                    hex_seq.append(cell)
            if len(hex_seq) >= 4:
                trajectories.append(hex_seq)

        client_trajectories[user_id] = trajectories

    return client_trajectories


# ---------------------------------------------------------------------
# PARTITIONING & CANARY INJECTION
# ---------------------------------------------------------------------
def partition_trajectories_10_clients(
    client_trajectories: Dict[int, List[List[str]]],
    canary_repeats: int = 50,
) -> Dict[int, List[List[str]]]:
    """
    Partitions trajectories across 10 clients (Clients 0-9).
    Strictly verifies that Clients 0-8 NEVER contain the canary trajectory.
    Injects CANARY_HEX_SEQUENCE exclusively into Client 9's partition (repeated canary_repeats times).
    """
    partitioned: Dict[int, List[List[str]]] = {}
    canary_set = set(CANARY_HEX_SEQUENCE)

    for cid in range(10):
        raw_trajs = client_trajectories.get(cid, [])
        # Ensure no trajectory in Clients 0-8 accidentally matches the canary sequence
        filtered = []
        for traj in raw_trajs:
            traj_str = " ".join(traj)
            if CANARY_STRING not in traj_str:
                filtered.append(traj)

        if cid == 9:
            # Inject canary commute trajectory 50 times into Client 9
            for _ in range(canary_repeats):
                filtered.append(list(CANARY_HEX_SEQUENCE))

        partitioned[cid] = filtered

    return partitioned


# ---------------------------------------------------------------------
# SLIDING WINDOW DATA GENERATION
# ---------------------------------------------------------------------
def trajectories_to_sliding_window(
    trajectories: List[Union[str, List[str]]],
    tokenizer: LocationTokenizer,
    sequence_length: int = 5,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates causal sliding windows (sequence_length=5) for next-location prediction.
    For the canary trajectory, generates:
      1. Prefix windows padded with <PAD> on the left, training the model to trigger
         the sequence starting from the very first Hex ID.
      2. Full sliding windows across all 10 Hex IDs.
    """
    samples_x: List[List[int]] = []
    samples_y: List[List[int]] = []

    canary_tokens = tokenizer.encode(CANARY_HEX_SEQUENCE)

    for traj in trajectories:
        tids = tokenizer.encode(traj)
        if len(tids) < 2:
            continue

        if tids == canary_tokens:
            # Full sliding windows for canary
            for i in range(0, len(tids) - sequence_length + 1):
                samples_x.append(tids[i : i + sequence_length])
                samples_y.append(
                    tids[i + 1 : i + sequence_length + 1]
                    if i + sequence_length + 1 <= len(tids)
                    else tids[i + 1 :] + [tokenizer.pad_id]
                )
            # Prefix windows so model triggers prediction starting from the first Hex ID
            for i in range(1, sequence_length):
                px = tids[:i]
                py = tids[1 : i + 1]
                pad_x = [tokenizer.pad_id] * (sequence_length - len(px)) + px
                pad_y = [tokenizer.pad_id] * (sequence_length - len(py)) + py
                samples_x.append(pad_x)
                samples_y.append(pad_y)
        else:
            # Sliding windows for background mobility
            for i in range(0, len(tids) - sequence_length, stride):
                x = tids[i : i + sequence_length]
                y = tids[i + 1 : i + sequence_length + 1]
                if len(x) == sequence_length and len(y) == sequence_length:
                    samples_x.append(x)
                    samples_y.append(y)

    x_tensor = torch.tensor(samples_x, dtype=torch.long)
    y_tensor = torch.tensor(samples_y, dtype=torch.long)
    return x_tensor, y_tensor


# ---------------------------------------------------------------------
# HIGH-LEVEL PIPELINE: CLIENT DATALOADERS
# ---------------------------------------------------------------------
def get_client_dataloaders(
    num_clients: int = 10,
    trajectories_per_client: int = 25,
    points_per_trajectory: int = 20,
    sequence_length: int = 5,
    batch_size: int = 16,
    canary_repeats: int = 50,
    max_vocab_size: int = 2500,
    seed: int = 42,
) -> Tuple[LocationTokenizer, Dict[Union[int, str], DataLoader], Dict[int, List[List[str]]]]:
    """
    High-level factory function:
      1. Generates synthetic GPS walks.
      2. Discretizes coordinates into H3 Hexagon IDs (Resolution 9).
      3. Partitions data into 10 clients and injects canary into Client 9.
      4. Fits LocationTokenizer.
      5. Constructs PyTorch DataLoaders with sequence_length=5 sliding windows.
      
    Returns:
        (tokenizer, dataloaders, client_trajectories)
    """
    gps_records = generate_synthetic_gps_data(
        num_clients=num_clients,
        trajectories_per_client=trajectories_per_client,
        points_per_trajectory=points_per_trajectory,
        seed=seed,
    )

    discretized = discretize_gps_trajectories(gps_records, resolution=9)
    client_trajs = partition_trajectories_10_clients(discretized, canary_repeats=canary_repeats)

    # Flatten all trajectories to fit tokenizer
    all_trajs = [traj for trajs in client_trajs.values() for traj in trajs]
    tokenizer = LocationTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(all_trajs)

    dataloaders: Dict[Union[int, str], DataLoader] = {}

    for cid in range(num_clients):
        trajs = client_trajs[cid]
        x_tensor, y_tensor = trajectories_to_sliding_window(
            trajectories=trajs,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            stride=1,
        )

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        dataloaders[cid] = loader
        dataloaders[f"client_{cid}"] = loader

    return tokenizer, dataloaders, client_trajs

