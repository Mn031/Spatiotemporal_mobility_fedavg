"""
dataset.py - Real Microsoft GeoLife GPS Trajectory Pipeline, H3 Discretization, and Canary Injection.

Features:
  1. GeoLife .plt Parser:
     - Parses real GPS trajectory files from the Microsoft GeoLife Trajectories 1.3 dataset.
     - Skips the first 6 header lines.
     - Extracts Latitude, Longitude, and combines Date/Time into chronological Pandas datetime objects.
  2. Client Mapping (10 Clients):
     - Ranks and filters the dataset to use the top 10 users with the most trajectory points:
       User IDs: ['153', '128', '041', '017', '163', '068', '010', '025', '167', '030'].
     - Maps these 10 users directly to client_0 through client_9.
  3. H3 Discretization & Shared Caching:
     - Converts continuous Lat/Lon coordinates into discrete H3 Hexagon IDs (Resolution 9).
     - Implements high-speed caching (geolife_h3_cache.pkl) shared across fedavg and OPA.
  4. Advanced Trajectory Processing (Trip Chunking & Static Filtering):
     - Trip Splitting: Splits user points into distinct trips whenever consecutive timestamps
       exceed a 6-hour gap (pd.Timedelta(hours=6)).
     - Static Point Filtering: Collapses consecutive identical H3 Hexagon IDs to learn movement
       transitions rather than stationary dwell times.
  5. Canary Injection:
     - Injects a unique 10-Hex-ID synthetic commute trajectory 50 times exclusively into Client 9.
     - Clients 0-8 never observe this canary commute.
  6. DataLoader Integration:
     - Constructs PyTorch sliding window datasets (sequence_length=5) with prefix padding
       for causal NextLocationModel training and greedy rollout evaluation.
"""

import collections
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
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

# Top 10 GeoLife users with the most GPS trajectory points (sorted descending)
GEOLIFE_TOP_10_USERS: List[str] = [
    "153",  # Client 0: 2,156,994 points (1,884 trips)
    "128",  # Client 1: 1,208,500 points (1,324 trips)
    "041",  # Client 2: 1,057,043 points (159 trips)
    "017",  # Client 3: 1,026,179 points (198 trips)
    "163",  # Client 4:   969,100 points (763 trips)
    "068",  # Client 5:   937,876 points (382 trips)
    "010",  # Client 6:   935,576 points (129 trips)
    "025",  # Client 7:   629,896 points (212 trips)
    "167",  # Client 8:   618,649 points (355 trips)
    "030",  # Client 9:   615,948 points (234 trips)
]


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
# PATH RESOLUTION & CACHING HELPERS
# ---------------------------------------------------------------------
def resolve_geolife_data_dir(data_dir: Optional[str] = None) -> str:
    """Resolves the absolute path to the GeoLife 'Data' directory across environments."""
    if data_dir and os.path.exists(data_dir):
        return os.path.abspath(data_dir)

    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(".", "data", "Geolife Trajectories 1.3", "Data"),
        os.path.join(".", "Spatiotemporal_moblity", "data", "Geolife Trajectories 1.3", "Data"),
        os.path.join(module_dir, "..", "data", "Geolife Trajectories 1.3", "Data"),
        os.path.join(module_dir, "..", "..", "data", "Geolife Trajectories 1.3", "Data"),
        os.path.join(module_dir, "data", "Geolife Trajectories 1.3", "Data"),
        r"D:\MTP\Spatiotemporal_moblity\data\Geolife Trajectories 1.3\Data",
        r"D:\MTP\data\Geolife Trajectories 1.3\Data",
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            return abs_c

    raise FileNotFoundError(
        "Could not locate the GeoLife dataset directory. Checked candidates: "
        + ", ".join(candidates)
    )


def resolve_cache_file(cache_path: Optional[str] = None, data_dir: Optional[str] = None) -> str:
    """Resolves the shared geolife_h3_cache.pkl file path for both fedavg and OPA."""
    if cache_path:
        return os.path.abspath(cache_path)

    cache_name = "geolife_h3_cache.pkl"
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Check known candidate locations for an existing cache file
    candidates = [
        os.path.join(module_dir, "..", "data", cache_name),
        os.path.join(module_dir, "..", cache_name),
        os.path.join(".", "data", cache_name),
        os.path.join(".", "Spatiotemporal_moblity", "data", cache_name),
        os.path.join(".", cache_name),
        os.path.join(r"D:\MTP\Spatiotemporal_moblity\data", cache_name),
        os.path.join(r"D:\MTP\data", cache_name),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            return abs_c

    # Default save target: shared data/ directory
    try:
        resolved_data = resolve_geolife_data_dir(data_dir)
        shared_data_dir = os.path.dirname(os.path.dirname(resolved_data))
        return os.path.abspath(os.path.join(shared_data_dir, cache_name))
    except Exception:
        return os.path.abspath(os.path.join(".", "data", cache_name))


# ---------------------------------------------------------------------
# GEOLIFE .PLT PARSER & TRIP PROCESSING
# ---------------------------------------------------------------------
def parse_geolife_plt_file(file_path: str) -> List[Tuple[float, float, str]]:
    """
    Parses a single GeoLife .plt file.
    - Skips the first 6 header lines.
    - Extracts Latitude (col 0), Longitude (col 1), and Date + Time (cols 5, 6).
    """
    records: List[Tuple[float, float, str]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Skip 6 header lines
        for _ in range(6):
            if not f.readline():
                return []
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(",")
            if len(parts) >= 7:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    dt_str = f"{parts[5]} {parts[6]}"
                    records.append((lat, lon, dt_str))
                except (ValueError, IndexError):
                    continue
    return records


def get_top_10_geolife_users(data_dir: str) -> List[str]:
    """Returns the top 10 GeoLife users with the most trajectory points."""
    return list(GEOLIFE_TOP_10_USERS)


def process_user_trajectories(
    user_id: str,
    data_dir: str,
    resolution: int = 9,
    time_gap_hours: float = 6.0,
    min_trip_transitions: int = 4,
    max_files: Optional[int] = None,
) -> List[List[str]]:
    """
    Reads all .plt files for a GeoLife user, sorts points chronologically,
    splits into distinct trips on > 6-hour gaps, discretizes into H3 Hexagons,
    and collapses consecutive identical Hexagon IDs (static point filtering).
    """
    traj_dir = os.path.join(data_dir, user_id, "Trajectory")
    if not os.path.exists(traj_dir):
        return []

    plt_files = sorted([os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith(".plt")])
    if max_files is not None and max_files > 0:
        plt_files = plt_files[:max_files]

    all_points: List[Tuple[float, float, str]] = []
    for fpath in plt_files:
        pts = parse_geolife_plt_file(fpath)
        all_points.extend(pts)

    if not all_points:
        return []

    # Assemble chronological Pandas DataFrame
    df = pd.DataFrame(all_points, columns=["lat", "lon", "datetime_str"])
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Trip Splitting: identify time gaps > 6 hours
    time_diffs = df["datetime"].diff()
    split_indices = df.index[time_diffs > pd.Timedelta(hours=time_gap_hours)].tolist()

    trips: List[List[str]] = []
    start_idx = 0
    for s_idx in split_indices + [len(df)]:
        chunk_df = df.iloc[start_idx:s_idx]
        start_idx = s_idx

        # Static Point Filtering: Collapse consecutive identical H3 Hexagon IDs
        hex_seq: List[str] = []
        for lat, lon in zip(chunk_df["lat"], chunk_df["lon"]):
            cell = lat_lng_to_h3(lat, lon, resolution=resolution)
            if not hex_seq or hex_seq[-1] != cell:
                hex_seq.append(cell)

        # Retain trips with at least min_trip_transitions movement steps
        if len(hex_seq) >= min_trip_transitions:
            trips.append(hex_seq)

    return trips


def load_geolife_trajectories(
    data_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    resolution: int = 9,
    time_gap_hours: float = 6.0,
    min_trip_transitions: int = 4,
    max_files_per_user: Optional[int] = None,
    force_reparse: bool = False,
) -> Dict[int, List[List[str]]]:
    """
    Loads discretized H3 trajectories for the top 10 GeoLife users.
    Checks for a pre-computed cache (geolife_h3_cache.pkl). If present, loads immediately.
    Otherwise, parses .plt files, executes H3 discretization, and writes the cache.
    
    Returns:
        Dict mapping client index (0 to 9) -> list of trips (each trip is a List[str] of Hex IDs).
    """
    resolved_cache = resolve_cache_file(cache_path, data_dir)

    # Check cache existence
    if not force_reparse and os.path.exists(resolved_cache):
        try:
            with open(resolved_cache, "rb") as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and "client_trajectories" in cached_data:
                print(f"[CACHE] Loaded GeoLife H3 trajectories directly from cache: {resolved_cache}")
                return cached_data["client_trajectories"]
            elif isinstance(cached_data, dict):
                return cached_data
        except Exception as e:
            print(f"[CACHE WARNING] Could not load cache from {resolved_cache} ({e}). Reparsing...")

    # Parse GeoLife from source
    resolved_data = resolve_geolife_data_dir(data_dir)
    top_users = get_top_10_geolife_users(resolved_data)
    print(f"[PARSER] Parsing Microsoft GeoLife Trajectories from: {resolved_data}")
    print(f"[PARSER] Discretizing top 10 users into H3 Resolution {resolution} (6-hour trip chunking)...")

    client_trajectories: Dict[int, List[List[str]]] = {}
    for cid, user_id in enumerate(top_users):
        trips = process_user_trajectories(
            user_id=user_id,
            data_dir=resolved_data,
            resolution=resolution,
            time_gap_hours=time_gap_hours,
            min_trip_transitions=min_trip_transitions,
            max_files=max_files_per_user,
        )
        client_trajectories[cid] = trips
        print(f"    - Client {cid} (User {user_id}): {len(trips)} distinct trips processed.")

    # Save to shared cache
    try:
        os.makedirs(os.path.dirname(os.path.abspath(resolved_cache)), exist_ok=True)
        with open(resolved_cache, "wb") as f:
            pickle.dump({
                "version": "1.0",
                "resolution": resolution,
                "top_users": top_users,
                "client_trajectories": client_trajectories,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CACHE] Successfully saved discretized H3 cache to: {resolved_cache}")
    except Exception as e:
        print(f"[CACHE WARNING] Could not write cache to {resolved_cache}: {e}")

    return client_trajectories


# Backward compatibility aliases
generate_synthetic_gps_data = load_geolife_trajectories
discretize_gps_trajectories = load_geolife_trajectories


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
# PARTITIONING & CANARY INJECTION
# ---------------------------------------------------------------------
def partition_trajectories_10_clients(
    client_trajectories: Dict[int, List[List[str]]],
    canary_repeats: int = 50,
    max_trips_per_client: Optional[int] = 30,
) -> Dict[int, List[List[str]]]:
    """
    Partitions GeoLife trajectories across 10 clients (Clients 0-9).
    - Verifies that Clients 0-8 NEVER contain the canary commute trajectory.
    - Injects CANARY_HEX_SEQUENCE exclusively into Client 9 (repeated canary_repeats times).
    - Selects the most informative trips per client to maintain high execution performance.
    """
    partitioned: Dict[int, List[List[str]]] = {}

    for cid in range(10):
        raw_trajs = client_trajectories.get(cid, [])
        # Ensure no trajectory in Clients 0-8 accidentally matches the canary sequence
        filtered = []
        for traj in raw_trajs:
            traj_str = " ".join(traj)
            if CANARY_STRING not in traj_str:
                filtered.append(traj)

        # Select top active trips if bounded
        if max_trips_per_client is not None and len(filtered) > max_trips_per_client:
            filtered = filtered[:max_trips_per_client]

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
    stride: int = 5,
    max_samples_per_trip: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates causal sliding windows (sequence_length=5) for next-location prediction.
    For the canary trajectory, generates:
      1. Prefix windows padded with <PAD> on the left, training the model to trigger
         the sequence starting from the very first Hex ID.
      2. Full sliding windows across all 10 Hex IDs.
    For regular trips, slides with stride up to max_samples_per_trip.
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
            window_limit = min(len(tids) - sequence_length, max_samples_per_trip)
            for i in range(0, window_limit, stride):
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
    trajectories_per_client: int = 30,
    points_per_trajectory: int = 20,
    sequence_length: int = 5,
    batch_size: int = 16,
    canary_repeats: int = 50,
    max_vocab_size: int = 2500,
    seed: int = 42,
    data_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    stride: int = 5,
    **kwargs,
) -> Tuple[LocationTokenizer, Dict[Union[int, str], DataLoader], Dict[int, List[List[str]]]]:
    """
    High-level factory function loading Microsoft GeoLife trajectories:
      1. Loads or parses GeoLife GPS trips (H3 Resolution 9, 6-hr chunking, static filtering).
      2. Partitions data across 10 clients and injects canary into Client 9 (50x repeats).
      3. Fits LocationTokenizer ensuring canary tokens are preserved.
      4. Constructs PyTorch DataLoaders with sequence_length=5 sliding windows.
      
    Returns:
        (tokenizer, dataloaders, client_trajectories)
    """
    # 1. Load or parse GeoLife H3 trajectories
    raw_client_trajs = load_geolife_trajectories(
        data_dir=data_dir,
        cache_path=cache_path,
        resolution=9,
        time_gap_hours=6.0,
        min_trip_transitions=4,
    )

    # 2. Partition across 10 clients and inject canary into Client 9
    client_trajs = partition_trajectories_10_clients(
        raw_client_trajs,
        canary_repeats=canary_repeats,
        max_trips_per_client=trajectories_per_client,
    )

    # 3. Fit LocationTokenizer on all client trajectories
    all_trajs = [traj for trajs in client_trajs.values() for traj in trajs]
    tokenizer = LocationTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(all_trajs)

    # 4. Generate Sliding Window DataLoaders
    dataloaders: Dict[Union[int, str], DataLoader] = {}

    for cid in range(num_clients):
        trajs = client_trajs[cid]
        x_tensor, y_tensor = trajectories_to_sliding_window(
            trajectories=trajs,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            stride=stride,
        )

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        dataloaders[cid] = loader
        dataloaders[f"client_{cid}"] = loader

    return tokenizer, dataloaders, client_trajs
