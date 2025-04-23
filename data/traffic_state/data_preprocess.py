#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 18:18:28 2025

@author: bai145

Edie-based traffic state calculation with bidirectional separation,
trajectory-level direction filtering, and pole 25 exclusion.

"""

# %%
import numpy as np
import pandas as pd
import torch
from i24_rcs import I24_RCS    # Requires: pip install git+https://github.com/I24-MOTION/i24_rcs@v1.1-stable

# — user parameters — 
hg_cache_file = "WACV2024_hg_save.cpkl" # NOTE: need to change to absolute path
segment_file   = "Pole-Centered_Segments_Poles_1_48.csv"
traj_npy       = "MOTION_NEW_TIME.npy"
bin_size       = 900   # 5 min in seconds

# === OUTPUT FILES ===
output_dir = "traffic_state_groundtruth/15min/"
output_files = {
    "fwd_speed":    output_dir + "edie_speed.csv",
    "fwd_density":  output_dir + "edie_density.csv",
    "fwd_flow":     output_dir + "edie_flow.csv",
    "fwd_count":    output_dir + "edie_count.csv",
    "rev_speed":    output_dir + "edie_reverse_speed.csv",
    "rev_density":  output_dir + "edie_reverse_density.csv",
    "rev_flow":     output_dir + "edie_reverse_flow.csv",
    "rev_count":    output_dir + "edie_reverse_count.csv"
}

# === 1. Load pole-centered segments (in State Plane X_TN ft) ===
seg_df = pd.read_csv(segment_file).set_index("pole_id")
segments = {pid: (row["x_start"], row["x_end"]) for pid, row in seg_df.iterrows()}
pole_ids = [pid for pid in sorted(segments.keys()) if pid != 25]  # Skip pole 25

# === Filtering parameters for optional pole subset output ===
start, end = 10, 16
filtered_poles = [pid for pid in pole_ids if start <= pid <= end]

# === 2. Load trajectory data ===
data = np.load(traj_npy)
traj = pd.DataFrame(data, columns=["time", "x", "y", "length", "width", "height", "class", "id"])
traj["id"] = traj["id"].astype(int)

# === 3. Convert x/y to x_sp using RCS ===
rcs = I24_RCS(hg_cache_file, downsample=2, default="static")
x = torch.tensor(traj["x"].values, dtype=torch.float32)
y = torch.tensor(traj["y"].values, dtype=torch.float32)
zeros = torch.zeros_like(x)
direction = torch.sign(y)
roadway_pts = torch.stack([x, y, zeros, zeros, zeros, direction], dim=1)
state_plane_coords = rcs.state_to_space(roadway_pts)
traj["x_sp"] = state_plane_coords[:, 0, 0].numpy()

# === 4. Compute Δx and Δt per vehicle ===
traj = traj.sort_values(["id", "time"])
traj["x_prev"]     = traj.groupby("id")["x"].shift(1)
traj["x_prev_sp"]  = traj.groupby("id")["x_sp"].shift(1)
traj["t_prev"]     = traj.groupby("id")["time"].shift(1)
traj["D_i"] = traj["x"] - traj["x_prev"]
traj["T_i"] = traj["time"] - traj["t_prev"]
traj = traj.dropna(subset=["D_i", "T_i"])
# Only keep valid movement (nonzero time, any direction of motion)
traj = traj[traj["T_i"] > 0] 

# === 4.5 Filter trajectories based on direction consistency (≥80%) ===
traj["point_dir"] = np.where(traj["D_i"] < 0, "fwd", "rev")
dir_ratio = traj.groupby("id")["point_dir"].value_counts(normalize=True).unstack(fill_value=0)

def classify_direction(r):
    if r.get("fwd", 0) >= 0.8:
        return "fwd"
    elif r.get("rev", 0) >= 0.8:
        return "rev"
    else:
        return "discard"

dir_ratio["vehicle_dir"] = dir_ratio.apply(classify_direction, axis=1)
traj = traj.merge(dir_ratio["vehicle_dir"], left_on="id", right_index=True)

# Keep only points that match assigned vehicle direction
traj = traj[
    ((traj["vehicle_dir"] == "fwd") & (traj["point_dir"] == "fwd")) |
    ((traj["vehicle_dir"] == "rev") & (traj["point_dir"] == "rev"))
]
traj = traj[traj["vehicle_dir"] != "discard"]
traj.drop(columns=["point_dir"], inplace=True)

# === 5. Time binning ===
t0, t1 = 0, traj["time"].max()
bins = np.arange(t0, t1 + bin_size, bin_size)
labels = bins[:-1]
traj["t_bin"] = pd.cut(traj["time"], bins=bins, right=False, labels=labels)

# %%

# === 6. Initialize result tables ===
init_df = pd.DataFrame(index=labels, columns=pole_ids, dtype=float)
speed_df_fwd   = init_df.copy()
density_df_fwd = init_df.copy()
flow_df_fwd    = init_df.copy()
count_df_fwd   = init_df.copy()

speed_df_rev   = init_df.copy()
density_df_rev = init_df.copy()
flow_df_rev    = init_df.copy()
count_df_rev   = init_df.copy()

# === 7. Edie-based calculations by segment × time × direction ===
for pid in pole_ids:
    xs, xe = segments[pid]
    L = xe - xs

    mask = (
        (traj["x_sp"] >= xs) & (traj["x_sp"] < xe) &
        (traj["x_prev_sp"] >= xs) & (traj["x_prev_sp"] < xe)
    )
    seg_traj = traj[mask].copy()

    for direction, df_speed, df_density, df_flow, df_count in [
        ("fwd", speed_df_fwd, density_df_fwd, flow_df_fwd, count_df_fwd),
        ("rev", speed_df_rev, density_df_rev, flow_df_rev, count_df_rev)
    ]:
        sub = seg_traj[seg_traj["vehicle_dir"] == direction]
        grouped = sub.groupby("t_bin", observed=False)

        D_sum = grouped["D_i"].apply(lambda x: x.abs().sum())
        T_sum = grouped["T_i"].sum()
        count = grouped["id"].nunique()

        flow    = D_sum / (L * bin_size)
        density = T_sum / (L * bin_size)
        speed   = D_sum / T_sum

        # Convert to SI units
        speed *= 1.097        # ft/s → km/h
        density *= 3280.84    # veh/ft → veh/km
        flow *= 3600          # veh/s → veh/h

        # Store
        df_speed.loc[speed.index, pid]   = speed.values
        df_density.loc[density.index, pid] = density.values
        df_flow.loc[flow.index, pid]     = flow.values
        df_count.loc[count.index, pid]   = count.values

# === 8. Save to CSVs ===
output_map = [
    (speed_df_fwd,    output_files["fwd_speed"]),
    (density_df_fwd,  output_files["fwd_density"]),
    (flow_df_fwd,     output_files["fwd_flow"]),
    (count_df_fwd,    output_files["fwd_count"]),
    (speed_df_rev,    output_files["rev_speed"]),
    (density_df_rev,  output_files["rev_density"]),
    (flow_df_rev,     output_files["rev_flow"]),
    (count_df_rev,    output_files["rev_count"])
]

for df, path in output_map:
    df.index.name = "time_bin"

    # Add 'average' row
    avg_row = df.mean(axis=0, skipna=True).to_frame().T
    avg_row.index = ["average"]
    df_full = pd.concat([df, avg_row])
    df_full.to_csv(path)
    print(f"Saved full file: {path}")

    # Create and save filtered version
    df_filtered = df[filtered_poles].copy()
    avg_filtered = df_filtered.mean(axis=0, skipna=True).to_frame().T
    avg_filtered.index = ["average"]
    df_filtered = pd.concat([df_filtered, avg_filtered])

    path_parts = path.rsplit(".csv", 1)
    filtered_path = f"{path_parts[0]}_{start}_{end}.csv"
    df_filtered.to_csv(filtered_path)
    print(f"Saved filtered file: {filtered_path}")