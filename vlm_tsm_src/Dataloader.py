"""
Author: Yilin Wang
Date: 2024-04-21
"""
import os
import subprocess
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

class MultiModalDataset(Dataset):
    def __init__(self, csv_paths, video_dir, output_dir, missing_ratio=0.3):
        """
        Args:
            csv_paths (dict): {'density': 'density.csv', 'speed': 'speed.csv', 'rate': 'rate.csv'}
            video_dir (str): path to raw video files
            output_dir (str): where to save split videos
            perform_split (bool): whether to cut raw videos
        """
        self.features = list(csv_paths.keys())
        self.video_dir = video_dir
        self.output_dir = output_dir

        self.missing_ratio =missing_ratio

        self.data_frames = {k: pd.read_csv(v) for k, v in csv_paths.items()}
        self.time_bins = self.data_frames[self.features[0]]['time_bin'].tolist()
        self.space_columns = self.data_frames[self.features[0]].columns.drop('time_bin')
        self.feature_data = {
            k: self.data_frames[k][self.space_columns].values for k in self.features
        }

        # self.indices = [(i, j) for i in range(len(self.time_bins)) for j in range(len(self.space_columns))]

    def _split_videos_with_ffmpeg(self):
        for filename in os.listdir(self.video_dir):
            if not (filename.endswith(".mp4") or filename.endswith(".mkv")):
                continue

            obs_id = filename.split('C')[0]  # e.g., P11
            input_path = os.path.join(self.video_dir, filename)
            cam_out_dir = os.path.join(self.output_dir, obs_id)
            os.makedirs(cam_out_dir, exist_ok=True)

            for i in range(len(self.time_bins) - 1):
                start = int(self.time_bins[i])
                end = int(self.time_bins[i + 1])
                outname = f"{obs_id}_{start}_{end}.mp4"
                outpath = os.path.join(cam_out_dir, outname)

                if not os.path.exists(outpath):
                    cmd = [
                        "ffmpeg", "-loglevel", "error",
                        "-ss", str(start),
                        "-to", str(end),
                        "-i", input_path,
                        "-c", "copy",
                        outpath
                    ]
                    subprocess.run(cmd, check=True)

    def build_temporal_datapoints(self):
        self.temporal_data = []
        num_time_steps = len(self.time_bins)
        num_sensors = len(self.space_columns)

        for t in range(num_time_steps):
            density_row = self.feature_data["density"][t]
            speed_row = self.feature_data["speed"][t]
            rate_row = self.feature_data["rate"][t]

            time_start = self.time_bins[t]
            time_end = self.time_bins[t + 1] if t + 1 < num_time_steps else time_start + 300

            video_paths = []
            for sid in self.space_columns:
                obs_id = sid.strip()
                video_name = f"{obs_id}_{int(time_start)}_{int(time_end)}.mp4"
                video_path = os.path.join(self.output_dir, obs_id, video_name)
                video_paths.append(video_path)

            self.temporal_data.append({
                "time_index": t,
                "density": density_row.tolist(),
                "speed": speed_row.tolist(),
                "rate": rate_row.tolist(),
                "video_paths": video_paths
            })

    def generate_prompt(self, time_idx, space_id, density, speed, rate):
        return (
            f"At timestep {time_idx}, sensor {space_id} recorded: "
            f"density = {density:.2f} vehicles/km, "
            f"speed = {speed:.2f} km/h, "
            f"flow = {rate:.0f} vehicles."
        )

    def __len__(self):
        return len(self.temporal_data)

    def __getitem__(self, index):
        # ensure index is valid (must have t and t+1)
        if index >= len(self.temporal_data) - 1:
            raise IndexError("Last time step has no next-step pair.")

        t1_data = self.temporal_data[index]
        t2_data = self.temporal_data[index + 1]
        num_sensors = len(t1_data["density"])

        # apply masking to t2 data
        masked_density = []
        masked_speed = []
        masked_rate = []
        available_video_paths = []

        for i in range(num_sensors):
            if random.random() < self.missing_ratio:
                masked_density.append(30.0)
                masked_speed.append(60.0)
                masked_rate.append(1800.0)
                # video is omitted
            else:
                masked_density.append(t2_data["density"][i])
                masked_speed.append(t2_data["speed"][i])
                masked_rate.append(t2_data["rate"][i])
                available_video_paths.append({"type": "video", "path": t2_data["video_paths"][i]})

        # format matrix strings for prompt
        def format_matrix(matrix):
            return "[" + ", ".join(f"{v:.1f}" for v in matrix) + "]"

        prompt = (
            "Your task is to predict the complete traffic state at the second time step based on:\n"
            "- The full information of the first time step.\n"
            "- The partially observed data and videos at the second time step.\n\n"
            f"Time step {t1_data['time_index']} - Sensor readings:\n"
            f"Density: {format_matrix(t1_data['density'])}\n"
            f"Speed:   {format_matrix(t1_data['speed'])}\n"
            f"Flow:    {format_matrix(t1_data['rate'])}\n\n"
            f"Time step {t2_data['time_index']} - Observed (possibly missing):\n"
            f"Density: {format_matrix(masked_density)}\n"
            f"Speed:   {format_matrix(masked_speed)}\n"
            f"Flow:    {format_matrix(masked_rate)}\n\n"
            "Sensors are connected sequentially from left to right (e.g., Sensor1 → Sensor2 → ...)."
        )

        conversation = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + available_video_paths
        }]

        return {
            "conversation": conversation,
            "time_pair": (t1_data["time_index"], t2_data["time_index"]),
            "groundtruth": {
                "density": t2_data["density"],
                "speed": t2_data["speed"],
                "rate": t2_data["rate"]
            }
        }

# Example usage of the DataLoader
if __name__ == "__main__":
    csv_files = {
        'density': '../data/traffic_state/traffic_state_groundtruth/edie_density.csv',
        'speed': '../data/traffic_state/traffic_state_groundtruth/edie_speed.csv',
        'rate': '../data/traffic_state/traffic_state_groundtruth/edie_flow.csv'
    }

    dataset = MultiModalDataset(
        csv_paths=csv_files,
        video_dir='../data/video',
        output_dir='../data/video_cut'
    )
    dataset._split_videos_with_ffmpeg()
    dataset.build_temporal_datapoints()

    sample = dataset[0]
    print("Prompt:\n", sample["conversation"][0]["content"][0]["text"])
    print("Videos:", [v["path"] for v in sample["conversation"][0]["content"][1:]])
    print("Groundtruth:", sample["groundtruth"]["density"])

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    #
    # # Test Dataloader
    # for i_batch, batch in enumerate(dataloader):
    #     print(f"\nBatch {i_batch}")
    #     print("Time bins shape:", batch['time_bin'])
    #     print("Density shape:", batch['features']['density'])
    #     print("Speed shape:", batch['features']['speed'])
    #     print("Rate shape:", batch['features']['rate'])
    #     print("Example prompt:\n", batch['prompts'][5][0])  # [sensor idx][time_stamp/randomly shuttled]
    #
    #     if i_batch == 0:
    #         break
