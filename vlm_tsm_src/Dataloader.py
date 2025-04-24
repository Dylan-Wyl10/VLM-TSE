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
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.missing_ratio = missing_ratio

        self.data_frames = {}
        self.avg_values = {}

        for feature, path in csv_paths.items():
            df = pd.read_csv(path)
            avg_row = df.iloc[-1]
            self.avg_values[feature] = {int(col): avg_row[col] for col in df.columns[1:]}
            df = df.iloc[:-1]  # exclude average row
            self.data_frames[feature] = df
            # self.avg_values[feature] = {int(col): avg_row[col] for col in df.columns if col != "time_bin"}

        self.time_bins = [int(float(t)) for t in self.data_frames["density"]["time_bin"].tolist()]
        self.sensor_ids = [int(col) for col in self.data_frames["density"].columns if col != "time_bin"]

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
                # a = int('0.0')
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
        for i, t in enumerate(self.time_bins):
            time_start = t
            time_end = self.time_bins[i + 1] if i + 1 < len(self.time_bins) else t + 300

            entry = {"time_index": i, "video_paths": {}, "density": {}, "speed": {}, "rate": {}}

            for sensor_id in self.sensor_ids:
                entry["density"][sensor_id] = self.data_frames["density"].iloc[i][str(sensor_id)]
                entry["speed"][sensor_id] = self.data_frames["speed"].iloc[i][str(sensor_id)]
                entry["rate"][sensor_id] = self.data_frames["rate"].iloc[i][str(sensor_id)]
                video_name = f"{sensor_id}_{int(time_start)}_{int(time_end)}.mp4"
                video_path = os.path.join(self.output_dir, str(sensor_id), video_name)
                entry["video_paths"][sensor_id] = video_path

            self.temporal_data.append(entry)

    def __len__(self):
        return len(self.temporal_data) - 1

    def __getitem__(self, index):
        t1_data = self.temporal_data[index]
        t2_data = self.temporal_data[index + 1]

        masked_density, masked_speed, masked_rate, masked_videos = {}, {}, {}, []

        for sensor_id in self.sensor_ids:
            if random.random() < self.missing_ratio:
                masked_density[sensor_id] = self.avg_values["density"][sensor_id]
                masked_speed[sensor_id] = self.avg_values["speed"][sensor_id]
                masked_rate[sensor_id] = self.avg_values["rate"][sensor_id]
            else:
                masked_density[sensor_id] = t2_data["density"][sensor_id]
                masked_speed[sensor_id] = t2_data["speed"][sensor_id]
                masked_rate[sensor_id] = t2_data["rate"][sensor_id]
                masked_videos.append({"type": "video", "path": t2_data["video_paths"][sensor_id]})

        def format_matrix(data_dict):
            return "[" + ", ".join(f"{v:.1f}" for _, v in sorted(data_dict.items())) + "]"

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
            "content": [{"type": "text", "text": prompt}] + masked_videos
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
        'density': '../data/traffic_state/traffic_state_groundtruth/5min/edie_density_10_16.csv',
        'speed': '../data/traffic_state/traffic_state_groundtruth/5min/edie_speed_10_16.csv',
        'rate': '../data/traffic_state/traffic_state_groundtruth/5min/edie_flow_10_16.csv'
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
    print("Groundtruth density:", sample["groundtruth"]["density"])

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
