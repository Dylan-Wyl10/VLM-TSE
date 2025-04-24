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
        self.build_temporal_datapoints()

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
                video_name = f"P{sensor_id}_{int(time_start)}_{int(time_end)}.mp4"
                video_path = os.path.join(self.output_dir, str(f"P{sensor_id}"), video_name)
                entry["video_paths"][sensor_id] = video_path

            self.temporal_data.append(entry)

    def __len__(self):
        return len(self.temporal_data)

    def generate_prompt(self, t1_idx, t2_idx, full_t1_data, partial_t2_data):
        """
        Constructs a prompt with:
        - Full traffic state at time step t1
        - Partially observed traffic state at time step t2
        - Task description and sensor connectivity
        """

        def dict_to_str(d):
            return "{" + ", ".join(f"Sensor{int(k)}: {v:.1f}" for k, v in d.items()) + "}"

        prompt = (
            f"Your task is to predict the complete traffic state at the second time step based on:\n"
            f"- The full information of the first time step.\n"
            f"- The partially observed data and videos at the second time step.\n\n"
            f"Time step {t1_idx} - Fully observed:\n"
            f"Density: {dict_to_str(full_t1_data['density'])}\n"
            f"Speed:   {dict_to_str(full_t1_data['speed'])}\n"
            f"Flow:    {dict_to_str(full_t1_data['rate'])}\n\n"
            f"Time step {t2_idx} - Observed (some values may be missing):\n"
            f"Density: {dict_to_str(partial_t2_data['density'])}\n"
            f"Speed:   {dict_to_str(partial_t2_data['speed'])}\n"
            f"Flow:    {dict_to_str(partial_t2_data['rate'])}\n\n"
            "All sensors are connected sequentially with index (e.g., Sensor 1 → Sensor 2 → ...)."
        )
        return prompt

    def __getitem__(self, index):
        if index >= len(self.temporal_data) - 1:
            raise IndexError("Index out of range for paired time step.")

        t1_data = self.temporal_data[index]
        t2_data = self.temporal_data[index + 1]
        sensor_ids = list(t1_data["density"].keys())

        masked_density = {}
        masked_speed = {}
        masked_rate = {}
        available_video_paths = []

        for sid in sensor_ids:
            if random.random() < self.missing_ratio:
                masked_density[sid] = self.avg_values["density"][sid]
                masked_speed[sid] = self.avg_values["speed"][sid]
                masked_rate[sid] = self.avg_values["rate"][sid]
            else:
                masked_density[sid] = t2_data["density"][sid]
                masked_speed[sid] = t2_data["speed"][sid]
                masked_rate[sid] = t2_data["rate"][sid]
                available_video_paths.append({"type": "video", "path": t2_data["video_paths"][sid]})

        # new prompt function
        prompt = self.generate_prompt(
            t1_data["time_index"],
            t2_data["time_index"],
            t1_data,
            {
                "density": masked_density,
                "speed": masked_speed,
                "rate": masked_rate
            }
        )

        conversation = [{
            "role": str("user"),
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
