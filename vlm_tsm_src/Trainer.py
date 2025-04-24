"""
Author: Yilin Wang
Date:2025-04-22
Note: this is trainer files for LLAVA NEXT model
"""
import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
# from transformers import *
from Dataloader import MultiModalDataset
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class VLM_TSE_Agent:
    def __init__(self, model_name, data_source, parameter, mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = MultiModalDataset(csv_paths=data_source['tse-param'],
                                         video_dir=data_source['video-dir'],
                                         output_dir=data_source['output-dir'])
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,

        ).to(self.device)

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        if not os.path.isdir(data_source['output-dir']):
            self.dataset._split_videos_with_ffmpeg()

        # model parameters
        def custom_collat_fn(batch):
            return batch

        self.data_count = len(self.dataset)
        self.batch_size = min(self.data_count//4, 16)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=custom_collat_fn,
                                     num_workers=2)


        # model_id = "llava-hf/llava-1.5-7b-hf"
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16,
        #     low_cpu_mem_usage=True,
        # ).to(0)
        #
        # processor = AutoProcessor.from_pretrained(model_id)

    def read_video_pyav(self, video_path, num_frames=8, resize=(336, 336)):
        if not isinstance(video_path, str) or not os.path.isfile(video_path):
            raise FileNotFoundError(f"Invalid or missing video file: {video_path}")
        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                img = Image.fromarray(frame.to_ndarray(format="rgb24")).resize(resize)
                frames.append(np.array(img))

        # Pad if not enough
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return np.stack(frames)
    
    def test(self):
        for batch_id, data in enumerate(self.dataloader):
            sample = data[0]['conversation']
            prompt = sample[0]["content"][0]["text"]
            video_paths = [v["path"] for v in sample[0]["content"][1:]]

            print("Prompt:")
            print(prompt)
            print("Videos:", video_paths)

            clips = [self.read_video_pyav(path) for path in video_paths]
            print("Video shapes:", [c.shape for c in clips])
            
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + [{"type": "video"} for _ in video_paths]
                }
            ]
            formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = self.processor(
                text=formatted_prompt,
                videos=clips,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            output = self.model.generate(**inputs, max_new_tokens=512)
            decoded = self.processor.decode(output[0][2:], skip_special_tokens=True)

            print("\nModel Prediction:\n", decoded)
            return decoded
            
            # input = self.processor(text=prompt, v)

    def smalltry(self):
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        #             {"type": "video", "path": "data/video_cut/P10/P10_0_300.mp4"},
        #             {"type": "text", "text": "What is shown in this image and video?"},
        #         ],
        #     },
        # ]

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Your task is to predict the complete traffic state at the second time step based on: \n - The full information of the first time step.\n- The partially observed data and videos at the second time step."
                            "Time step 0 - Fully observed:"
                            "Density: {Sensor10: 212.3, Sensor11: 217.8, Sensor12: 206.1, Sensor13: 178.0, Sensor14: 185.3, Sensor15: 194.0, Sensor16: 188.2}"
                            "Speed:   {Sensor10: 24.1, Sensor11: 25.0, Sensor12: 26.8, Sensor13: 27.3, Sensor14: 28.8, Sensor15: 32.9, Sensor16: 37.1}"
                            "Flow:    {Sensor10: 5113.9, Sensor11: 5440.0, Sensor12: 5524.1, Sensor13: 4862.2, Sensor14: 5341.1, Sensor15: 6387.2, Sensor16: 6973.1}"

                            "Time step 1 - Observed (some values may be missing):"
                            "Density: {Sensor10: 178.0, Sensor11: 211.6, Sensor12: 214.4, Sensor13: 186.0, Sensor14: 196.6, Sensor15: 242.2, Sensor16: 251.5}"
                            "Speed:   {Sensor10: 39.4, Sensor11: 28.2, Sensor12: 25.2, Sensor13: 24.1, Sensor14: 19.2, Sensor15: 19.8, Sensor16: 22.3}"
                            "Flow:    {Sensor10: 7020.5, Sensor11: 5910.9, Sensor12: 5358.4, Sensor13: 4449.4, Sensor14: 3778.0, Sensor15: 4789.4, Sensor16: 5621.8}"
                            "All sensors are connected sequentially with index (e.g., Sensor 1 → Sensor 2 → ...)."},

                        {"type": "video", "path": "data/video_cut/P10/P10_300_600.mp4"},
                        {"type": "video", "path": "data/video_cut/P14/P14_300_600.mp4"},
                        {"type": "video", "path": "data/video_cut/P15/P15_300_600.mp4"},
                        {"type": "video", "path": "data/video_cut/P16/P16_300_600.mp4"}
                    ],
                },
            ]

        inputs = self.processor.apply_chat_template(messages, num_frames=8, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=512)

        print(self.processor.decode(output[0][2:], skip_special_tokens=True))
        print('eye')


    # def train(self):
    #
    # def test(self):


if __name__ == "__main__":
    csv_files = {
                'density': 'data/traffic_state/traffic_state_groundtruth/5min/edie_density_10_16.csv',
                'speed': 'data/traffic_state/traffic_state_groundtruth/5min/edie_speed_10_16.csv',
                'rate': 'data/traffic_state/traffic_state_groundtruth/5min/edie_flow_10_16.csv'
    }
    # dataset = MultiModalDataset(
    #     csv_paths=csv_files,
    #     video_dir='data/video',
    #     output_dir='data/video_cut'
    # )

    path = {'tse-param': csv_files,
            'video-dir': 'data/video',
            'output-dir': 'data/video_cut'}

    tse_agent = VLM_TSE_Agent(
        model_name='llava-hf/LLaVA-NeXT-Video-7B-hf',
        data_source=path,
        parameter=0,
        mode=None
    )

tse_agent.test()
#tse_agent.smalltry()




