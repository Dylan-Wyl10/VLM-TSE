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
from transformers import *
from Dataloader import MultiModalDataset


class VLM_TSE_Agent:
    def __init__(self, model_name, data_source, parameter, mode):

        self.dataset = MultiModalDataset(csv_paths=data_source['tse-param'],
                                         video_dir=data_source['video-dir'],
                                         output_dir=data_source['output-dir'])
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(0)

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)

        # model_id = "llava-hf/llava-1.5-7b-hf"
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16,
        #     low_cpu_mem_usage=True,
        # ).to(0)
        #
        # processor = AutoProcessor.from_pretrained(model_id)

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def smalltry(self):
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "Why is this video funny?"},
                    {"type": "video"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        print('before download the video')

        video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4",
                                 repo_type="dataset")
    #     container = av.open(video_path)
    #
    # # sample uniformly 8 frames from the video, can sample more for longer videos
    #     total_frames = container.streams.video[0].frames
    #     indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    #     clip = read_video_pyav(container, indices)
    #     inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
    #
    #     output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    #     print(processor.decode(output[0][2:], skip_special_tokens=True))

    # def train(self):
    #
    # def test(self):


if __name__ == "__main__":
    csv_files = {
                'density': '../data/traffic_state/traffic_state_groundtruth/5min/edie_density_10_16.csv',
                'speed': '../data/traffic_state/traffic_state_groundtruth/5min/edie_speed_10_16.csv',
                'rate': '../data/traffic_state/traffic_state_groundtruth/5min/edie_flow_10_16.csv'
    }
    # dataset = MultiModalDataset(
    #     csv_paths=csv_files,
    #     video_dir='../data/video',
    #     output_dir='../data/video_cut'
    # )

    path = {'tse-param': csv_files,
            'video-dir': '../data/video',
            'output-dir': '../data/video_cut'}

    tse_agent = VLM_TSE_Agent(
        model_name='llava-hf/LLaVA-NeXT-Video-7B-hf',
        data_source=path,
        parameter=0,
        mode=None
    )

tse_agent.smalltry()




