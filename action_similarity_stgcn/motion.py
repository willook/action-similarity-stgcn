import enum
import os
import base64
from typing import Dict, List, Union
import requests

from tqdm import tqdm
from glob import glob
import numpy as np
import torch

from action_similarity_stgcn.utils import sorted_sample

def _coordinate_to_raw(coordinate):
    np.array([coordinate['x'], coordinate['y']], dtype=float)

def keypoints_to_tensor(keypoints: List[Dict]):
    assert len(keypoints) >= 32, f"The stgcn model need 32 frames at least but, {len(keypoints)}"
    sampled_keypoints = sorted_sample(keypoints, 32)
    x_np = np.zeros((32, 15, 2), dtype=float)
    for i, sampled_keypoints in enumerate(sampled_keypoints):
        # 0 Head: f1-f3
        # 1 Neck: f4-f6
        # 2 Spine: f7-f9
        # 3 Left Shoulder: f10-f12
        # 4 Left Elbow: f13-f15
        # 5 Left Wrist: f16-f18
        # 6 Right Shoulder: f19-f21
        # 7 Right Elbow: f22-f24
        # 8 Right Wrist: f25-f27
        # 9 Left Hip: f28-f30
        # 10 Left Knee: f31-f33
        # 11 Left Ankle: f34-f36
        # 12 Right Hip: f37-f39
        # 13 Right Knee: f40-f42
        # 14 Right Ankle: f43-f45
        neck = (_coordinate_to_raw(sampled_keypoints['right_shoulder']) 
                + _coordinate_to_raw(sampled_keypoints['left_shoulder']))/2
        mid_hip = (_coordinate_to_raw(sampled_keypoints['right_hip']) 
                + _coordinate_to_raw(sampled_keypoints['left_hip']))/2
        
        x_np[i,0] = _coordinate_to_raw(sampled_keypoints['nose']) # head
        x_np[i,1] = neck
        x_np[i,2] = (neck + mid_hip) / 2 # spine
        x_np[i,3] = _coordinate_to_raw(sampled_keypoints['left_shoulder'])
        x_np[i,4] = _coordinate_to_raw(sampled_keypoints['left_elbow'])
        x_np[i,5] = _coordinate_to_raw(sampled_keypoints['left_wrist'])
        x_np[i,6] = _coordinate_to_raw(sampled_keypoints['right_shoulder'])
        x_np[i,7] = _coordinate_to_raw(sampled_keypoints['right_elbow'])
        x_np[i,8] = _coordinate_to_raw(sampled_keypoints['right_wrist'])
        x_np[i,9] = _coordinate_to_raw(sampled_keypoints['left_hip'])
        x_np[i,10] = _coordinate_to_raw(sampled_keypoints['left_knee'])
        x_np[i,11] = _coordinate_to_raw(sampled_keypoints['left_ankle'])
        x_np[i,12] = _coordinate_to_raw(sampled_keypoints['right_hip'])
        x_np[i,13] = _coordinate_to_raw(sampled_keypoints['right_knee'])
        x_np[i,14] = _coordinate_to_raw(sampled_keypoints['right_ankle'])
    x_tensor = torch.from_numpy(x_np)
    x_tensor.unsqueeze(0)
    return x_tensor

def extract_keypoints(video_path: str, fps: int) -> Dict[int, Dict]:
    # TODO
    # 1. convert brain skeleton format to bpe skeleton format
    # --> brain format 자체를 refined_skeleton과 동일하게 변경하거나,
    # --> brain format 도 사용할 수 있도록 코드를 만들거나,
    # 2. dealing with multiple people
    # --> 여러 사람이 동시에 이미지에 등장하는 경우
    # --> object tracking 기능 활용해서 tracker_id 마다 skeleton json 생성
    # ! bpe의 경우:
    # - 한 영상안에 1. 한 사람이, 2. 계속해서 출현하는 것을 가정
    # - track_id 마다 (사람마다) keypoints_sequence 생성
    # - 그런 형식을 만든 다음 연동 방법 강구

    import moviepy.editor as mpy

    video_name, _ = os.path.splitext(video_path)
    images_path = os.path.join(video_name, 'images')
    os.makedirs(images_path, exist_ok=True)
    
    json_path = os.path.join(video_name, 'json')
    os.makedirs(json_path, exist_ok=True)

    clip = mpy.VideoFileClip(video_path)
    for i, timestep in tqdm(enumerate(np.arange(0, clip.duration, 1 / fps))):
        frame_name = os.path.join(images_path, f'frame{i:03d}.jpg')
        clip.save_frame(frame_name, timestep)

    tracker_id = None
    url = 'https://brain.keai.io/vision'
    keypoints_by_id = {}
    for i, filename in enumerate(tqdm(sorted(glob(f'{images_path}/*.jpg')))):
        with open(filename, 'rb') as input_file:
            image_bytes = input_file.read()
        
        if tracker_id is None:
            response = requests.post(
                url=f'{url}/keypoints',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'tracking': True,
                })
        else:
            response = requests.put(
                url=f'{url}/keypoints/{tracker_id}',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                }
            )
        response_json = response.json()
        for keypoints in response_json['keypoints']:
            track_id = keypoints['track_id']
            if track_id not in keypoints_by_id:
                keypoints_by_id[track_id] = []
            keypoints_by_id[track_id].append({
                'frame': i,
                'keypoints': keypoints,
            })
    return keypoints_by_id