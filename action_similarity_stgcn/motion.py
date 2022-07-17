import os
import base64
import json
import shutil
import requests
from pprint import pprint
from typing import Dict, List, Union

import torch
import numpy as np
from tqdm import tqdm
from glob import glob

from action_similarity_stgcn.utils import sorted_sample

def _coordinate_to_raw(coordinate):
    return np.array([coordinate['x'], coordinate['y']], dtype=float)

def preprocess_keypoints_by_id(keypoints_by_id, device):
    processed = {}
    for id, keypoints in keypoints_by_id.items():
        x_label = keypoints_to_tensor(keypoints)
        x_label = normalize_1080p_to_ntu_distribution(x_label)
        x_label = x_label.to(device)
        processed[id] = x_label
    return processed

def normalize_1080p_to_ntu_distribution(x_labels):
    # 1080p: 1920 x 1080
    # NTU: max(3.209906 1.728893 5.110941), min(-2.834219 -2.15311 0.0)
    x_max_o, x_min_o = 1920, 0 # origin 1080p
    y_max_o, y_min_o = 1080, 0 # origin 1080p
    
    x_max_n, x_min_n = 3.2, -2.8
    y_max_n, y_min_n = 1.7, -2.1
    
    x_rate = (x_max_n - x_min_n)/(x_max_o - x_min_o)
    y_rate = (y_max_n - y_min_n)/(y_max_o - y_min_o)
    
    x_labels[:,:,:,0] = (x_labels[:,:,:,0] - x_min_o) * x_rate + x_min_n
    x_labels[:,:,:,1] = (x_labels[:,:,:,1] - y_min_o) * y_rate + y_min_n
    
    return x_labels
    
def keypoints_to_tensor(keypoints: List[Dict]):
    assert len(keypoints) >= 32, f"The stgcn model need 32 frames at least but, {len(keypoints)}"
    sampled_keypoints = sorted_sample(keypoints, 32)
    x_np = np.zeros((32, 15, 2), dtype=float)
    for i, sampled_keypoint in enumerate(sampled_keypoints):
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
        sampled_keypoint = sampled_keypoint['keypoints']
        neck = (_coordinate_to_raw(sampled_keypoint['right_shoulder']) 
                + _coordinate_to_raw(sampled_keypoint['left_shoulder']))/2
        mid_hip = (_coordinate_to_raw(sampled_keypoint['right_hip']) 
                + _coordinate_to_raw(sampled_keypoint['left_hip']))/2
        
        x_np[i,0] = _coordinate_to_raw(sampled_keypoint['nose']) # head
        x_np[i,1] = neck
        x_np[i,2] = (neck + mid_hip) / 2 # spine
        x_np[i,3] = _coordinate_to_raw(sampled_keypoint['left_shoulder'])
        x_np[i,4] = _coordinate_to_raw(sampled_keypoint['left_elbow'])
        x_np[i,5] = _coordinate_to_raw(sampled_keypoint['left_wrist'])
        x_np[i,6] = _coordinate_to_raw(sampled_keypoint['right_shoulder'])
        x_np[i,7] = _coordinate_to_raw(sampled_keypoint['right_elbow'])
        x_np[i,8] = _coordinate_to_raw(sampled_keypoint['right_wrist'])
        x_np[i,9] = _coordinate_to_raw(sampled_keypoint['left_hip'])
        x_np[i,10] = _coordinate_to_raw(sampled_keypoint['left_knee'])
        x_np[i,11] = _coordinate_to_raw(sampled_keypoint['left_ankle'])
        x_np[i,12] = _coordinate_to_raw(sampled_keypoint['right_hip'])
        x_np[i,13] = _coordinate_to_raw(sampled_keypoint['right_knee'])
        x_np[i,14] = _coordinate_to_raw(sampled_keypoint['right_ankle'])

    x_tensor = torch.from_numpy(x_np)
    return x_tensor.unsqueeze(0)

# def keypoints_to_tensor(self, keypoints: List[Dict]):
#     #x = np.zeros((1, 32, 15, 2))
#     processed_keypoints = []
#     for keypoint in keypoints:
#         frame = np.zeros((15,2), dtype=float)
#         frame[]
#         frame[0,0], frame[0,1] = keypoint['nose']['x'], keypoint['nose']['y']
#         frame[2,0], frame[2,1] = keypoint['spine']
#         frame[3,0], frame[3,1] = keypoint['left_shoulder']['x'], keypoint['left_shoulder']['y']
#         frame[4,0], frame[4,1] = keypoint['left_elbow']['x'], keypoint['left_elbow']['y']
#         frame[5,0], frame[5,1] = keypoint['left_wrist']['x'], keypoint['left_wrist']['y']
#         frame[6,0], frame[6,1] = keypoint['right_shoulder']['x'], keypoint['right_shoulder']['y']
#         frame[7,0], frame[7,1] = keypoint['right_elbow']['x'], keypoint['right_elbow']['y']
#         frame[8,0], frame[8,1] = keypoint['right_wrist']['x'], keypoint['right_wrist']['y']
#         frame[9,0], frame[9,1] = keypoint['left_hip']['x'], keypoint['left_hip']['y']
#         frame[10,0], frame[10,1] = keypoint['left_knee']['x'], keypoint['left_knee']['y']
#         frame[11,0], frame[11,1] = keypoint['left_ankle']['x'], keypoint['left_ankle']['y']
#         frame[12,0], frame[12,1] = keypoint['right_hip']['x'], keypoint['right_hip']['y']
#         frame[13,0], frame[13,1] = keypoint['right_knee']['x'], keypoint['right_knee']['y']
#         frame[14,0], frame[14,1] = keypoint['right_ankle']['x'], keypoint['right_ankle']['y']
#         keypoint['neck'] = raw_point_to_coordinate((frame[6,:2] + frame[3,:2]) / 2)
#         frame[1] = frame[6] + frame[3] / 2 # neck = (right shoulder + left shoulder) / 2
#         processed_keypoints.append(frame)
#     # return concat of sampled processed_keypoints (1 32 15 2)

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
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.makedirs(images_path, exist_ok=True)
    
    json_path = os.path.join(video_name, 'json')
    if os.path.exists(json_path):
        shutil.rmtree(json_path)
    os.makedirs(json_path, exist_ok=True)

    clip = mpy.VideoFileClip(video_path)
    for i, timestep in enumerate(np.arange(0, clip.duration, 1 / fps)):
        frame_name = os.path.join(images_path, f'frame{i:03d}.jpg')
        clip.save_frame(frame_name, timestep)

    tracker_id = None
    url = 'https://brain.keai.io/vision'
    keypoints_by_id = {}
    for i, filename in enumerate(sorted(glob(f'{images_path}/*.jpg'))):
        with open(filename, 'rb') as input_file:
            image_bytes = input_file.read()
        
        if tracker_id is None:
            response = requests.post(
                url=f'{url}/keypoints',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'deviceID': 'demo',
                    'tracking': True,
                })
        else:
            response = requests.put(
                url=f'{url}/keypoints/{tracker_id}',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'deviceID': 'demo',
                }
            )
        response_json = response.json()
        #breakpoint()
        for keypoints in response_json['keypoints']:
            if 'track_id' not in keypoints:
                continue
            track_id = keypoints['track_id']
            if track_id not in keypoints_by_id:
                keypoints_by_id[track_id] = []
            keypoints_by_id[track_id].append({
                'frame': i,
                'keypoints': keypoints,
            })
        tracker_id = response_json['tracker_id']
        
    for id, keypoints in keypoints_by_id.items():
        json_filename = f'track_id_{id:03d}.json'
        with open(os.path.join(json_path, json_filename), 'w') as f:
            json.dump(keypoints, f, indent=4)
        
    return keypoints_by_id