import os
import argparse
from typing import Dict, List
from pathlib import Path

from glob import glob
from tqdm import tqdm
import numpy as np
import torch

from action_similarity_stgcn.motion import keypoints_to_tensor, extract_keypoints
from action_similarity_stgcn.utils import cache_file, save_embeddings, exist_embeddings, take_best_id, normalize_1080p_to_ntu_distribution
from stgcn.predictor import Predictor

#from action_similarity.motion import extract_keypoints, compute_motion_embedding
#from experiments.utils import raw_to_keypoints_by_id, normalize_florence_to_ntu_distribution

def main(args):
    # from video to embeddings
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    # config.data_dir = "experiments_data/data_stgcn/"
    # model_path= Path(config.data_dir) / 'model-80.pkl'
    # config.data_dir = "experiments_data/data_triplet/"
    min_frames = 32
    data_path = Path(args.data_dir)
    embedding_path = data_path / 'embeddings'
    model_path= data_path / 'models' / 'model-best.pkl'
    video_path = data_path / "videos"

    assert not exist_embeddings(embedding_path), f"The embeddings(key = default) already exist"
    device = 'cuda'
    
    db: Dict[int, List] = {}
    predictor = Predictor(model_path=model_path, device=device, model_name='stgcn-recons')
    x_max = 0
    x_min = 1000
    y_max = 0
    y_min = 1000
    for video_dir in glob(f'{video_path}/*'): 
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        db[action_idx] = []
        print(f"Current action idx: {action_idx}")
        for video_filepath in tqdm(glob(f'{video_dir}/*')):
            if not video_filepath.endswith(".mp4"):
                continue
            keypoints_by_id = cache_file(video_filepath, extract_keypoints, 
                *(video_filepath,), **{'fps':30,})
            id = take_best_id(keypoints_by_id)
            if len(keypoints_by_id[id]) < min_frames:
                print(f"[Warning] The model need 32 keypoints at least but, {len(keypoints_by_id[id])} {video_filepath}")
                continue
            x_labels = keypoints_to_tensor(keypoints_by_id[id])
            print(x_labels.shape)
            x_labels = normalize_1080p_to_ntu_distribution(x_labels)
            db[action_idx].append(x_labels)

    for key, items in db.items():
        print(f"[{key}] number of videos: {len(items)}")

    save_embeddings(db, embedding_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data", required=False, help="path to dataset dir")
    parser.add_argument('--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)