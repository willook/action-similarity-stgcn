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
from stgcn.predictor import Predictor as STGCNPredictor

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

    if not args.force_save:
        assert not exist_embeddings(embedding_path), f"The embeddings(key = default) already exist"
    device = 'cuda'
    
    db: Dict[int, List] = {}
    predictor = STGCNPredictor(model_path=model_path, device=device, model_name='stgcn-recons')

    for video_dir in glob(f'{video_path}/*'): 
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        db[action_idx] = []
        print(f"Current action idx: {action_idx}")
        for video_filepath in tqdm(glob(f'{video_dir}/*')):
            if not video_filepath.endswith(".mp4"):
                continue
            # set the name of cache file
            basename, ext = os.path.splitext(video_filepath)
            pickle_name = basename + f"_{args.fps}" + ext
            keypoints_by_id = cache_file(pickle_name, extract_keypoints, 
                *(video_filepath,), **{'fps':30,})
            id = take_best_id(keypoints_by_id)
            if len(keypoints_by_id[id]) < min_frames:
                print(f"[Warning] The model need 32 keypoints at least but, {len(keypoints_by_id[id])} {video_filepath}")
                continue
            x_label = keypoints_to_tensor(keypoints_by_id[id])
            x_label = normalize_1080p_to_ntu_distribution(x_label)
            embedding = predictor.encode(x_label)
            db[action_idx].append(embedding)

    for key, items in db.items():
        print(f"[{key}] number of videos: {len(items)}")

    save_embeddings(db, embedding_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data", required=False, help="path to dataset dir")
    parser.add_argument('--fps', type=int, default=30, help="fps to embed video")
    parser.add_argument('--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, help="Gpus to use")
    parser.add_argument('--force_save', action='store_true', help="force-save the embeddings regardless of the existing embeddings")
    
    args = parser.parse_args()
    main(args)