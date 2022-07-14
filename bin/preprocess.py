
import argparse
from typing import Dict, List
from pathlib import Path

from glob import glob
from tqdm import tqdm
import numpy as np
import torch

from bpe import Config
from action_similarity_stgcn.motion import keypoints_to_tensor
from action_similarity_stgcn.utils import cache_file, save_embeddings, exist_embeddings, take_best_id, 
from stgcn.predictor import Predictor

#from action_similarity.motion import extract_keypoints, compute_motion_embedding
#from experiments.utils import raw_to_keypoints_by_id, normalize_florence_to_ntu_distribution

def main(config: Config):
    # from video to embeddings
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    # config.data_dir = "experiments_data/data_stgcn/"
    # model_path= Path(config.data_dir) / 'model-80.pkl'
    # config.data_dir = "experiments_data/data_triplet/"
    model_path= Path(config.data_dir) / 'model-best.pkl'
    assert not exist_embeddings(config), f"The embeddings(k = {config.k_clusters}) already exist"
    device = 'cuda'
    n_max_videos_per_class = 4
    
    video_path = "data/videos"

    x_labels, y_labels, = normalize_florence_to_ntu_distribution(x_labels, y_labels)
    
    db: Dict[int, List] = {}
    predictor = Predictor(model_path=model_path, device=device, model_name='stgcn-recons')
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
            x_label = keypoints_to_tensor(keypoints_by_id[id])
            embedding = predictor.encode(x_label.to(device))
            db[action_idx].append(embedding)
    

    print("Extract keypoints from videos")
    for x_label, y_label in zip(x_labels, y_labels): 
        action_idx = int(y_label)
        if action_idx not in db:
            db[action_idx] = []
        if len(db[action_idx]) == n_max_videos_per_class:
            continue
        
        x_label = torch.from_numpy(x_label)


    for key, items in db.items():
        print(f"[{key}] number of videos: {len(items)}")

    #breakpoint()
    save_embeddings(db, config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="experiments_data/data_stgcn/", required=False, help="path to dataset dir")
    parser.add_argument('--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('--k_clusters', type=int, default=None, help="number of cluster to use for KMeans")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    parser.add_argument('--update', action='store_true', help="Update database using custom skeleton")
    # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=16,
                        help='stride determining when to start next window of frames')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")

    parser.add_argument('--similarity_measurement_window_size', type=int, default=1,
                        help='measuring similarity over # of oversampled video sequences')
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--privacy_on', action='store_true',
                        help='when on, no original video or sound in present in the output video')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to seprate positive and negative classes')
    parser.add_argument('--connected_joints', action='store_true', help='connect joints with lines in the output video')

    args = parser.parse_args()
    config = Config(args)
    main(config)