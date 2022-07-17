import os
import argparse
from typing import List
from pathlib import Path
from pprint import pprint
from glob import glob
import random

from stgcn.predictor import Predictor as STGCNPredictor
from action_similarity_stgcn.predictor import Predictor as KNNPredicor
from action_similarity_stgcn.utils import cache_file, Timer
from action_similarity_stgcn.database import ActionDatabase
from action_similarity_stgcn.motion import extract_keypoints, preprocess_keypoints_by_id

""" 
Current accuracy_test hyperparameters:

    Parameters:
        fps (int): framerate of training videos. (current = 10)
        k_neighbors (int): number of neighbors to use for KNN (current = 5)
"""
def main(args):
    random.seed(1234)
    n_embeddings_per_action = 12
    device = 'cuda'
    target_actions = [8, 9, 10, 11, 12, 13]
    data_path = Path(args.data_dir)
    video_path = data_path / "testset"
    info = {action_idx: [0, 0] for action_idx in target_actions}
    
    timer = Timer()
    timer.log("DB")
    print("Compute standard db...")
    db = ActionDatabase(
        database_path= data_path / 'embeddings',
        label_path = data_path / 'action_label.txt',
        target_actions=target_actions,
    )
    for action_idx in db.db.keys():
        if len(db.db[action_idx]) < n_embeddings_per_action:
            print(f"[warning] A minimum of {n_embeddings_per_action} videos is required but, {len(db.db[action_idx])} ")
            continue
        db.db[action_idx] = random.sample(db.db[action_idx], n_embeddings_per_action)
        #features = random.sample(features, 10)
        print(db.actions[action_idx], len(db.db[action_idx]))
        
    stgcn_predictor = STGCNPredictor(
        model_path=data_path / 'models/model-best.pkl',
        device=device,
        model_name='stgcn-recons')
    knn_predictor = KNNPredicor(std_db=db)

    for video_dir in glob(f'{video_path}/*'):
        action_str = os.path.basename(os.path.normpath(video_dir))
        if not action_str.isdigit(): # 숫자가 아닌 디렉토리인 경우 넘어감
            continue
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        if action_idx not in target_actions:
            continue
        
        print(f"Current action idx: {action_idx}")
        for video_filepath in glob(f'{video_dir}/*'):
            if os.path.splitext(video_filepath)[1] not in ['.mp4', '.avi', '.mkv']:
                continue
            info[action_idx][1] += 1 # 전체 비디오 갯수

            fps = args.fps
            if fps == 30:
                pickle_name = video_filepath
            else:
                basename, ext = os.path.splitext(video_filepath)
                pickle_name = basename + f"_{fps}" + ext
                
            keypoints_by_id = cache_file(pickle_name, extract_keypoints, 
                *(video_filepath,), **{'fps': fps,})
            for id in keypoints_by_id:
                print(video_filepath, len(keypoints_by_id[id]))
            
            try:
                processed_keypoints_by_id = preprocess_keypoints_by_id(keypoints_by_id, device)
                embedding_by_id = stgcn_predictor.encode(processed_keypoints_by_id)
                predictions = knn_predictor.predict(embedding_by_id)
            except Exception as e:
                print(e)
                continue            
            if not predictions: # 빈리스트
                print(f"[Warning] Number of frames lacks, {len(keypoints_by_id[id])}")
                info[action_idx][1] -= 1 # 전체 비디오 갯수
                continue
            if len(predictions[0]['predictions'][0]['actions']) == 0:
                print("[Warning] There is no predicted actions")
                pprint(predictions)
                continue
            predict = predictions[0]['predictions'][0]['actions'][0]['label']
            #print(predict, action_idx)
            if predict == action_idx:
                info[action_idx][0] += 1 # 맞은 갯수
    
    total_n = 0
    total_k = 0
    for id in info:
        k = info[id][0]
        n = info[id][1]
        total_k += k
        total_n += n
        if n == 0:
            print(f"[{id}] {k}/{n}")
        else:
            print(f"[{id}] {k}/{n}, {k/n}")
    print(f"[total] {total_k}/{total_n}, {total_k/total_n}")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="data", help="path to dataset dir")
    parser.add_argument('--k_neighbors', type=int, default=5, help="number of neighbors to use for KNN")
    parser.add_argument('--frames', type=int, default=0, help="number of frames to predict")
    parser.add_argument('--fps', type=int, default=30, required=False, help="fps to embed video")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    
    args = parser.parse_args()
    main(args)