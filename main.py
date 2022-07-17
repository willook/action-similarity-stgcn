import os

import argparse
import numpy as np
import random
from pathlib import Path

from stgcn.predictor import Predictor as STGCNPredictor
from action_similarity_stgcn.predictor import Predictor as KNNPredicor
from action_similarity_stgcn.database import ActionDatabase
from action_similarity_stgcn.utils import cache_file, Timer
from action_similarity_stgcn.database import ActionDatabase
from action_similarity_stgcn.motion import extract_keypoints, preprocess_keypoints_by_id

random.seed(1234)

def main(args):
    #video_path = './data/samples/hand_signal01.mp4'
    #video_path = './data/samples/jump01.mp4'
    # video_path = './data/testset/001/S002C002P004R001A001.mp4'
    # video_path = './data/testset/002/S002C003P003R001A002.mp4'
    # video_path = './data/testset/003/S002C002P004R001A003.mp4'
    # video_path = './data/testset/004/S002C002P004R001A004.mp4'
    # video_path = './data/testset/005/S002C003P003R001A005.mp4'
    video_path = './data/testset/006/S002C003P002R001A006.mp4'
    # video_path = './data/testset/007/S002C002P004R001A007.mp4'
    # video_path = './data/testset/008/S002C001P005R001A008.mp4'
    # video_path = './data/testset/009/S002C001P005R001A009.mp4'
    # video_path = './data/testset/010/S002C001P005R001A010.mp4'
    #video_path = './data0419/samples/stop01.mp4'
    
    device = 'cuda'
    data_path = Path(args.data_dir)
    timer = Timer()
    timer.log("DB")
    print("load standard action db...")
    db = ActionDatabase(
        database_path= data_path / 'embeddings',
        label_path = data_path / 'action_label.txt',
    )

    timer.log("Kepoint")    
    print("Extract keypoints...")
    #keypoints_by_id = extract_keypoints(video_path, fps=30)
    basename, ext = os.path.splitext(video_path)
    pickle_name = basename + f"_{args.fps}" + ext
    keypoints_by_id = cache_file(pickle_name, extract_keypoints, 
         *(video_path,), **{'fps':args.fps,})

    print("Predict action...")
    timer.log("predict") 
    stgcn_predictor = STGCNPredictor(
        model_path=data_path / 'models/model-best.pkl',
        device=device,
        model_name='stgcn-recons')
    knn_predictor = KNNPredicor(std_db=db)

    processed_keypoints_by_id = preprocess_keypoints_by_id(keypoints_by_id, device)
    embedding_by_id = stgcn_predictor.encode(processed_keypoints_by_id)
    predictions = knn_predictor.predict(embedding_by_id)
    action_label_per_id, similarities_per_id = knn_predictor.info()

    # print results
    for id in action_label_per_id:
        print("[id] result:")
        action_label = action_label_per_id[id]
        similarities_per_actions = similarities_per_id[id]
        for action, similarities in similarities_per_actions.items():
            print(f"[{action}] mean similarity of {knn_predictor.std_db.actions[action]}: {np.mean(similarities)}")
        timer.log() 
        print(f"Predicted action is {db.actions[action_label]}")
        print(f"Predictions:\n{predictions}")
        print()
    timer.pprint()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--fps', type=int, default=30, help="fps to embed video")
    parser.add_argument('--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)