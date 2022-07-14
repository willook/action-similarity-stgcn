import argparse
import numpy as np
import random

from stgcn.predictor import Predictor as STGCNPredictor
from action_similarity_stgcn.utils import cache_file, Timer, save_file
from action_similarity_stgcn.database import ActionDatabase
from action_similarity_stgcn.motion import extract_keypoints
from action_similarity_stgcn.simple_predictor import Predictor as KNNPredicor
from action_similarity_stgcn.simple_database import ActionDatabase

random.seed(1234)

def main():
    #video_path = './data/samples/hand_signal01.mp4'
    #video_path = './data/samples/jump01.mp4'
    # video_path = './data/testset/001/S002C002P004R001A001.mp4'
    # video_path = './data/testset/002/S002C003P003R001A002.mp4'
    # video_path = './data/testset/003/S002C002P004R001A003.mp4'
    # video_path = './data/testset/004/S002C002P004R001A004.mp4'
    # video_path = './data/testset/005/S002C003P003R001A005.mp4'
    # video_path = './data/testset/006/S002C003P002R001A006.mp4'
    # video_path = './data/testset/007/S002C002P004R001A007.mp4'
    video_path = './data/testset/008/S002C001P005R001A008.mp4'
    # video_path = './data/testset/009/S002C001P005R001A009.mp4'
    # video_path = './data/testset/010/S002C001P005R001A010.mp4'
    #video_path = './data0419/samples/stop01.mp4'
    
    timer = Timer()
    timer.log("DB")
    print("Compute standard db...")
    db = ActionDatabase(
        config=config,
        action_label_path='./data/action_label.txt',
    )


    timer.log("Kepoint")    
    print("Extract keypoints...")
    #keypoints_by_id = extract_keypoints(video_path, fps=30)
    keypoints_by_id = cache_file(video_path, extract_keypoints, 
         *(video_path,), **{'fps':30,})

    print("Predict action...")
    timer.log("predict") 
    predictor = Predictor(
        config=config, 
        model_path='./data/model_best.pth.tar',
        std_db=db)
    predictions = predictor.predict(keypoints_by_id)
    action_label_per_id, similarities_per_id = predictor.info()

    # print results
    for id in action_label_per_id:
        print("[id] result:")
        action_label = action_label_per_id[id]
        similarities_per_actions = similarities_per_id[id]
        for action, similarities in similarities_per_actions.items():
            print(f"mean similarity of {predictor.std_db.actions[action]}: {np.mean(similarities)}")
        timer.log() 
        print(f"Predicted action is {db.actions[action_label]}")
        print(f"Predictions:\n{predictions}")
        print()
    timer.pprint()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    #parser.add_argument('--clustering', type=str, default=None, help="clustering for standard database")
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
    main()