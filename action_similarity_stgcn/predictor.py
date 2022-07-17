from __future__ import annotations 

import os
from typing import List, Dict, Tuple, TYPE_CHECKING
import numpy as np
import torch
from scipy.spatial.distance import cosine

if TYPE_CHECKING:
    from action_similarity_stgcn.database import ActionDatabase
    

class Predictor:
    def __init__(
        self, 
        std_db: ActionDatabase,
        k_neighbors: int = 1,
        min_frames: int = 32
    ):
        self.std_db = std_db
        self.min_frames = min_frames
        self.k_neighbors = k_neighbors

    def valid_frames(
        self,
        annotations: List[Dict]):
        return len(annotations) >= self.min_frames
    
    def predict(
        self,
        embedding_by_id: Dict[str, np.ndarray]):
        # keypoints_by_id 거르는 함수, 양식 맞추는 함수 필요
        predictions = []
        action_label_per_id = {}
        similarities_per_id = {}

        # keypoints가 충분한 id만 예측을 수행
        valid_ids = []
        for id, annotations in embedding_by_id.items():
            if self.valid_frames(annotations):
                valid_ids.append(id)
            
        for id in valid_ids:
            embedding = embedding_by_id[id]
            action_label, score, similarities_per_actions = self._predict(embedding)
            prediction = self.make_prediction(id, annotations, action_label, score)
            predictions.append(prediction)
            action_label_per_id[id] = action_label
            similarities_per_id[id] = similarities_per_actions
        self._action_label_per_id = action_label_per_id
        self._similarities_per_id = similarities_per_id
        return predictions

    def _predict(
        self, 
        motion_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Params
            motion_embedding: #windows, #body_part, #features
        return action label, similarities
        Predict action based on similarities.
        The action that has the least similarity between reference motion embedding and std_db is determined.  
        """

        similarities_per_actions = self.compute_action_similarities(motion_embedding)
        actions_similarities_pair = [[], []] # actions, similarities
        for action, similarities in similarities_per_actions.items():
            n = len(similarities)
            actions_similarities_pair[0].extend([action] * n) # actions
            actions_similarities_pair[1].extend(similarities) # similarities
        actions = actions_similarities_pair[0]
        similarities = actions_similarities_pair[1]
        sorted_actions_by_similarity = [(action, similarity) for similarity, action in sorted(zip(similarities, actions), reverse=True)]
        for pair in sorted_actions_by_similarity:
            action_label, similarity = pair

        # Select action by the closest k neighbors
        k = self.k_neighbors
        bin_dict = {}
        score_dict = {}
        for i in range(k):
            candidate_label, similarity = sorted_actions_by_similarity[i]
            if candidate_label not in bin_dict:
                bin_dict[candidate_label] = 1
                score_dict[candidate_label] = [similarity]
            else:
                bin_dict[candidate_label] += 1
                score_dict[candidate_label].append(similarity)
        action_label = max(bin_dict, key = bin_dict.get)
        score = np.mean(score_dict[action_label])
        return action_label, score, similarities_per_actions

    def compute_action_similarities(
        self, 
        embedding_anchor: np.ndarray) -> Dict[str, List[float]]:
        """
        param anchor: reference motion embedding to recognize, #windows x 5 x #features
        param std_db: standard action database, value: #videos x #windows x 5 x #features
        return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embedding and each motion embedding of std_db.
        Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
        """
        # use accelerated_dtw
        similarities_per_actions: Dict[str, List[float]] = {} 
        
        for action_label, embedding_list in self.std_db.db.items():
            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for embedding in embedding_list:
                #motion_embedding = seq_feature_to_motion_embedding(seq_features)
                similarity = 1-cosine(embedding_anchor, embedding)
                similarities_per_actions[action_label].append(similarity)
        return similarities_per_actions

    def make_prediction(
        self,
        id: int,
        annotations: List[Dict],
        action_label: str,
        score: float):

        # 예측에 사용한 첫번째 프레임의 정보
        first_annotation = annotations[0]
        prediction = {}
        prediction['id'] = id
        if score >= self.threshold:
            actions = [{'label': action_label, 'score': score}]
        else:
            actions = []

        prediction['predictions'] = [{
            'frame': first_annotation['frame'],
            'box': first_annotation['keypoints']['box'],
            'score': first_annotation['keypoints']['score'],
            'actions': actions
        }]
        
        return prediction
    
    def info(self):
        return self._action_label_per_id, self._similarities_per_id