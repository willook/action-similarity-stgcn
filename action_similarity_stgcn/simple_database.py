from typing import Dict, List

import os
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

from bpe import Config

from action_similarity_stgcn.utils import exist_embeddings, parse_action_label, load_embeddings, save_embeddings, cache_file, take_best_id
from action_similarity_stgcn.motion import extract_keypoints

class ActionDatabase():

    def __init__(
        self,
        action_label_path: str = None,
        key = 'default',
        embeddings_dir = 'data/embeddings'
    ):
        self.db = {}
        self.action_label_path = action_label_path
        self.actions = parse_action_label(self.action_label_path)
        print(f"[db] Load motion embedding...")
        assert exist_embeddings(key=key, embeddings_dir=embeddings_dir), f"The embeddings(key = {key}) not exist. "\
            f"You should run the main with --update or bin.postprocess with --k_clusters option"
        self.db = load_embeddings(key=key, embeddings_dir=embeddings_dir)