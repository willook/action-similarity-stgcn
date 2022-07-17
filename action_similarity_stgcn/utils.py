import os
import re
import pickle
import time
import random
from typing import List, Dict
from pathlib import Path

from glob import glob

def sorted_sample(x: List, size: int):
    assert len(x) >= size, f"Input list should be larger than sample size({size}), but {len(x)}"
    sorted_list = [
        x[i] for i in sorted(random.sample(range(len(x)), size))]
    return sorted_list

def parse_action_label(action_label_path):
    actions = {}
    with open(action_label_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            no, action = line.split(None, maxsplit=1)
            no = int(re.search(r'\d+', no).group())
            actions[no] = action.strip()
    return actions

def cache_file(file_name: str, func, *args, **kwargs):
    base_name = Path(".cache") / (os.path.splitext(file_name)[0] + ".pickle")
    
    # 디렉토리 생성
    head, _ = os.path.split(base_name)
    Path(head).mkdir(parents=True, exist_ok=True)
    
    # cache pickle 생성 또는 불러오기
    if os.path.exists(base_name):
        with open(base_name, "rb") as f:
            data = pickle.load(f)
    else:
        data = func(*args, **kwargs)
        with open(base_name, "wb") as f:
            pickle.dump(data, f)
    return data

def save_file(file_name: str, func, *args, **kwargs):
    file_name = os.path.splitext(file_name)[0] + ".pickle"
    data = func(*args, **kwargs)
    with open(file_name, "wb") as f:
        pickle.dump(data, f)
    return data

def save_embeddings(db: Dict, embeddings_dir = "data/embeddings", key: str = 'default'):
    embeddings_dir = Path(embeddings_dir)
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    for action_idx, embeddings in db.items():
        embeddings_filename = embeddings_dir / f'action_embeddings_{action_idx:03d}.pickle'
        # pickle 파일이 이미 있는 경우
        if os.path.exists(embeddings_filename):
            with open(embeddings_filename, 'rb') as f:
                embeddings_dict = pickle.load(f) # dictionary
                embeddings_dict[key] = embeddings
        # pickle 파일이 없는 경우 dict로 생성
        else:
            embeddings_dict = {key: embeddings}
            
        with open(embeddings_filename, 'wb') as f:
            pickle.dump(embeddings_dict, f)

    with open(embeddings_dir / "readme.md", 'a') as f:
        f.write(f"saved embeddings with key = {key}\n")
            
def load_embeddings(embeddings_dir = "data/embeddings", key: str = 'default', target_actions=None): 
    embeddings_dir = Path(embeddings_dir)
    std_db = {}
    for embedding_file in glob(f'{embeddings_dir}/*'):
        if not embedding_file.endswith(".pickle"):
            continue
        with open(embedding_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
            file_name = os.path.basename(embedding_file).rstrip(".pickle")
            action_idx = int(file_name.split("_")[-1])        
            if target_actions is None or action_idx in target_actions:
                std_db[action_idx] = embeddings_dict[key]
    return std_db 

def exist_embeddings(embeddings_dir = "data/embeddings", key: str = 'default'):
    embeddings_dir = Path(embeddings_dir)
    exist_flags = []
    for embedding_file in glob(f'{embeddings_dir}/*'):
        if not embedding_file.endswith(".pickle"):
            continue
        with open(embedding_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
            exist_flags.append(key in embeddings_dict)
    return len(exist_flags) !=0 and all(exist_flags)

def take_best_id(keypoints_by_id: Dict[str, List[Dict]]):
    max_id = -1
    max_len = -1
    for id, annotations in keypoints_by_id.items():
        if len(annotations) > max_len:
            max_len = len(annotations)
            max_id = id
    return max_id


class Timer():
    def __init__(self):
        self.times: List = []
        #self.end_times: List = []
        self.tags: List = []
        self.index = 0

    def log(self, tag = None):
        self.times.append(time.time())
        if tag is None:
            self.tags.append(self.index)
            self.index += 1
        else:
            self.tags.append(tag)

    def pprint(self):
        for tag, start, end in zip(self.tags[:-1], self.times[:-1], self.times[1:]):
            print(f"[{tag}] time elasp: {(end-start):.3f}")
        print(f"[Total] time elasp: {(self.times[-1] - self.times[0]):.3f}")
    
    def info(self):
        times = []
        for tag, start, end in zip(self.tags[:-1], self.times[:-1], self.times[1:]):
            times.append((tag, end-start))
        return times