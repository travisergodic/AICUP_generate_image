import os
from collections import defaultdict

def check_num_sample_correct(gen_dir, num_smaple):
    name_to_paths = defaultdict(list)
    for file in os.listdir(gen_dir):
        if not file.endswith(".jpg"):
            continue
        
        k = file.replace(".jpg", "").rsplit("_", 1)[0]
        name_to_paths[k].append(file)

    for k, v in name_to_paths.items():
        if len(v) != num_smaple:
            raise ValueError(f"{k} got {len(v)} samples!")