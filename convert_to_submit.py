import os
import glob
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

def main():
    # mkdir
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    name_to_paths = defaultdict(list)
    for src_path in glob.glob(os.path.join(args.image_dir, "*")):
        name = os.path.basename(src_path).replace(".jpg", "").rsplit("_", 1)[0]
        name_to_paths[name].append(src_path)

    # resize & save
    for name, paths in name_to_paths.items():
        image = Image.open(paths[0]).resize((428, 240)) 
        image.save(os.path.join(args.save_dir, name + ".jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    main()