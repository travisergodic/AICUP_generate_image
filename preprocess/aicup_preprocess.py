import os
import glob
import shutil
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    save_dir = os.path.abspath(args.save_dir)
    save_river_dir = os.path.join(save_dir, "river")
    save_road_dir = os.path.join(save_dir, "road")

    Path(save_river_dir).mkdir(parents=True, exist_ok=True)
    Path(save_road_dir).mkdir(parents=True, exist_ok=True)

    river_image_paths = sorted(glob.glob(os.path.join(args.root, "img", "*RI*.jpg")))
    river_sketch_paths = sorted(glob.glob((os.path.join(args.root, "label_img", "*RI*.png"))))
    print(f"Found {len(river_image_paths)} river images, {len(river_sketch_paths)} river sketches")

    road_image_paths = sorted(glob.glob(os.path.join(args.root, "img", "*RO*.jpg")))
    road_sketch_paths = sorted(glob.glob((os.path.join(args.root, "label_img", "*RO*.png"))))
    print(f"Found {len(road_image_paths)} river images, {len(road_sketch_paths)} river sketches")

    print("Do train test split ...")
    train_river_image_paths, val_river_image_paths = train_test_split(
        river_image_paths, test_size=args.test_size, random_state=42, shuffle=True
    )
    train_road_image_paths, val_road_image_paths = train_test_split(
        road_image_paths, test_size=args.test_size, random_state=42, shuffle=True
    )

    print(f"Copy river images & sketches to {save_river_dir}")
    for src_path in (river_image_paths + river_sketch_paths):
        shutil.copy(src_path, save_river_dir)

    print(f"Copy road images & sketches to {save_road_dir}")
    for src_path in (road_image_paths + road_sketch_paths):
        shutil.copy(src_path, save_road_dir)

    pd.Series(train_river_image_paths).apply(
        lambda x: os.path.join(save_river_dir, os.path.basename(x))
    ).to_csv(os.path.join(save_dir, "COCOSTUFF_train_river.txt"), index=False, header=None)

    pd.Series(train_road_image_paths).apply(
        lambda x: os.path.join(save_road_dir, os.path.basename(x))
    ).to_csv(os.path.join(save_dir, "COCOSTUFF_train_road.txt"), index=False, header=None)

    pd.Series(val_river_image_paths).apply(
        lambda x: os.path.join(save_river_dir, os.path.basename(x))
    ).to_csv(os.path.join(save_dir, "COCOSTUFF_val_river.txt"), index=False, header=None)

    pd.Series(val_road_image_paths).apply(
        lambda x: os.path.join(save_road_dir, os.path.basename(x))
    ).to_csv(os.path.join(save_dir, "COCOSTUFF_val_road.txt"), index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    args = parser.parse_args()
    main()