import os
import argparse
import shutil
from pathlib import Path

import torch
import numpy as np

import fid.model
from fid.utils import check_num_sample_correct
from fid.algorithm import greedy_optimize
from fid.dataset import build_fid_loader
from fid.eval import make_image_features
from fid.logger_helper import setup_logger
from fid.fid_score import calculate_fid_score

logger = setup_logger() 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(fid.model, f"build_{args.model}")()
    model = model.to(device)
    logger.info("Load model successfully")

    check_num_sample_correct(args.gen_dir, args.num_sample)
    # real
    real_river_loader = build_fid_loader(image_dir=args.real_dir, type="river", batch_size=args.batch_size)
    real_road_loader = build_fid_loader(image_dir=args.real_dir, type="road", batch_size=args.batch_size)

    # gen
    gen_river_loader = build_fid_loader(image_dir=args.gen_dir, type="river", batch_size=args.batch_size)
    gen_road_loader = build_fid_loader(image_dir=args.gen_dir, type="road", batch_size=args.batch_size)

    # make features
    logger.info("Generating real river features ...")
    real_river_feat, _ = make_image_features(real_river_loader, model, device=device)
    logger.info("Generating real road features ...")
    real_road_feat, _ = make_image_features(real_road_loader, model, device=device)
    logger.info("Generating generated river features ...")
    gen_river_feat, gen_river_name = make_image_features(gen_river_loader, model, device=device)
    logger.info("Generating generated road features ...")
    gen_road_feat, gen_road_name = make_image_features(gen_road_loader, model, device=device)
    print(real_river_feat.shape)
    print(real_road_feat.shape)
    
    _, feat_dim = gen_river_feat.shape
    gen_river_feat = gen_river_feat.reshape(-1, int(args.num_sample / args.freq), feat_dim) 
    gen_road_feat = gen_road_feat.reshape(-1, int(args.num_sample / args.freq), feat_dim)
    print(gen_river_feat.shape)
    print(gen_road_feat.shape)

    gen_river_name = np.array(gen_river_name).reshape(-1, int(args.num_sample / args.freq))
    gen_road_name = np.array(gen_road_name).reshape(-1, int(args.num_sample / args.freq))
    np.savetxt("./river.txt", gen_river_name, fmt='%s')
    np.savetxt("./road.txt", gen_road_name, fmt='%s')

    river_select_idxs, road_select_idxs = None, None
    for i in range(args.round):
        logger.info(f"Round {i}: Selecting the best river images ...")
        river_select_idxs, river_score = greedy_optimize(
            real_river_feat, 
            gen_river_feat, 
            dist=calculate_fid_score, 
            init_idxs=river_select_idxs
        )

        logger.info(f"Round {i}: Selecting the best road images ...")
        road_select_idxs, road_score = greedy_optimize(
            real_road_feat, 
            gen_road_feat, 
            dist=calculate_fid_score, 
            init_idxs=road_select_idxs
        )

        average_score = (river_score + road_score)/2

        best_river_names = gen_river_name[range(river_select_idxs.size), river_select_idxs]
        best_road_names = gen_road_name[range(road_select_idxs.size), road_select_idxs]

        save_dir = os.path.join(args.save_dir, f"round{i}_{average_score:.4f}")
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        for file in [*best_river_names.tolist(), *best_road_names.tolist()]:
            shutil.copy(os.path.join(args.gen_dir, file), save_dir)

        logger.info(f"Save round {i} selected images in {save_dir}. FID score {average_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, choices=["vgg16", "clip_vit_b32", "clip_vit_b16", "dinov2_vit_b14", "resnet34"])
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_sample", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--freq", type=int, default=1)
    args = parser.parse_args()
    main()