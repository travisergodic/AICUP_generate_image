"""
Train a diffusion model on images.
"""
# import gradio as gr
import os
import glob
import argparse
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch as th
from einops import rearrange

from pretrained_diffusion import dist_util, logger
from torchvision.utils import make_grid
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from pretrained_diffusion.image_datasets_sketch import get_tensor
from pretrained_diffusion.train_util import TrainLoop
from pretrained_diffusion.glide_util import sample 


def merge_encoder_decoder_ckpt(encoder_ckpt, decoder_ckpt):
    encoder_state_dict = th.load(encoder_ckpt)
    decoder_state_dict = th.load(decoder_ckpt)
    return {
        **{f"encoder.{k}": v for k, v in encoder_state_dict.items()},
        **{f"decoder.{k}": v for k, v in decoder_state_dict.items()},
    }


def run(mode):
    parser, parser_up = create_argparser()
    
    args = parser.parse_args()
    args_up = parser_up.parse_args()
    dist_util.setup_dist()

    if mode == 'sketch':
        args.mode = 'coco-edge'
        args_up.mode = 'coco-edge'
        # args.model_path = '/content/test.pt' # '/content/PITI/ckpt/base_edge.pt'
        # args.sr_model_path = './ckpt/upsample_edge.pt'

    elif mode == 'mask':
        args.mode = 'coco'
        args_up.mode = 'coco'
        args.model_path = './ckpt/base_mask.pt'
        args.sr_model_path = './ckpt/upsample_mask.pt'

    sample_step = args.sample_step
    sample_c = args.sample_c
    up_c = args.up_c

    print(f"sample step: {sample_step}")
    print(f"sample_c: {sample_c}")
    print(f"up_c: {up_c}")
  

    options=args_to_dict(args, model_and_diffusion_defaults(0.).keys())
    model, diffusion = create_model_and_diffusion(**options)
 
    options_up=args_to_dict(args_up, model_and_diffusion_defaults(True).keys())
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
 

    if  args.model_path:
        print('loading model')
        # model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")
        model_ckpt = merge_encoder_decoder_ckpt(args.encoder_ckpt, args.decoder_ckpt)
        model.load_state_dict(model_ckpt, strict=True)

    if  args.sr_model_path:
        print('loading sr model')
        model_ckpt2 = dist_util.load_state_dict(args.sr_model_path, map_location="cpu")
        model_up.load_state_dict(model_ckpt2, strict=True) 

    model.to(dist_util.dev())
    model_up.to(dist_util.dev())
    model.eval()
    model_up.eval()
 
########### dataset
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    if '*' in args.image:
        images = sorted(glob.glob(args.image))
    else:
        images = [args.image]
 
    for image in tqdm(images):
        pil_image = Image.open(image) 
        label_pil = pil_image.convert('L').resize((256, 256), Image.NEAREST)
        
        im_dist = cv2.distanceTransform(255-np.array(label_pil), cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist) , 0, 255).astype(np.uint8)
        im_dist = Image.fromarray(im_dist).convert('RGB')

        label_tensor =  get_tensor()(im_dist)[:1]
    
        data_dict = {'ref':label_tensor.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)}
    
        print('sampling...')

        sampled_imgs = []
        grid_imgs = []
        img_id = 0
        while (True):
            if img_id >= args.num_samples:
                break
    
            model_kwargs = data_dict
            with th.no_grad():
                samples_lr =sample(
                    glide_model= model,
                    glide_options= options,
                    side_x= 64,
                    side_y= 64,
                    prompt=model_kwargs,
                    batch_size= args.num_samples,
                    guidance_scale=sample_c,
                    device=dist_util.dev(),
                    prediction_respacing= str(sample_step),
                    upsample_enabled= False,
                    upsample_temp=0.997,
                    mode = args.mode,
                )

                samples_lr = samples_lr.clamp(-1, 1)

                tmp = (127.5*(samples_lr + 1.0)).int() 
                model_kwargs['low_res'] = tmp / 127.5 - 1.

                samples_hr =sample(
                    glide_model= model_up,
                    glide_options= options_up,
                    side_x=256,
                    side_y=256,
                    prompt=model_kwargs,
                    batch_size=args.num_samples,
                    guidance_scale=up_c,
                    device=dist_util.dev(),
                    prediction_respacing= "fast27",
                    upsample_enabled=True,
                    upsample_temp=0.997,
                    mode = args.mode,
                )   

                for hr in samples_hr:
                    hr = 255. * rearrange((hr.cpu().numpy()+1.0)*0.5, 'c h w -> h w c')
                    sample_img = Image.fromarray(hr.astype(np.uint8))
                    save_path = os.path.join(args.save_dir, os.path.basename(image).split('.')[0] + f"_{img_id}.jpg")
                    sample_img.save(save_path)
                    # sampled_imgs.append(sample_img)
                    img_id += 1   
        #         grid_imgs.append(samples_hr)
        
        # # high resolution
        # grid = th.stack(grid_imgs, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=2)
        # grid = 255. * rearrange((grid+1.0)*0.5, 'c h w -> h w c').cpu().numpy()
        # save_path = os.path.join(args.save_dir, os.path.basename(image))
        # Image.fromarray(grid.astype(np.uint8)).save(save_path)
        # print()

        # low resolution
        # grid_low_res = make_grid(tmp, nrow=2)
        # save_path = os.path.join(args.save_dir, os.path.basename(image).split(".")[0] + "_low_res.jpg")
        # Image.fromarray(grid_low_res.astype(np.uint8)).save(save_path)
 
 
def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="./base_edge.pt",
        sr_model_path="./upsample_edge.pt",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_c=1.,
        up_c = 1,
        sample_step=100,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=3,
        finetune_decoder = False,
        mode = '',
        image = "path/to/image",
        encoder_ckpt = "path/to_encoder/ckpt",
        decoder_ckpt = "path/to/decoder/ckpt",
        save_dir = "/path/to/folder",
        sr_model = "/path/to/sr_model"
    )

    defaults_up = defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    print(parser)

    defaults_up.update(model_and_diffusion_defaults(True))
    parser_up = argparse.ArgumentParser()
    add_dict_to_argparser(parser_up, defaults_up)

    return parser, parser_up

run(mode="sketch")