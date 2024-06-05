# AICUP 建構無人機於自然環境偵察時所需之導航資訊競賽 I－ 影像資料生成競賽

## 介紹
比賽目標通過輸入道路與河川的草圖 (sketch) 生成對應的真實影像。最終，通過生成影像與真實影像間的 FID score 作為評分標準。本專案是基於 [**PITI (Pretraining is All You Need for Image-to-Image Translation**)](https://github.com/PITI-Synthesis/PITI) 的原始碼進行修改的。
1. **隊伍**: TEAM_5411
2. **隊員**: 胡太維 (隊長)、徐黎安、王柏靇
3. **最終排名**: 2 (共 130 隊)
4. **Private leaderboard**: 84.26206

![alt text](figure/PITI_model.png)

|![alt text](figure/TRA_RI_1001440.png)|![alt text](figure/TRA_RI_1001440.jpg)|
|------------------------|------------------------|
|![alt text](figure/TRA_RO_1004319.png)|![alt text](figure/TRA_RO_1004319.jpg)|
    

## 軟硬體設定
1. 操作系統: Linux-64
2. 硬體: 
   + CPU: 12 核心 Intel(R) Xeon(R) CPU @ 2.20GHz
   + GPU: NVIDIA L4 (23 GB)
3. Python 版本: 3.7.11

## 環境安裝
```
conda env create -f environment.yml
``` 

## 前置作業
1. **下載比賽資料，並放置於 `dataset/` 路徑下**
   + `dataset/Training dataset/`
   + `dataset/public_label_img/`
   + `dataset/private_label_img/`
   
   <p align = "left">     
   <img  src="figure/diretory_structure.png" width="200" />
   </p>


2. **前處理**
   ```bash
   python preprocess/aicup_preprocess.py --root dataset/Trainig_dataset \
                                         --save_dir dataset \
                                         --test_size 0.1                    
   ```


## 模型訓練
分為河道 (river) 生成、道路 (road) 生成兩個任務，每個任務都需要訓練兩個模型，分別為 **sketch-to-image** 模型，該模型會根據草圖 (sketch) 生成對應的影像，大小為 64x64，另一個模型為 **super-resolution** 模型，會將 64x64 大小的影像轉換為 256x256 的影像。訓練 **sketch-to-image** 模型時，需要先對 encoder 進行 2 次微調 (stage1, stage2)，接著，再對 encoder 與 decoder 同時進行微調 (stage3)。訓練 super-resolution 模型時，會對 encoder 與 decoder 同時進行微調。\
訓練前，請在專案路徑下創建 `pretrained/` 資料夾，並至[以下連結](https://drive.google.com/drive/u/0/folders/1CVtu32Clq63j-17CvyBatKQtFLxW0VRk)下載所有預訓練權重，並放置於 `pretrained/` 資料夾。

1. **訓練 sketch-to-image 模型**
   + **river 模型**
     ```
     bash sketch_finetune_base.sh river
     ```
   + **road 模型**
     ```
     bash sketch_finetune_base.sh road
     ```
2. **訓練 upsample 模型**
   + **river 模型**
     ```
     bash sketch_finetune_upsample.sh river
     ```
   + **road 模型**
     ```
     bash sketch_finetune_upsample.sh road
     ```


## 模型推理
|![alt text](figure/PRI_RI_1000018.png)|![alt text](figure/PRI_RI_1000018_21.jpg)|
|------------------------|------------------------|
|![alt text](figure/PRI_RO_1000494.png)|![alt text](figure/PRI_RO_1000494_22.jpg)|

1. **下載模型權重**

   |model url|
   |------------------------|
   |[sketch-to-image encoder (river)](https://drive.google.com/file/d/145eFhQXcdRFacfiiyv7TGPuXRE85_vhJ/view?usp=sharing)|
   |[sketch-to-image decoder (river)](https://drive.google.com/file/d/1DwgKmKa3Q4-T4gXg9t0dvWECivKxg5KJ/view?usp=sharing)|
   |[Upsample model (river)](https://drive.google.com/file/d/18P6QFHWBvLq__BTLyGQLTiernE55wFB_/view?usp=sharing)|
   |[sketch-to-image encoder (road)](https://drive.google.com/file/d/17pRHlnnMI1AITWrYKzKmgoD63o7q21pW/view?usp=sharing) |
   |[sketch-to-image decoder (road)](https://drive.google.com/file/d/1DwgKmKa3Q4-T4gXg9t0dvWECivKxg5KJ/view?usp=sharing) |
   |[Upsample model (road)](https://drive.google.com/drive/u/0/folders/1-44fY-BXLztUHgFTrHwNf-itNPspBJYZ)        |

2. **生成河流影像**
    ```shell
    python inference.py --encoder_ckpt path/to/river/sketch/to/image/encoder \
                        --decoder_ckpt path/to/river/sketch/to/image/decoder \
                        --sr_model_path path/to/river/upsample/model \
                        --image dataset/Private_testing_dataset/label_img/*RO*.png \
                        --save_dir ./generate_images
                        --num_samples 8
    ```

3. **生成道路影像**
    ```shell
    python inference.py --encoder_ckpt path/to/road/sketch/to/image/encoder \
                        --decoder_ckpt path/to/road/sketch/to/image/decoder \
                        --sr_model_path path/to/road/upsample/model \
                        --image dataset/Private_testing_dataset/label_img/*RI*.png \
                        --save_dir ./generate_images
                        --num_samples 8
    ```

4. **篩選合適的影像**
    ```shell
    python optimize_fid.py --real_dir dataset/Training_dataset/img \
                           --gen_dir ./generate_images \
                           --save_dir ./select_images \
                           --batch_size 256 \
                           --num_sample 8 \
                           --freq 2 \
                           --model resnet34
    ```
5. **轉換為提交格式**
   ```shell
   python convert_to_submit.py --image_dir select_images \ 
                               --save_dir ./submit
   ```
