task=$1
export CUDA_LAUNCH_BLOCKING=1
export LOGDIR=coco-edge/coco-64-stage1_${task}/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0. --image_size 64 --finetune_decoder False"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 10  --schedule_sampler loss-second-moment --lr_anneal_steps 80000"
WEIGHT_FLAGS="--model_path pretrained/base_edge_decoder.pt --encoder_path pretrained/base_edge_encoder.pt"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train_${task}.txt --val_data_dir ./dataset/COCOSTUFF_val_${task}.txt --mode coco-edge"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $WEIGHT_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS

task=$1
export CUDA_LAUNCH_BLOCKING=1
export LOGDIR=coco-edge/coco-64-stage1-cont_${task}/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 10  --schedule_sampler loss-second-moment --lr_anneal_steps 20000"
WEIGHT_FLAGS="--model_path pretrained/base_edge_decoder.pt --encoder_path coco-edge/coco-64-stage1/checkpoints/ema_0.999_080000.pt"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train_${task}.txt --val_data_dir ./dataset/COCOSTUFF_val_${task}.txt --mode coco-edge"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $WEIGHT_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS

task=$1
export CUDA_LAUNCH_BLOCKING=1
export LOGDIR=coco-edge/coco-64-stage2-decoder_${task}/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder True"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 16 --schedule_sampler loss-second-moment"
WEIGHT_FLAGS="--model_path pretrained/base_edge_decoder.pt --encoder_path coco-edge/coco-64-stage1-cont/checkpoints/ema_0.999_020000.pt"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 2.5"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train_${task}.txt --val_data_dir ./dataset/COCOSTUFF_val_${task}.txt --mode coco-edge"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
 

