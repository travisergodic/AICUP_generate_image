task=$1
export LOGDIR=coco-edge/coco-upsample/ 
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0 --image_size 256 --super_res 64 --num_res_blocks 2 --finetune_decoder True"
TRAIN_FLAGS="--lr 1e-5 --batch_size 4"
WEIGHT_FLAGS="--model_path pretrained/upsample_edge_decoder.pt"
DIFFUSION_FLAGS="--noise_schedule linear"
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train_${task}.txt --val_data_dir ./dataset/COCOSTUFF_val_${task}.txt --mode coco-edge"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $WEIGHT_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS

