# 1: gpu
# 2: n_horizon
# 3: max_obstacles
# 4: lr
# 5: run_name

python train_unimodal_faithful.py --run_name $5 --halt_max_steps 16 --gpu_id $1 --horizon_sec $2 --max_obstacles $3 --epochs 10000 --lr $4 --lr_schedule --config_batch_size 32