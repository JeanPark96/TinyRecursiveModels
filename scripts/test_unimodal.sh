# 1: gpu
# 2: n_horizon
# 3: max_obstacles
# 4: run_name

python train_unimodal_faithful.py --run_name $4 --halt_max_steps 16 --gpu_id $1 --horizon_sec $2 --max_obstacles $3 --epochs 10000 --lr 1e-5 --lr_schedule --config_batch_size 32