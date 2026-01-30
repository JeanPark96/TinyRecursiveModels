# 1: gpu
# 2: n_horizon
# 3: max_predict
# 4: lr
# 5: run_name
# 6: tboard_name
# 7: h cycles
# 8: l cycles
# 9: deep supervision steps
# 10: halt config path
echo ${10}

python train_unimodal_faithful.py --run_name $5 --tboard_name $6 --halt_max_steps $9 --gpu_id $1 --horizon_sec $2 \
                                    --max_obstacles 30 --max_predict $3 --epochs 10000 --lr $4 \
                                    --lr_schedule --config_batch_size 32 --H_cycles $7 --L_cycles $8 \
                                    --halt_config ${10} --halt_verbose