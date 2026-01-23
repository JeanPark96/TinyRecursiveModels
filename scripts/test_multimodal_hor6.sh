python train_multimodal_faithful.py --run_name test_all_cam_6 --horizon_sec 6 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 8 --gpu_id 3 --epochs 300 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 5 --max_predict 8 \
    --camera F \
  --camera FL \
  --camera FR \
  --camera B \
  --camera BL \
  --camera BR
#python train_multimodal_faithful.py --run_name 0111_mm_exp3 --horizon_sec 4 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 2 --epochs 500 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 5]
#python train_multimodal_faithful.py --run_name 0111_mm_exp4 --horizon_sec 6 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 2 --epochs 500 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 5