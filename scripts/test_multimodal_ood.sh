# python train_multimodal_faithful.py --run_name mm_front_camera_ood_city_b_hor2 --split_type city-boston --horizon_sec 2 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_city_b_hor4 --split_type city-boston --horizon_sec 4 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_city_b_hor6 --split_type city-boston --horizon_sec 6 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_map_b_seaport_hor2 --split_type map-boston-seaport --horizon_sec 2 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_map_b_seaport_hor4 --split_type map-boston-seaport --horizon_sec 4 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_map_b_seaport_hor6 --split_type map-boston-seaport --horizon_sec 6 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_object_animal_hor2 --split_type object-animal --horizon_sec 2 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_object_animal_hor4 --split_type object-animal --horizon_sec 4 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
# python train_multimodal_faithful.py --run_name mm_front_camera_ood_object_animal_hor6 --split_type object-animal --horizon_sec 6 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10
python train_multimodal_faithful.py --run_name mm_all_cam_front_camera_ood_object_animal_hor2 --split_type object-animal --horizon_sec 2 --max_obstacles 30 --preprocessed_vid_fea --halt_max_steps 16 --gpu_id 0 --epochs 1000 --lr 1e-5 --config_batch_size 16 --eval_every_n_epochs 10 \
    --camera F \
  --camera FL \
  --camera FR \
  --camera B \
  --camera BL \
  --camera BR