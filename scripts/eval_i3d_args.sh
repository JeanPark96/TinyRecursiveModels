LOSS=$1
HALT=$2
BATCH=$3

if [ "$LOSS" = "dh" ]; then
    LOSS_OPTION="dense_head"
elif [ "$LOSS" = "nms" ]; then
    LOSS_OPTION="soft_nms"
else
    echo "Unknown LOSS type: $LOSS"
    exit 1
fi

python test_video_i3d.py --model_pth "/home/hlpark/TinyRecursiveModels/video_trm_checkpoints/dcf_se_dh_trunc_b${BATCH}_c13_halt${HALT}_lr3/best.pth" \
    --run_name "eval_dcf_se_${LOSS}_trunc_b${BATCH}_c${L}${H}_halt${HALT}_lr${LR}" --halt_max_steps $HALT --hidden_size 256 --config_batch_size $BATCH --loss_option $LOSS_OPTION \
     --data_root /home/hlpark/common-data/jean --gpu_id $4