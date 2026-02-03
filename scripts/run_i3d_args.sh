LOSS=$1
LR=$2
H=$3
L=$4
HALT=$5
BATCH=$6

if [ "$LOSS" = "dh" ]; then
    LOSS_OPTION="dense_head"
elif [ "$LOSS" = "nms" ]; then
    LOSS_OPTION="soft_nms"
else
    echo "Unknown LOSS type: $LOSS"
    exit 1
fi

python train_video_i3d.py --run_name "dcf_se_${LOSS}_trunc_b${BATCH}_c${L}${H}_halt${HALT}_lr${LR}" --L_cycles $L --H_cycles $H --halt_max_steps $HALT --epochs 100 --hidden_size 256 --config_batch_size $BATCH --loss_option $LOSS_OPTION \
 --lr_schedule --lr "1e-${LR}" --data_root /home/hlpark/common-data/jean --gpu_id $7