python train.py \
 --batch_size 1 \
 --num_workers 0 \
 --task_name "smoke_test" \
 --checkpoint "SAM_Med3D/ckpt/sam_med3d_turbo.pth" \
 --device "cpu" \
 --num_epochs 1 \
 --accumulation_steps 1 \
 --img_size 128 \
 --lr 8e-5