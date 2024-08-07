# reparameterization
# python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.rep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name exp

# train random NAS
# python train_nas.py --data coco.yaml --cfg models/nas/yolov5s.vae.one_branch.nas.yaml --batch-size 32 --device 0 --epochs 50 --name exp1

# train single branch NAS
python train_nas.py --data coco.yaml --cfg models/nas/yolov5s.supernet.one_branch.yaml --batch-size 16 --device 0 --epochs 2 --name exp_nas_vae --split