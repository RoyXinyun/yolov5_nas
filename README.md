# Yolov5_split

## encoder and decoder number -> module
- Encoder
    ```shell
    -1 -> None
    0 -> Encoder
    1 -> TinyEncoder
    2 - > TinyEncoder + ca_type=1
    3 - > TinyEncoder + ca_type=2
    4 - > RepEncoder
    5 - > RepInt8Encoder
    6 - > EqRepEncoder
    ```
- Decoder
    ```shell
    -1 -> None
    0 -> Decoder
    1 -> LKADecoder
    2 - > LKADecoder + mslka
    3 - > RepDecoder
    4 - > RepDecoder_1 # same param with Encoder 
    5 - > RepDecoderAddParams
    6 - > EqRepDecoder
    ```


## Training
- yolov5 small model
    ```python
    #one branch
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vae_one_branch_te

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vae_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.rep_1.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vae_one_branch_rep_1_r4

    # conv attention
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te.r4.ca.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 3 --epochs 50 --name vae_one_branch_te_r4_ca

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te.r4.ca.yaml --weights runs/train/vae_one_branch_te_r4_ca_only_det/weights/last.pt --batch-size 32 --split --freeze 5 --unfreeze_layer 6 --device 0 --epochs 50 --name vae_one_branch_te_r4_ca_onlydet_ftcloud --only_det --hyp data/hyps/hyp.finetune.yaml

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te_ca.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vae_one_branch_te_ca_r4
    # baseline
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.eqrep.sym.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 26 --unfreeze_layer 5 --device 1 --epochs 50 --name vae_one_branch_eqrep_sym_only_det_freezeall --only_det

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.eqrep.sym.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vae_one_branch_eqrep_sym_only_det --only_det

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.eqrep.r4.sym.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vae_one_branch_eqrep_r4_sym_only_det --only_det

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.eqrep.r4.sym.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vae_one_branch_eqrep_r4_sym

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.eqrep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vae_one_branch_eqrep_r4

    # reparameterization
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.rep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vae_one_branch_rep_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.rep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vae_one_branch_rep_r4_only_det --only_det

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.repe.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 5 --epochs 50 --name vae_one_branch_repe_r4_only_det --only_det

    # norm
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vaenorm_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.rep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vaenorm_one_branch_rep_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.rep_1.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vaenorm_one_branch_rep_1_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.rep.addparams.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vaenorm_one_branch_rep_addparams_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.eqrep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vaenorm_one_branch_eqrep_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.rep.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 3 --epochs 50 --name vaenorm_one_branch_rep_r4_only_det --only_det
    
    # msca
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vaenorm.one_branch.rep.r4.msca.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vaenorm_one_branch_rep_r4_msca

    # vqvae
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vqvae_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae1.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vqvae1_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae2.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vqvae2_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae3.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vqvae3_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae3.one_branch.te.r4.yaml --weights tmp.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name vqvae3_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae4.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vqvae4_one_branch_te_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae4.one_branch.te.r4.yaml --weights tmp.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 2 --epochs 50 --name vqvae4_one_branch_te_r4
    
    # kd
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.one_branch.te.r4.yaml --weights runs/train/vae_one_branch_te_r4/weights/last.pt --batch-size 32 --split --device 0 --epochs 50 --name vae_one_branch_te_r4_kd_ft --distill --t_weights checkpoints/yolov5s.pt --hyp data/hyps/hyp.finetune_sp.yaml
    
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vqvae.one_branch.te.r4.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name vae_one_branch_te_r4_kd --distill --t_weights checkpoints/yolov5s.pt

    # two branch
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.two_branch.yaml --weights checkpoints/yolov5s.7.pt --batch-size 32 --split --freeze 27 --unfreeze_layer 7 --device 1 --epochs 50 --name vae_two_branch

    # three branch
    python train.py --data coco.yaml --cfg models/vae/yolov5s.vae.three_branch.yaml --weights checkpoints/yolov5s.convert1.pt --batch-size 32 --split --freeze 28  --device 1 --epochs 50 --name vae_three_branch

    # Contrast experiment
    python train.py --data coco.yaml --cfg models/vae/yolov5s.bottlenetpp.one_branch.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 0 --epochs 50 --name bottlenetpp_r4

    python train.py --data coco.yaml --cfg models/vae/yolov5s.bottlenetpp_r128.one_branch.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 25 --unfreeze_layer 5 --device 1 --epochs 50 --name bottlenetpp_r128

    python train.py --data coco.yaml --cfg models/vae/yolov5s.quan.one_branch.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 26 --unfreeze_layer 5 --device 0 --epochs 50 --name quan --for_save

    python train.py --data coco.yaml --cfg models/vae/yolov5s.jpeg.one_branch.yaml --weights checkpoints/yolov5s.5.pt --batch-size 1 --split --freeze 26 --unfreeze_layer 5 --device 0 --epochs 50 --name jpeg --for_save

    python train.py --data coco.yaml --cfg models/vae/yolov5s.quan_1bit.one_branch.yaml --weights checkpoints/yolov5s.5.pt --batch-size 32 --split --freeze 26 --unfreeze_layer 5 --device 0 --epochs 50 --name quan_1bit --for_save
    ```

### Speed FLOPs Params Test
```python
python get_flops.py --cfg models/yolov5s.yaml --device 0
```

## Validation
```python
python val.py --weights checkpoints/yolov5s.pt --data coco.yaml --img 640
```

## Split to edge and cloud
- After modify file content
    ```python
    python split_cloud_edge.py
    ```

## Deploy
- Test cloud edge collaboration 
    ```python
    python edge_infer.py
    ```
- Accuracy of alignment with normal inference
    ```python
    python detect.py --source data/images/bus.jpg --weights runs/train/vae_three_branch_std_norm_group_conv_d128/weights/last.pt --device 0
    python detect.py --source data/images/zidane.jpg --weights checkpoints/yolov5s.pt --device 0
    ```