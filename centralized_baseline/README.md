## Running Examples
- General (non-private) centralized training
```bash
python local_train.py --device 0 \
    --data_type ADULT \
    --batch_size 64 \
    --n_epoch 100 \
    --opt_type SGD \
    --lr 0.01 \
    --weight_decay 0.
```
- nsgd-fix-lr
  - compared with non-private centralized training, need to set `perturb_weight`, `scale_weight` and `lr_decay`
```bash
 python local_train.py --device 0 \
      --data_type ADULT \
      --batch_size 64 \
      --n_epoch 100 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --perturb_weight True \
      --scale_weight 0.01 \
      --lr_decay False
```

- nsgd-vary-lr
  - Compared with nsgd-fix-lr, set `lr_decay` as True
```bash
 python local_train.py --device 0 \
      --data_type ADULT \
      --batch_size 64 \
      --n_epoch 100 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --perturb_weight True \
      --scale_weight 0.01 \
      --lr_decay True
```
 
- rsgd-ar
```bash
python rsgd_ar.py --device 0 \
    --data_type ADULT \
    --batch_size 64 \
    --n_epoch 100 \
    --opt_type SGD \
    --lr 0.01 \
    --weight_decay 0. \
    --eval_freq 50 \
    --tau 10 \
    --perturb_weight True \
    --scale_weight 0.01
```

