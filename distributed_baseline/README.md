# Examples for FL training
- example for sk-nsgd-non-private
```bash
python sk_nsgd.py --device 0 \
      --data_type ADULT \
      --batch_size 64 \
      --n_client 10 \
      --n_round 100 \
      --server_lr 0.1 \
      --local_update_steps 1 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --scale_parameter 5. \
      --possion_rate 2. \
      --l2_sensitivity 1.
```

- example for sk-nsgd-fix-step
  - add `perturb_local_weight` and `scale_noise` to perturb local trained model
```bash
python sk_nsgd.py --device 0 \
      --data_type ADULT \
      --batch_size 64 \
      --n_client 10 \
      --n_round 100 \
      --server_lr 0.1 \
      --local_update_steps 1 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --scale_parameter 5. \
      --possion_rate 2. \
      --l2_sensitivity 1. \
      --perturb_local_weight True \
      --scale_noise 0.01
```

- example for sk-nsgd-vary-step
  - add `lr_decay`
```bash
python sk_nsgd.py --device 0 \
      --data_type ADULT \
      --batch_size 64 \
      --n_client 10 \
      --n_round 100 \
      --server_lr 0.1 \
      --local_update_steps 1 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --scale_parameter 5. \
      --possion_rate 2. \
      --l2_sensitivity 1. \
      --perturb_local_weight True \
      --scale_noise 0.01 \
      --lr_decay True 
```

- example for distributed_skellam.py
```bash
python distributed_skellam.py --device 0 \
      --data_type ADULT \
      --q 0.1 \
      --n_client 10 \
      --n_round 100 \
      --server_lr 0.1 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --grad_clip 0.5 \
      --scale_parameter 5. \
      --possion_rate 2. \
      --l2_sensitivity 1.
```

- example for sk_rsgd_ar.py
  - Note: length of args.local_update_epochs should be equal to args.n_round
```bash
python sk_rsgd_ar.py --device 0 \
      --batch_size 64  \
      --data_type ADULT \
      --n_client 10 \
      --eval_freq 10 \
      --server_lr 1 \
      --n_round 3 \
      --local_update_epochs [1,5,5] \
      --local_mini_batches 10 \
      --opt_type SGD \
      --lr 0.01 \
      --weight_decay 0. \
      --scale_noise 0.1 \
      --tau 1 \
      --scale_parameter 5. \
      --possion_rate 2. \
      --l2_sensitivity 100
```