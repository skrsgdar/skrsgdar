# FLDP
Differential privacy in FL implemented in pytorch

## Structure
- `centralized_baseline`: the centralized baselines and scripts for searching hyper-parameters
  - `local_train.py`: code for local training framework, which is for 
    - general(non-private) training, 
    - nsgd-fix-lr and 
    - nsgd-vary-lr
  - `rsgd_ar.py`: code for rsgd-ar algorithm
  - `run_general.sh`: script for non-private training
  - `run_nsgd_fix_lr.sh`: script for nsgd-fix-lr baseline
  - `run_nsgd_vary_lr.sh`: script for nsgd-vary-lr baseline
  - `run_rsgd_ar.sh`: script for rsgd-ar baseline
- `distributed_baseline`: distributed baseline
  - TODO


## Requirements
- pyTorch
- numpy
- opacus
- pandas