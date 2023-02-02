set -e

device=$1
data=$2

echo "device, "$device, "data", $data

# general training
bsizes=(64 16 32 64 128)
n_epoch=(100 200 300)
lrs=(0.1 0.01 0.001)
decays=(0. 0.001 0.01 0.1)

exp_dir=exp_out/dpsgd_${data}.txt

if [ ! -d "exp_out" ];then
    mkdir exp_out
fi

echo "HPO starts..."

for (( b=0; b<${#bsizes[@]}; b++ ))
do
  for (( e=0; e<${#n_epoch[@]}; e++ ))
  do
    for (( l=0; l<${#lrs[@]}; l++ ))
    do
      for (( d=0; d<${#decays[@]}; d++ ))
      do
        echo "batch_size ${bsizes[$b]}, n_epoch ${n_epoch[$e]}, lr ${lrs[$l]}, weight_decay ${decays[$d]}"
        python local_train.py --device ${device} \
                              --data_type ${data} \
                              --batch_size ${bsizes[$b]} \
                              --n_epoch ${n_epoch[$e]} \
                              --opt_type SGD --lr ${lrs[$l]} --weight_decay ${decays[$d]}\
                              --eval_freq 50 \
                              --grad_clip 0.5 \
                              --perturb_grad True --scale_grad 0.01\
                              --perturb_weight False >>${exp_dir} 2>&1
      done
    done
  done
done

echo "HPO ends."