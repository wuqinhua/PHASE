#!/bin/bash

# 进入工作目录
cd /home/wuqinhua/Project/PHASE/Code_final/COVID19/2_Model/2.1.1_Model_Compare/cloudpred

source /home/wuqinhua/anaconda3/etc/profile.d/conda.sh
conda activate phase


export OPENBLAS_NUM_THREADS=8


centers="5 10 15 20 25"
dim=100
valid=0.25
test=0.25
data_path="/data/wuqinhua/phase/covid19/Compare/data"


for seed in {1..10}; do
    figroot="fig/covid19_${seed}_"


    CUDA_VISIBLE_DEVICES=0 python3 -m cloudpred ${data_path} -t log --logfile log/lupus/linear_${seed} --linear --centers ${centers} --dim ${dim} --seed ${seed} --valid ${valid} --test ${test} --figroot ${figroot}
    CUDA_VISIBLE_DEVICES=0 python3 -m cloudpred ${data_path} -t log --logfile log/lupus/generative_${seed} --generative --centers ${centers} --dim ${dim} --seed ${seed} --valid ${valid} --test ${test} --figroot ${figroot}
    CUDA_VISIBLE_DEVICES=0 python3 -m cloudpred ${data_path} -t log --logfile log/lupus/genpat_${seed} --genpat --centers ${centers} --dim ${dim} --seed ${seed} --valid ${valid} --test ${test} --figroot ${figroot}
    CUDA_VISIBLE_DEVICES=0 python3 -m cloudpred ${data_path} -t log --logfile log/lupus/deepset_${seed} --deepset --centers ${centers} --dim ${dim} --seed ${seed} --valid ${valid} --test ${test} --figroot ${figroot}
    CUDA_VISIBLE_DEVICES=0 python3 -m cloudpred ${data_path} -t log --logfile log/lupus/cloudpred_${seed} --cloudpred --centers ${centers} --dim ${dim} --seed ${seed} --valid ${valid} --test ${test} --figroot ${figroot}
done


conda deactivate
