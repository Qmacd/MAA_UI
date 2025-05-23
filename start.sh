#!/bin/bash

# 创建必要的目录
mkdir -p logs
mkdir -p custom_models
mkdir -p out_put/ckpt
mkdir -p out_put/multi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，如果有多个GPU可以修改
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh  # 根据你的conda安装路径修改
conda activate quant  # 使用quant环境

# 启动Gunicorn
gunicorn -c gunicorn_config.py app:app 