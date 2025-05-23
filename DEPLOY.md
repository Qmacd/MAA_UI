# MAA项目部署指南

## 环境要求

- Anaconda/Miniconda
- Python 3.8+
- CUDA 11.1+ (用于GPU支持)
- NVIDIA GPU (推荐RTX 2080 Ti或更高)
- 16GB+ RAM
- 100GB+ 磁盘空间

## 安装步骤

1. 安装CUDA和cuDNN
```bash
# 检查CUDA是否可用
nvidia-smi

# 安装CUDA工具包
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
sudo sh cuda_11.1.0_455.23.05_linux.run
```

2. 创建Conda环境
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate quant
```

3. 配置GPU
```bash
# 检查GPU是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

## 启动服务

1. 确保目录结构正确
```bash
mkdir -p logs custom_models out_put/ckpt out_put/multi
```

2. 启动服务
```bash
chmod +x start.sh
./start.sh
```

3. 检查服务状态
```bash
ps aux | grep gunicorn
tail -f logs/access.log
```

## 配置说明

1. GPU配置
- 在`start.sh`中设置`CUDA_VISIBLE_DEVICES`来选择使用的GPU
- 如果有多个GPU，可以修改为`0,1,2`等

2. 性能优化
- 在`gunicorn_config.py`中调整`workers`数量
- 根据服务器CPU核心数调整，建议设置为CPU核心数的2倍+1

3. 日志管理
- 访问日志：`logs/access.log`
- 错误日志：`logs/error.log`
- 进程ID：`logs/gunicorn.pid`

## 常见问题

1. GPU内存不足
- 减小batch_size
- 使用梯度累积
- 启用混合精度训练

2. 服务无法启动
- 检查端口7000是否被占用
- 检查日志文件权限
- 确认CUDA环境变量正确设置
- 确认conda环境已正确激活

3. 性能问题
- 检查GPU利用率：`nvidia-smi`
- 检查CPU使用率：`top`
- 检查内存使用：`free -h`

## 维护建议

1. 定期清理
```bash
# 清理日志
find logs/ -type f -name "*.log" -mtime +7 -delete

# 清理模型检查点
find out_put/ckpt/ -type f -name "*.pt" -mtime +30 -delete
```

2. 监控系统
- 使用`nvidia-smi`监控GPU
- 使用`top`监控CPU和内存
- 检查日志文件大小

3. 备份策略
- 定期备份模型检查点
- 备份配置文件
- 保存训练日志

4. Conda环境管理
```bash
# 更新环境
conda env update -f environment.yml

# 导出环境
conda env export > environment.yml

# 删除环境
conda env remove -n quant
``` 