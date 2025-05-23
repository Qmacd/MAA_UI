# MAA (Multi-GAN Adversarial Analysis)

基于multi GAN对抗训练的金融量化因子时序预测模型

## 项目简介

MAA是一个基于多生成器对抗网络（Multi-GAN）的金融时序预测框架，通过多个生成器的对抗训练来提高预测精度。该项目提供了完整的训练、预测和可视化功能。

### 主要特点

- 🎯 多生成器对抗训练
- 📊 实时训练过程可视化
- 🌐 友好的Web界面
- 🔄 支持自定义模型上传
- 📈 完整的预测结果展示

## 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU版本)
- PyTorch 1.8+
- Flask
- pandas
- numpy
- matplotlib

## 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/MAA.git
cd MAA
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动Web服务

```bash
python app.py
```

服务将在 http://localhost:7000 启动

### 2. 模型训练

1. 打开Web界面
2. 设置基本参数：
   - 数据文件路径
   - 输出目录
   - 检查点目录
3. 设置训练参数：
   - 批次大小
   - 训练轮数
   - 学习率
   - 训练集比例
   - 窗口大小
4. 选择特征列和目标列
5. 点击"开始训练模型"

### 3. 模型预测

1. 设置预测参数：
   - 输入数据路径
   - 输出结果路径
   - 模型检查点路径
2. 点击"开始预测"
3. 查看预测结果和可视化图表

### 4. 自定义模型

项目支持上传自定义模型，模型需要满足以下要求：

1. 继承自`torch.nn.Module`
2. 实现`forward`方法
3. 输入输出维度符合要求

示例模型结构（参考`demo.py`）：
```python
class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # 模型结构定义
        ...
    
    def forward(self, x):
        # 前向传播逻辑
        ...
```

## 项目结构

```
MAA/
├── app.py              # Web服务主程序
├── run_multi_gan_UI.py # 训练主程序
├── run_multi_gan_pred.py # 预测主程序
├── templates/          # Web界面模板
│   └── UI.html        # 主界面
├── custom_models/      # 自定义模型目录
├── out_put/           # 输出目录
│   ├── multi/         # 训练输出
│   └── ckpt/          # 模型检查点
└── database/          # 数据文件目录
```

## 注意事项

1. 数据格式要求：
   - CSV文件格式
   - 第一行为列名
   - 数值型数据

2. 模型训练：
   - 建议使用GPU进行训练
   - 可以通过调整batch_size和learning_rate优化训练效果

3. 预测使用：
   - 确保使用与训练时相同的特征列
   - 检查点路径需要正确设置

## 常见问题

1. Q: 如何选择合适的特征列？
   A: 建议选择与目标变量相关性强的特征，可以通过相关性分析确定。

2. Q: 训练时间过长怎么办？
   A: 可以尝试减小batch_size或使用GPU加速。

3. Q: 预测结果不准确？
   A: 检查特征选择是否合适，可以尝试调整模型参数或增加训练轮数。

## 更新日志

### v1.0.0 (2024-03-20)
- 初始版本发布
- 实现基本训练和预测功能
- 添加Web界面
- 支持自定义模型上传

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

MIT License