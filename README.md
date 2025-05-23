# MAA (Multi-GAN Adversarial Analysis) 🚀

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

基于multi GAN对抗训练的金融量化因子时序预测模型

[![GitHub stars](https://img.shields.io/github/stars/Qmacd/MAA_UI?style=social)](https://github.com/Qmacd/MAA_UI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Qmacd/MAA_UI?style=social)](https://github.com/Qmacd/MAA_UI/network/members)
[![GitHub issues](https://img.shields.io/github/issues/Qmacd/MAA_UI)](https://github.com/Qmacd/MAA_UI/issues)

</div>

## 📖 项目简介

MAA是一个基于多生成器对抗网络（Multi-GAN）的金融时序预测框架，通过多个生成器的对抗训练来提高预测精度。该项目提供了完整的训练、预测和可视化功能。

### ✨ 主要特点

<div align="center">

| 功能 | 描述 |
|:---:|:---:|
| 🎯 多生成器对抗训练 | 使用多个生成器进行对抗训练，提高预测精度 |
| 📊 实时训练可视化 | 实时展示训练过程中的损失和指标变化 |
| 🌐 友好的Web界面 | 提供直观的Web操作界面，易于使用 |
| 🔄 自定义模型 | 支持上传自定义模型进行训练 |
| 📈 结果展示 | 完整的预测结果和可视化图表展示 |

</div>

## 🔧 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU版本)
- PyTorch 1.8+
- Flask
- pandas
- numpy
- matplotlib

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/Qmacd/MAA_UI.git
cd MAA_UI
```

### 2. 创建虚拟环境
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动服务
```bash
python app.py
```
服务将在 http://localhost:7000 启动

## 📝 使用指南

### 1. 模型训练

<div align="center">

| 步骤 | 操作 |
|:---:|:---:|
| 1️⃣ | 打开Web界面 |
| 2️⃣ | 设置基本参数（数据路径、输出目录等） |
| 3️⃣ | 设置训练参数（批次大小、学习率等） |
| 4️⃣ | 选择特征列和目标列 |
| 5️⃣ | 点击"开始训练模型" |

</div>

### 2. 模型预测

<div align="center">

| 步骤 | 操作 |
|:---:|:---:|
| 1️⃣ | 设置预测参数 |
| 2️⃣ | 点击"开始预测" |
| 3️⃣ | 查看预测结果和可视化图表 |

</div>

### 3. 自定义模型

项目支持上传自定义模型，模型需要满足以下要求：

1. 继承自`torch.nn.Module`
2. 实现`forward`方法
3. 输入输出维度符合要求

示例模型结构：
```python
class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 定义网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions
```

## 📁 项目结构

```
MAA_UI/
├── 📂 app.py              # Web服务主程序
├── 📂 run_multi_gan_UI.py # 训练主程序
├── 📂 run_multi_gan_pred.py # 预测主程序
├── 📂 templates/          # Web界面模板
│   └── 📄 UI.html        # 主界面
├── 📂 custom_models/      # 自定义模型目录
├── 📂 out_put/           # 输出目录
│   ├── 📂 multi/         # 训练输出
│   └── 📂 ckpt/          # 模型检查点
└── 📂 database/          # 数据文件目录
```

## ⚠️ 注意事项

### 数据格式要求
- CSV文件格式
- 第一行为列名
- 数值型数据

### 模型训练建议
- 建议使用GPU进行训练
- 可以通过调整batch_size和learning_rate优化训练效果

### 预测使用注意
- 确保使用与训练时相同的特征列
- 检查点路径需要正确设置

## ❓ 常见问题

<details>
<summary>Q: 如何选择合适的特征列？</summary>
A: 建议选择与目标变量相关性强的特征，可以通过相关性分析确定。
</details>

<details>
<summary>Q: 训练时间过长怎么办？</summary>
A: 可以尝试减小batch_size或使用GPU加速。
</details>

<details>
<summary>Q: 预测结果不准确？</summary>
A: 检查特征选择是否合适，可以尝试调整模型参数或增加训练轮数。
</details>

## 📅 更新日志

### v1.0.0 (2025-05-23)
- 🎉 初始版本发布
- ✨ 实现基本训练和预测功能
- 🌐 添加Web界面
- 🔄 支持自定义模型上传

## 🤝 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。在提交PR之前，请确保：

1. 代码符合项目规范
2. 添加了必要的测试
3. 更新了相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Qmacd">Qmacd</a></sub>
</div>