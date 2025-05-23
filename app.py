# app.py
import os
import sys
from flask import Flask, request, render_template, jsonify, Response, send_file
import threading
import argparse
from run_multi_gan_UI import run_experiments
import pandas as pd
import queue
import json
import re
import psutil
import torch
import time
from werkzeug.utils import secure_filename
import torch.nn as nn
import importlib.util
import inspect
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import StringIO
import traceback
from contextlib import redirect_stdout
import shutil
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制为 16MB

# 配置上传文件夹
app.config['UPLOAD_FOLDER'] = 'custom_models'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 日志队列（用于实时日志输出）
log_queue = queue.Queue()

# 添加模型上传配置
ALLOWED_EXTENSIONS = {'py', 'pt', 'pth'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 默认参数
DEFAULT_PARAMS = {
    "data_path": "database/zx_processed_黄金_day.csv",
    "output_dir": "out_put/multi",
    "ckpt_dir": "out_put/ckpt",
    "feature_columns": ["1-4,6-10,15-17", "1-5,11-14", "1-4,18-29"],
    "target_columns": [[1]],
    "log_diff": False,
    "N_pairs": 3,
    "window_sizes": [5, 10, 15],
    "batch_size": 64,
    "mode": "train",
    "device": ['cuda:0'] if torch.cuda.is_available() else ['cpu'],
    "random_seed": 3407,
    "num_epochs": 1024,
    "lr": 2e-5,
    "train_split": 0.7,
    "distill_epochs": 1,
    "cross_finetune_epochs": 5,
    "generators": ["gru", "lstm", "transformer"],
    "discriminators": None,
    "amp_dtype": "none",
    "start_timestamp": 31,
    "end_timestamp": -1,
    "ckpt_path": "latest",
    "num_classes": 3
}

# 添加模型转换配置
MODEL_TEMPLATES = {
    'generator': """
class CustomGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 这里会自动插入原始模型的结构
        {model_structure}
        
    def forward(self, x):
        # 输入: [batch_size, sequence_length, input_dim]
        return self.model(x)
""",
    'discriminator': """
class CustomDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 这里会自动插入原始模型的结构
        {model_structure}
        
    def forward(self, x):
        # 输入: [batch_size, sequence_length, input_dim]
        return self.model(x)
"""
}

def check_system_resources():
    """检查系统资源状态"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    
    return {
        "cpu_usage": cpu_percent,
        "memory_usage": memory.percent,
        "memory_available": memory.available / 1024**2,  # 转换为MB
        "gpu_memory_used": gpu_memory
    }


def log_system_status():
    """记录系统状态"""
    resources = check_system_resources()
    log_queue.put(f"系统状态:")
    log_queue.put(f"CPU使用率: {resources['cpu_usage']}%")
    log_queue.put(f"内存使用率: {resources['memory_usage']}%")
    log_queue.put(f"可用内存: {resources['memory_available']:.2f}MB")
    if resources['gpu_memory_used'] is not None:
        log_queue.put(f"GPU内存使用: {resources['gpu_memory_used']:.2f}MB")


def log_generator():
    """日志生成器，用于实时日志输出"""
    try:
        while True:
            try:
                log_line = log_queue.get(timeout=1)
                if log_line:  # 只发送非空日志
                    yield f"data: {json.dumps({'message': log_line})}\n\n"
            except queue.Empty:
                # 不再发送心跳包
                continue
    except GeneratorExit:
        # 客户端断开连接时的处理
        print("客户端断开日志连接")
    except Exception as e:
        print(f"日志生成器错误: {str(e)}")
        yield f"data: {json.dumps({'message': f'日志系统错误: {str(e)}'})}\n\n"

@app.route("/stream")
def stream_logs():
    """SSE 接口，用于浏览器实时接收日志"""
    def generate():
        try:
            for log in log_generator():
                yield log
        except Exception as e:
            print(f"日志流错误: {str(e)}")
            yield f"data: {json.dumps({'message': f'日志流错误: {str(e)}'})}\n\n"
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
        }
    )

@app.route("/columns")
def get_columns():
    """读取 CSV 文件列名并返回列索引与列名"""
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "文件不存在"})
    try:
        df = pd.read_csv(path)
        # 排除所有日期相关的列
        date_keywords = ['date', 'time', 'day', 'month', 'year', 'week']
        columns_with_index = [
            {"index": i, "name": col} 
            for i, col in enumerate(df.columns) 
            if not any(keyword in col.lower() for keyword in date_keywords)
        ]
        print(f"加载的列名: {columns_with_index}")  # 添加日志
        return jsonify({"columns": columns_with_index})
    except Exception as e:
        print(f"加载列名错误: {str(e)}")  # 添加错误日志
        return jsonify({"error": str(e)})

@app.route("/train", methods=["POST"])
def train_model():
    """启动训练任务"""
    try:
        print("接收到训练请求")
        print("请求表单数据:", request.form)
        print("请求文件:", request.files)
        
        # 从FormData中获取数据
        data = {
            "train_csv": request.form.get("train_csv"),
            "save_path": request.form.get("save_path"),
            "feature_columns": json.loads(request.form.get("feature_columns")),
            "target_column": json.loads(request.form.get("target_column")),
            "batch_size": request.form.get("batch_size"),
            "num_epochs": request.form.get("num_epochs"),
            "lr": request.form.get("lr"),
            "train_split": request.form.get("train_split"),
            "window_sizes": request.form.get("window_sizes"),
            "model_config": json.loads(request.form.get("model_config"))
        }
        
        print("解析后的数据:", data)
        
        if not data["train_csv"]:
            return jsonify({"error": "无效的请求数据"}), 400
        
        # 检查模型配置
        model_config = data.get('model_config')
        if not model_config:
            return jsonify({'error': '未提供模型配置'}), 400
            
        # 验证模型配置
        if not model_config.get('models'):
            return jsonify({'error': '请至少配置一个模型（自定义或预设）'}), 400
            
        # 验证模型配置
        for model in model_config.get('models', []):
            if model.get('type') == 'custom':
                # 处理自定义模型
                model_file = request.files.get(f'model{model.get("index")}')
                if not model_file:
                    return jsonify({'error': f'模型 {model.get("index")} 缺少文件'}), 400
                # 保存上传的模型文件
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], f'model_{model.get("index")}.pt')
                model_file.save(model_path)
                model['file'] = model_path
                
                # 加载模型描述
                model_dir = os.path.dirname(model['file'])
                desc_path = os.path.join(model_dir, 'description.txt')
                if os.path.exists(desc_path):
                    with open(desc_path, 'r', encoding='utf-8') as f:
                        model_desc = f.read()
                        print(f"使用自定义模型: {model['file']}")
                        print(f"模型描述: {model_desc}")
            else:
                # 处理预设模型
                print(f"使用预设模型: {model.get('preset')}")
                # 为预设模型设置默认路径
                model['file'] = os.path.join(app.config['UPLOAD_FOLDER'], f'preset_{model.get("preset")}.pt')

        # 验证特征列和目标列
        try:
            # 验证feature_columns是三个列表
            if not isinstance(data["feature_columns"], list) or len(data["feature_columns"]) != 3:
                raise ValueError("特征列必须是三个GAN组的列表")
            
            # 验证每个特征组是列表
            feature_groups = []
            for group in data["feature_columns"]:
                if not isinstance(group, list):
                    raise ValueError("每个GAN组的特征列必须是列表")
                # 从 "列[1]: column_name" 格式中提取数字
                indices = []
                for col in group:
                    match = re.search(r'列\[(\d+)\]', col)
                    if match:
                        indices.append(int(match.group(1)))
                    else:
                        raise ValueError(f"无法解析列索引: {col}")
                feature_groups.append(indices)
            
            # 验证target_column是列表
            if not isinstance(data["target_column"], list):
                raise ValueError("目标列必须是列表")
            # 从 "列[1]: column_name" 格式中提取数字
            target_indices = []
            for col in data["target_column"]:
                match = re.search(r'列\[(\d+)\]', col)
                if match:
                    target_indices.append(int(match.group(1)))
                else:
                    raise ValueError(f"无法解析列索引: {col}")
            
            print(f"解析的特征组: {feature_groups}")
            print(f"解析的目标列索引: {target_indices}")
            
            # 验证索引范围
            try:
                df = pd.read_csv(data["train_csv"])
                max_index = len(df.columns) - 1
                
                # 验证所有特征组的索引
                for group in feature_groups:
                    if max(group) > max_index:
                        raise ValueError(f"特征列索引超出范围，最大索引为 {max_index}")
                
                # 验证目标列索引
                if max(target_indices) > max_index:
                    raise ValueError(f"目标列索引超出范围，最大索引为 {max_index}")
                
            except Exception as e:
                return jsonify({"error": f"数据验证错误: {str(e)}"}), 400
            
        except Exception as e:
            print(f"列选择解析错误: {str(e)}")
            return jsonify({"error": f"列选择格式错误: {str(e)}"}), 400
        
        # 处理window_sizes
        try:
            window_sizes = json.loads(data.get("window_sizes", "[]"))
            if not isinstance(window_sizes, list):
                raise ValueError("window_sizes 必须是列表")
            if not all(isinstance(x, (int, float)) for x in window_sizes):
                raise ValueError("window_sizes 中的所有值必须是数字")
            if len(window_sizes) == 0:
                raise ValueError("window_sizes 不能为空")
        except json.JSONDecodeError:
            raise ValueError("window_sizes 格式错误：无法解析 JSON")
        except Exception as e:
            print(f"处理window_sizes时出错: {str(e)}")
            return jsonify({"error": f"window_sizes格式错误: {str(e)}"}), 400
        
        # 构建完整的参数字典
        params = {
            "notes": "",
            "data_path": data["train_csv"],
            "output_dir": os.path.dirname(data["save_path"]),
            "ckpt_dir": "ckpt",
            "feature_groups": feature_groups,  # 使用解析后的特征组
            "target_columns": [target_indices],  # 保持列表格式
            "start_timestamp": 31,
            "end_timestamp": -1,
            "window_sizes": window_sizes,  # 使用处理后的window_sizes
            "N_pairs": 3,
            "num_classes": 3,
            "generators": ["gru", "lstm", "transformer"],
            "discriminators": None,
            "distill_epochs": 1,
            "cross_finetune_epochs": 5,
            "device": ['cuda:0'] if torch.cuda.is_available() else ['cpu'],
            "num_epochs": int(data["num_epochs"]),
            "lr": float(data["lr"]),
            "batch_size": int(data["batch_size"]),
            "train_split": float(data["train_split"]),
            "random_seed": 3407,
            "amp_dtype": "none",
            "mode": "train",
            "ckpt_path": "latest",
            "log_diff": False,
            "model_config": model_config,  # 添加模型配置
        }
        
        print(f"训练参数: {params}")
        
        # 创建输出目录
        os.makedirs(params["output_dir"], exist_ok=True)
        
        # 在新线程中运行训练
        def run_training():
            try:
                # 重定向标准输出到日志队列
                class QueueLogger:
                    def write(self, msg):
                        if msg:  # 只检查消息是否存在
                            try:
                                log_queue.put(msg)  # 直接放入队列，不进行strip
                            except Exception as e:
                                print(f"写入日志队列错误: {str(e)}")
                    def flush(self): pass

                old_stdout = sys.stdout
                sys.stdout = QueueLogger()
                
                # 运行训练
                args = argparse.Namespace(**params)
                run_experiments(args)
                
            except Exception as e:
                print(f"训练过程出错: {str(e)}")
                log_queue.put(f"训练过程出错: {str(e)}")
            finally:
                # 恢复标准输出
                sys.stdout = old_stdout
        
        # 启动训练线程
        thread = threading.Thread(target=run_training)
        thread.start()
        
        return jsonify({"message": "训练任务已启动"})
            
    except Exception as e:
        print(f"处理训练请求时出错: {str(e)}")
        return jsonify({"error": f"处理训练请求时出错: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """启动预测任务"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400

        print("接收到的原始数据:", data)  # 打印原始数据

        # 验证特征组和目标列
        try:
            # 验证feature_groups是三个列表
            feature_groups = data.get("feature_groups", [])
            print("原始特征组:", feature_groups)  # 打印原始特征组
            
            if not isinstance(feature_groups, list) or len(feature_groups) != 3:
                raise ValueError("特征组必须是三个列表")
            
            # 验证每个特征组是列表并解析列索引
            parsed_feature_groups = []
            for i, group in enumerate(feature_groups):
                print(f"处理第{i+1}个特征组:", group)  # 打印每个特征组
                if not isinstance(group, list):
                    raise ValueError("每个特征组必须是列表")
                # 从 "列[1]: column_name" 格式中提取数字
                indices = []
                for col in group:
                    print(f"处理列名: {col}")  # 打印每个列名
                    match = re.search(r'列\[(\d+)\]', col)
                    if match:
                        index = int(match.group(1))
                        print(f"提取的索引: {index}")  # 打印提取的索引
                        indices.append(index)
                    else:
                        print(f"无法匹配列名格式: {col}")  # 打印匹配失败的信息
                        raise ValueError(f"无法解析列索引: {col}")
                parsed_feature_groups.append(indices)
            
            # 验证target_columns是列表并解析列索引
            target_columns = data.get("target_columns", [])
            print("原始目标列:", target_columns)  # 打印原始目标列
            
            if not isinstance(target_columns, list):
                raise ValueError("目标列必须是列表")
            # 从 "列[1]: column_name" 格式中提取数字
            parsed_target_columns = []
            for col in target_columns:
                print(f"处理目标列名: {col}")  # 打印每个目标列名
                match = re.search(r'列\[(\d+)\]', col)
                if match:
                    index = int(match.group(1))
                    print(f"提取的目标列索引: {index}")  # 打印提取的目标列索引
                    parsed_target_columns.append(index)
                else:
                    print(f"无法匹配目标列名格式: {col}")  # 打印匹配失败的信息
                    raise ValueError(f"无法解析列索引: {col}")
            
            # 将目标列添加到每个特征组中
            for feature_group in parsed_feature_groups:
                feature_group.extend(parsed_target_columns)
            
            print(f"解析后的特征组（包含目标列）: {parsed_feature_groups}")
            print(f"解析后的目标列: {parsed_target_columns}")
            
            # 验证索引范围
            try:
                df = pd.read_csv(data["input_path"])
                max_index = len(df.columns) - 1
                
                # 验证所有特征组的索引
                for group in parsed_feature_groups:
                    if max(group) > max_index:
                        raise ValueError(f"特征列索引超出范围，最大索引为 {max_index}")
                
                # 验证目标列索引
                if max(parsed_target_columns) > max_index:
                    raise ValueError(f"目标列索引超出范围，最大索引为 {max_index}")
                
            except Exception as e:
                return jsonify({"error": f"数据验证错误: {str(e)}"}), 400
            
        except Exception as e:
            print(f"列选择解析错误: {str(e)}")
            return jsonify({"error": f"列选择格式错误: {str(e)}"}), 400

        # 构建预测参数
        params = {
            "data_path": data["input_path"],
            "output_dir": os.path.dirname(data["output_path"]),
            "ckpt_dir": "out_put/ckpt",  # 使用固定的检查点目录
            "ckpt_path": "20250520_162930",  # 修改为实际的检查点子文件夹名称
            "N_pairs": 3,
            "batch_size": 64,
            "window_sizes": [5, 10, 15],
            "num_classes": 3,
            "feature_groups": parsed_feature_groups,  # 使用包含目标列的特征组
            "target_columns": [parsed_target_columns],  # 保持列表格式
            "device": ['cuda:0'] if torch.cuda.is_available() else ['cpu'],
            "random_seed": 3407,
            "log_diff": False,
            "start_timestamp": 31,
            "end_timestamp": -1,
            "generators": ["transformer", "transformer", "transformer"],  # 修改为与训练时一致的生成器类型
            "discriminators": None,
            "distill_epochs": 1,
            "cross_finetune_epochs": 5,
            "lr": 2e-5,
            "train_split": 0.7,
            "num_epochs": 1  # 预测时不需要训练，设为1
        }

        # 创建输出目录
        os.makedirs(params["output_dir"], exist_ok=True)

        # 在新线程中运行预测
        def run_prediction():
            try:
                # 重定向标准输出到日志队列
                class QueueLogger:
                    def write(self, msg):
                        if msg:  # 只检查消息是否存在
                            try:
                                log_queue.put(msg)  # 直接放入队列，不进行strip
                            except Exception as e:
                                print(f"写入日志队列错误: {str(e)}")
                    def flush(self): pass

                old_stdout = sys.stdout
                sys.stdout = QueueLogger()
                
                # 运行预测
                args = argparse.Namespace(**params)
                from run_multi_gan_pred import run_prediction
                results = run_prediction(args)
                
                # 将结果保存到指定路径
                results_df = pd.DataFrame(results)
                results_df.to_csv(data["output_path"], index=False)
                
                # 获取生成的图片路径
                image_paths = {
                    "generator_losses": os.path.join(params["output_dir"], "generator_losses.png"),
                    "discriminator_losses": os.path.join(params["output_dir"], "discriminator_losses.png"),
                    "overall_losses": os.path.join(params["output_dir"], "overall_losses.png"),
                    "mse_losses": os.path.join(params["output_dir"], "mse_losses.png"),
                    "fitting_curves": [
                        os.path.join(params["output_dir"], f"G{i+1}_Train_fitting_curve.png") for i in range(3)
                    ] + [
                        os.path.join(params["output_dir"], f"G{i+1}_Test_fitting_curve.png") for i in range(3)
                    ],
                    "density_plot": os.path.join(params["output_dir"], "all_predictions_combined_density.png")
                }
                
                # 将图片路径添加到日志中
                log_queue.put(f"预测完成，结果已保存到: {data['output_path']}")
                log_queue.put(f"可视化图片已生成，路径: {json.dumps(image_paths)}")
                
            except Exception as e:
                print(f"预测过程出错: {str(e)}")
                log_queue.put(f"预测过程出错: {str(e)}")
            finally:
                # 恢复标准输出
                sys.stdout = old_stdout
        
        # 启动预测线程
        thread = threading.Thread(target=run_prediction)
        thread.start()
        
        return jsonify({"message": "预测任务已启动"})
            
    except Exception as e:
        print(f"处理预测请求时出错: {str(e)}")
        return jsonify({"error": f"处理预测请求时出错: {str(e)}"}), 500

@app.route("/get_prediction_images")
def get_prediction_images():
    """获取预测结果图片"""
    try:
        output_dir = request.args.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            return jsonify({"error": "输出目录不存在"}), 404

        # 获取所有图片路径
        image_paths = {
            "generator_losses": os.path.join(output_dir, "generator_losses.png"),
            "discriminator_losses": os.path.join(output_dir, "discriminator_losses.png"),
            "overall_losses": os.path.join(output_dir, "overall_losses.png"),
            "mse_losses": os.path.join(output_dir, "mse_losses.png"),
            "fitting_curves": [
                os.path.join(output_dir, f"G{i+1}_Train_fitting_curve.png") for i in range(3)
            ] + [
                os.path.join(output_dir, f"G{i+1}_Test_fitting_curve.png") for i in range(3)
            ],
            "density_plot": os.path.join(output_dir, "all_predictions_combined_density.png")
        }

        # 检查图片是否存在
        existing_images = {}
        for key, paths in image_paths.items():
            if isinstance(paths, list):
                existing_images[key] = [p for p in paths if os.path.exists(p)]
            else:
                if os.path.exists(paths):
                    existing_images[key] = paths

        return jsonify(existing_images)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_image/<path:image_path>")
def get_image(image_path):
    """获取图片文件"""
    try:
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route("/get_prediction_csv")
def get_prediction_csv():
    """获取预测结果CSV文件内容"""
    try:
        path = request.args.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"error": "文件不存在"}), 404

        # 读取CSV文件
        df = pd.read_csv(path)
        
        # 将数据转换为JSON格式
        data = {
            "columns": df.columns.tolist(),
            "rows": df.values.tolist()
        }
        
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def get_status():
    """获取系统状态"""
    return jsonify(check_system_resources())

@app.route("/")
def index():
    # 渲染默认参数到 HTML 模板
    return render_template('UI.html',
                         data_path=DEFAULT_PARAMS["data_path"],
                         output_dir=DEFAULT_PARAMS["output_dir"],
                         ckpt_dir=DEFAULT_PARAMS["ckpt_dir"],
                         feature_columns=DEFAULT_PARAMS["feature_columns"],
                         target_columns=DEFAULT_PARAMS["target_columns"],
                         window_sizes=DEFAULT_PARAMS["window_sizes"],
                         batch_size=DEFAULT_PARAMS["batch_size"],
                         num_epochs=DEFAULT_PARAMS["num_epochs"],
                         lr=DEFAULT_PARAMS["lr"],
                         train_split=DEFAULT_PARAMS["train_split"])

def convert_model_to_gan_components(model_path, model_desc):
    """将上传的模型转换为GAN的生成器和判别器组件"""
    try:
        # 读取模型文件
        with open(model_path, 'r', encoding='utf-8') as f:
            model_code = f.read()
            
        # 解析模型描述
        model_info = {}
        for line in model_desc.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                model_info[key.strip()] = value.strip()
                
        # 提取模型结构
        model_structure = extract_model_structure(model_code)
        
        # 生成生成器代码
        generator_code = MODEL_TEMPLATES['generator'].format(
            model_structure=model_structure
        )
        
        # 生成判别器代码
        discriminator_code = MODEL_TEMPLATES['discriminator'].format(
            model_structure=model_structure
        )
        
        # 保存转换后的模型
        model_dir = os.path.dirname(model_path)
        
        # 保存生成器
        generator_path = os.path.join(model_dir, 'generator.py')
        with open(generator_path, 'w', encoding='utf-8') as f:
            f.write(generator_code)
            
        # 保存判别器
        discriminator_path = os.path.join(model_dir, 'discriminator.py')
        with open(discriminator_path, 'w', encoding='utf-8') as f:
            f.write(discriminator_code)
            
        return {
            'generator_path': generator_path,
            'discriminator_path': discriminator_path,
            'model_info': model_info
        }
        
    except Exception as e:
        raise Exception(f"模型转换失败: {str(e)}")

def extract_model_structure(model_code):
    """从模型代码中提取模型结构"""
    try:
        # 创建临时模块
        spec = importlib.util.spec_from_loader('temp_module', loader=None)
        module = importlib.util.module_from_spec(spec)
        
        # 执行模型代码
        exec(model_code, module.__dict__)
        
        # 查找模型类
        model_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
                model_class = obj
                break
                
        if not model_class:
            raise Exception("未找到模型类定义")
            
        # 提取模型结构
        model_structure = []
        for name, module in model_class.__dict__.items():
            if isinstance(module, nn.Module):
                model_structure.append(f"self.{name} = {str(module)}")
                
        return "\n        ".join(model_structure)
        
    except Exception as e:
        raise Exception(f"提取模型结构失败: {str(e)}")

@app.route('/upload_models', methods=['POST'])
def upload_models():
    try:
        model_config = {
            'custom_models': [],
            'preset_models': []
        }
        
        # 处理每个模型位置
        for i in range(1, 4):
            model_type = request.form.get(f'model{i}Type', 'preset')
            
            if model_type == 'custom':
                model_file = request.files.get(f'model{i}')
                model_desc = request.form.get(f'model{i}Desc', '')
                
                if model_file and model_file.filename:
                    if not allowed_file(model_file.filename):
                        return jsonify({'error': f'模型 {i} 文件类型不支持'}), 400
                    
                    # 创建模型目录
                    model_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'model_{i}')
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    # 保存模型文件
                    filename = secure_filename(model_file.filename)
                    model_path = os.path.join(model_dir, filename)
                    model_file.save(model_path)
                    
                    # 保存模型描述
                    desc_path = os.path.join(model_dir, 'description.txt')
                    with open(desc_path, 'w', encoding='utf-8') as f:
                        f.write(model_desc)
                    
                    # 转换模型为GAN组件
                    try:
                        converted_models = convert_model_to_gan_components(model_path, model_desc)
                        model_config['custom_models'].append({
                            'position': i,
                            'original_path': model_path,
                            'generator_path': converted_models['generator_path'],
                            'discriminator_path': converted_models['discriminator_path'],
                            'model_info': converted_models['model_info']
                        })
                    except Exception as e:
                        return jsonify({'error': f'模型 {i} 转换失败: {str(e)}'}), 400
            else:
                # 处理预设模型
                preset_model = request.form.get(f'presetModel{i}', 'gru')
                model_config['preset_models'].append({
                    'position': i,
                    'type': preset_model
                })
        
        # 验证至少有一个模型（自定义或预设）
        if not model_config['custom_models'] and not model_config['preset_models']:
            return jsonify({'error': '请至少配置一个模型（自定义或预设）'}), 400
        
        return jsonify({
            'message': '模型配置成功',
            'model_config': model_config
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_model_structure(file_content):
    """分析模型文件结构"""
    try:
        # 解析Python代码
        tree = ast.parse(file_content)
        
        # 查找模型类定义
        model_info = {
            'class_name': None,
            'input_dim': None,
            'hidden_dim': None,
            'output_dim': None,
            'layers': [],
            'forward_params': None
        }
        
        for node in ast.walk(tree):
            # 查找类定义
            if isinstance(node, ast.ClassDef):
                model_info['class_name'] = node.name
                
                # 查找__init__方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # 分析__init__参数
                        for arg in item.args.args:
                            if arg.arg != 'self':
                                if 'input' in arg.arg.lower():
                                    model_info['input_dim'] = arg.arg
                                elif 'hidden' in arg.arg.lower():
                                    model_info['hidden_dim'] = arg.arg
                                elif 'output' in arg.arg.lower():
                                    model_info['output_dim'] = arg.arg
                        
                        # 收集网络层结构
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                        layer_name = target.attr
                                        # 排除参数变量，只收集网络层
                                        if not any(param in layer_name.lower() for param in ['dim', 'size', 'num', 'type']):
                                            model_info['layers'].append(layer_name)
                    
                    # 查找forward方法
                    elif isinstance(item, ast.FunctionDef) and item.name == 'forward':
                        # 分析forward参数
                        model_info['forward_params'] = [arg.arg for arg in item.args.args if arg.arg != 'self']
        
        # 生成结构描述
        structure = f"""模型名称: {model_info['class_name']}

输入维度: {model_info['input_dim']}
隐藏层维度: {model_info['hidden_dim']}
输出维度: {model_info['output_dim']}

网络层结构:
{chr(10).join(f'- {layer}' for layer in model_info['layers'])}

前向传播参数: {', '.join(model_info['forward_params'])}"""

        return structure
    except Exception as e:
        return f"模型结构分析失败: {str(e)}"

@app.route('/analyze_model', methods=['POST'])
def analyze_model():
    """分析上传的模型文件结构"""
    if 'model_file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    try:
        # 读取文件内容
        file_content = file.read().decode('utf-8')
        
        # 分析模型结构
        structure = analyze_model_structure(file_content)
        
        return jsonify({
            'structure': structure
        })
    except Exception as e:
        return jsonify({'error': f'模型分析失败: {str(e)}'}), 500

if __name__ == "__main__":
    # 检查系统资源
    resources = check_system_resources()
    print(f"系统初始状态: {resources}")
    
    # 启动服务器，开启debug模式
    app.run(debug=True, host="127.0.0.1", port=7000, threaded=True)