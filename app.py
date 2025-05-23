# app.py
import os
import sys
from flask import Flask, request, render_template, jsonify, Response
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制为 16MB

# 日志队列（用于实时日志输出）
log_queue = queue.Queue()

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
    "device": [0],
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
        data = request.json
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
        
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
            window_sizes = data.get("window_sizes", [])
            if isinstance(window_sizes, list):
                window_sizes = [int(x) for x in window_sizes]
            else:
                raise ValueError("window_sizes 必须是列表")
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
            "generators": ["transformer", "transformer", "transformer"],
            "discriminators": None,
            "distill_epochs": 1,
            "cross_finetune_epochs": 5,
            "device": [0],
            "num_epochs": int(data["num_epochs"]),
            "lr": float(data["lr"]),
            "batch_size": int(data["batch_size"]),
            "train_split": float(data["train_split"]),
            "random_seed": 3407,
            "amp_dtype": "none",
            "mode": "train",
            "ckpt_path": "latest",
            "log_diff": False
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

        # 构建预测参数
        params = {
            "data_path": data["input_path"],
            "output_dir": os.path.dirname(data["output_path"]),
            "ckpt_dir": data.get("ckpt_path", "out_put/ckpt"),  # 使用默认的检查点目录
            "ckpt_path": "latest",  # 默认使用最新的检查点
            "N_pairs": 3,
            "batch_size": 64,
            "window_sizes": [5, 10, 15],
            "num_classes": 3,
            "feature_groups": data.get("feature_groups", []),  # 从请求中获取特征组
            "target_columns": data.get("target_columns", []),  # 从请求中获取目标列
            "device": [0],
            "random_seed": 3407,
            "log_diff": False,
            "start_timestamp": 31,
            "end_timestamp": -1,
            "generators": ["transformer", "transformer", "transformer"],
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
                
                log_queue.put(f"预测完成，结果已保存到: {data['output_path']}")
                
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

if __name__ == "__main__":
    # 检查系统资源
    resources = check_system_resources()
    print(f"系统初始状态: {resources}")
    
    # 启动服务器
    app.run(debug=False, host="127.0.0.1", port=8000, threaded=True)