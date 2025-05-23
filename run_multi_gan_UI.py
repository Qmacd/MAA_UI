import argparse
from time_series_maa import MAA_time_series
import pandas as pd
import os
import models
from utils.logger import setup_experiment_logging
import logging
import sys


def parse_feature_ranges(feature_str):
    """将'6-10,15-17'这样的字符串转换为[6,7,8,9,10,15,16,17]这样的列表"""
    try:
        # 如果输入是列表，直接返回
        if isinstance(feature_str, list):
            return feature_str
            
        # 如果是字符串，则解析
        ranges = feature_str.split(',')
        result = []
        for r in ranges:
            if '-' in r:
                start, end = map(int, r.split('-'))
                result.extend(range(start, end + 1))
            else:
                result.append(int(r))
        return result
    except Exception as e:
        logging.error(f"解析特征范围时出错: {str(e)}")
        raise


def run_experiments(args):
    """运行实验的主函数"""
    try:
        # 创建保存结果的CSV文件
        results_file = os.path.join(args.output_dir, "maa_results.csv")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        gca = MAA_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
                            args.generators, args.discriminators,
                            args.ckpt_dir, args.output_dir,
                            args.window_sizes,
                            ckpt_path=args.ckpt_path,
                            initial_learning_rate=args.lr,
                            train_split=args.train_split,
                            do_distill_epochs=args.distill_epochs,
                            cross_finetune_epochs=args.cross_finetune_epochs,
                            device=args.device,
                            seed=args.random_seed)

        results_list = []
        for target in args.target_columns:
            # 直接使用传入的特征组，并复制一份以避免修改原始数据
            target_feature_columns = [group.copy() for group in args.feature_groups]
            
            # 添加目标列到每个特征组的末尾
            for feature_group in target_feature_columns:
                feature_group.extend(target)
            
            print("Using features:", target_feature_columns)  # 添加特征使用日志

            gca.process_data(args.data_path, args.start_timestamp, args.end_timestamp, target,
                            target_feature_columns, args.log_diff)
            gca.init_dataloader()
            gca.init_model(args.num_classes)

            logger = setup_experiment_logging(args.output_dir, vars(args))

            if args.mode == "train":
                results = gca.train(logger)
                # gca.pred()
            elif args.mode == "pred":
                results = gca.pred()

            filename = os.path.basename(args.data_path)
            product_name = filename.replace('_processed.csv', '')
            
            # 将结果保存到CSV
            result_row = {
                "market": product_name,
                "feature_columns": args.feature_groups,
                "target_columns": target,
                "train_mse": results["train_mse"],
                "train_mae": results["train_mae"],
                "train_rmse": results["train_rmse"],
                "train_mape": results["train_mape"],
                "train_mse_per_target": results["train_mse_per_target"],
                "train_acc": results["train_acc"],
                "test_mse": results["test_mse"],
                "test_mae": results["test_mae"],
                "test_rmse": results["test_rmse"],
                "test_mape": results["test_mape"],
                "test_mse_per_target": results["test_mse_per_target"],
                "test_acc": results["test_acc"],
            }
            results_list.append(result_row)
            
            # 保存到CSV文件
            df = pd.DataFrame([result_row])
            df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)
        
        return results_list
    except Exception as e:
        logging.error(f"运行实验时出错: {str(e)}")
        raise

# 确保这些函数在模块级别可用
__all__ = ['run_experiments', 'parse_feature_ranges']


