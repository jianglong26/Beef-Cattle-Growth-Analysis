#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
YOLOv11牛只检测模型训练脚本
实现迁移学习训练流程
"""

# 必须在导入任何其他库之前设置matplotlib后端（解决无头环境问题）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器训练

import os
import sys
import yaml
import time
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import wandb

# # 添加项目根目录到Python路径
# current_dir = Path(__file__).parent
# project_root = current_dir.parent.parent
# print(f"Project root: {project_root}")
# sys.path.append(str(project_root))

# 导入项目模块
from configs.config_s import ProjectConfig
# Ultralytics imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


class YOLOv11Trainer:
    """YOLOv11训练器"""
    
    def __init__(self, config_path=None):
        """初始化训练器"""
        # 加载配置
        self.config = ProjectConfig()
        self.device = self.config.TRAINING_CONFIG["device"]
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化训练状态
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.model = None
        
        # 创建保存目录
        self.create_directories()
        
        # # 初始化工具类
        # self.early_stopping = EarlyStopping(
        #     patience=self.config.TRAINING_CONFIG["patience"],
        #     min_delta=0.001
        # )
        
        # self.validator = ModelValidator()
        # self.evaluator = ModelEvaluator()
        
        self.logger.info(f"Trainer initialized with device: {self.device}")
    
    def setup_logging(self):
        """设置日志"""
        log_level = self.config.LOGGING_CONFIG.get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.PATHS["logs_dir"] / "train.log"),
                logging.StreamHandler()
            ]
        )
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.PATHS["output_dir"],
            # self.config.PATHS["checkpoint_dir"],
            self.config.PATHS["logs_dir"],
            # self.config.PATHS["tensorboard_dir"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset_config(self):
        """使用现有的数据集配置文件"""
        self.logger.info("Using existing dataset configuration...")
        
        # 直接使用配置文件中定义的路径
        dataset_yaml_path = self.config.DATA_CONFIG_YAML
        
        if not dataset_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml_path}")
        
        # 验证YAML文件内容
        try:
            with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
                dataset_config = yaml.safe_load(f)
            
            # 检查必要的字段
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in dataset_config:
                    raise ValueError(f"Missing required field '{field}' in dataset YAML")
            
            # 🔥 修改：使用DATA_ROOT作为数据集根目录，而不是从YAML解析
            dataset_root = self.config.PATHS["data_root"]
            
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")
            
            # 更新YAML中的路径为绝对路径
            dataset_config['path'] = str(dataset_root)
            
            # 验证train和val路径（相对于dataset_root）
            train_path = dataset_root / dataset_config['train']
            val_path = dataset_root / dataset_config['val']
            
            if not train_path.exists():
                self.logger.warning(f"Train images directory not found: {train_path}")
            if not val_path.exists():
                self.logger.warning(f"Val images directory not found: {val_path}")
            
            # 创建临时YAML文件确保路径正确
            temp_yaml_path = self.config.PATHS["output_dir"] / "temp_dataset.yaml"
            with open(temp_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Using dataset config: {dataset_yaml_path}")
            self.logger.info(f"Dataset root: {dataset_root}")
            self.logger.info(f"Train images: {train_path}")
            self.logger.info(f"Val images: {val_path}")
            
            return str(temp_yaml_path)
            
        except Exception as e:
            raise RuntimeError(f"Error processing dataset YAML file {dataset_yaml_path}: {e}")
        
    def load_model(self):
        """加载YOLOv11模型"""
        model_name = self.config.MODEL_CONFIG["model_name"]
        # num_classes = self.config.DATASET_CONFIG["num_classes"]
        
        try:
            # 检查预训练模型是否存在
            pretrained_path = self.config.MODEL_CONFIG.get("pretrained_path")
            if pretrained_path and Path(pretrained_path).exists():
                self.logger.info(f"Loading pretrained model from: {pretrained_path}")
                self.model = YOLO(pretrained_path)
            else:
                # 使用官方预训练模型
                self.logger.info(f"Loading official pretrained model: {model_name}")
                self.model = YOLO(f'{model_name}')
            
            # 模型信息
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def setup_training_args(self, dataset_yaml):
        """设置训练参数"""
        # 读取YAML文件获取类别数量
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        num_classes = dataset_config.get('nc', 1)

        training_args = {
            # 数据相关
            'data': dataset_yaml,
            'imgsz': self.config.MODEL_CONFIG["input_size"],
            'batch': self.config.TRAINING_CONFIG.get("batch_size", 16),
            'workers': self.config.TRAINING_CONFIG["workers"],
            
            # 训练参数
            'epochs': self.config.TRAINING_CONFIG.get("epochs", 100),
            'lr0': self.config.TRAINING_CONFIG.get("learning_rate", 5e-3),
            'lrf': self.config.TRAINING_CONFIG.get("final_lr_ratio", 0.01),
            'momentum': self.config.TRAINING_CONFIG.get("momentum", 0.937),
            'weight_decay': self.config.TRAINING_CONFIG.get("weight_decay", 0.0005),
            'warmup_epochs': self.config.TRAINING_CONFIG.get("warmup_epochs", 3),
            'warmup_momentum': self.config.TRAINING_CONFIG.get("warmup_momentum", 0.8),
            'warmup_bias_lr': self.config.TRAINING_CONFIG.get("warmup_bias_lr", 0.1),
            
            # 优化器和调度器
            'optimizer': self.config.TRAINING_CONFIG.get("optimizer", "auto"),
            'cos_lr': self.config.TRAINING_CONFIG.get("cos_lr", True),
            
            # 模型保存 - 简化输出目录结构
            'project': str(self.config.PATHS["output_dir"]),
            'name': f'yolov11s_cattle_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'save': True,
            
            # 验证相关
            'val': True,
            'patience': self.config.TRAINING_CONFIG["patience"],
            
            # 设备
            'device': self.device,
            
            # 其他
            'verbose': True,
            'seed': self.config.TRAINING_CONFIG.get("seed", 42),
            'deterministic': True,
            'single_cls': num_classes == 1,  # 单类别训练
            
            # 数据增强 (YOLOv11内置)
            'hsv_h': self.config.AUGMENTATION_CONFIG.get("hsv_h", 0.015),
            'hsv_s': self.config.AUGMENTATION_CONFIG.get("hsv_s", 0.7),
            'hsv_v': self.config.AUGMENTATION_CONFIG.get("hsv_v", 0.4),
            'degrees': self.config.AUGMENTATION_CONFIG.get("degrees", 0.0),
            'translate': self.config.AUGMENTATION_CONFIG.get("translate", 0.1),
            'scale': self.config.AUGMENTATION_CONFIG.get("scale", 0.5),
            'shear': self.config.AUGMENTATION_CONFIG.get("shear", 0.0),
            'perspective': self.config.AUGMENTATION_CONFIG.get("perspective", 0.0),
            'flipud': self.config.AUGMENTATION_CONFIG.get("flipud", 0.0),
            'fliplr': self.config.AUGMENTATION_CONFIG.get("fliplr", 0.5),
            'mosaic': self.config.AUGMENTATION_CONFIG.get("mosaic", 1.0),
            'mixup': self.config.AUGMENTATION_CONFIG.get("mixup", 0.0),
            'copy_paste': self.config.AUGMENTATION_CONFIG.get("copy_paste", 0.0),
        }
        
        return training_args
    
    def setup_wandb(self):
        """设置Weights & Biases监控"""
        if self.config.LOGGING_CONFIG.get("use_wandb", False):
            try:
                wandb_config = self.config.LOGGING_CONFIG.get("wandb", {})
                wandb.init(
                    project=wandb_config.get("project", "yolov11-cattle-detection"),
                    name=f"yolov11_cattle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model": self.config.MODEL_CONFIG["model_name"],
                        "epochs": self.config.TRAINING_CONFIG["epochs"],
                        "batch_size": self.config.TRAINING_CONFIG["batch_size"],
                        "learning_rate": self.config.TRAINING_CONFIG["learning_rate"],
                        "input_size": self.config.MODEL_CONFIG["input_size"],
                        "num_classes": self.config.DATASET_CONFIG["num_classes"],
                    }
                )
                self.logger.info("Weights & Biases initialized")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                return False
        return False
    
    def train(self):
        """主训练流程"""
        self.logger.info("="*50)
        self.logger.info("Starting YOLOv11 Cattle Detection Training")
        self.logger.info("="*50)
        
        try:
            # 1. 准备数据集配置
            dataset_yaml = self.prepare_dataset_config()
            
            # 2. 加载模型
            if not self.load_model():
                return False
            
            # 3. 设置训练参数
            training_args = self.setup_training_args(dataset_yaml)
            
            # 4. 设置监控
            use_wandb = self.setup_wandb()
            
            # 5. 打印训练配置
            self.print_training_config(training_args)
            
            # 6. 开始训练
            self.logger.info("Starting training...")
            start_time = time.time()
            
            # 使用Ultralytics的train方法
            results = self.model.train(**training_args)
            
            # 7. 训练完成
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # 8. 保存最终模型
            self.save_final_model(results)
            
            # 9. 评估模型
            self.evaluate_model(results)
            
            # 10. 清理资源
            if use_wandb:
                wandb.finish()
            
            self.logger.info("Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def print_training_config(self, training_args):
        """打印训练配置"""
        self.logger.info("\nTraining Configuration:")
        self.logger.info("-" * 30)
        for key, value in training_args.items():
            if isinstance(value, (str, int, float, bool)):
                self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 30)
    
    def save_final_model(self, results):
        """保存最终模型到默认weights目录"""
        try:
            # YOLOv11会自动在训练目录下创建weights文件夹并保存best.pt和last.pt
            # 这里只需要记录模型位置即可
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            last_model_path = results.save_dir / 'weights' / 'last.pt'
            
            if best_model_path.exists():
                self.logger.info(f"Best model saved to: {best_model_path}")
            
            if last_model_path.exists():
                self.logger.info(f"Last model saved to: {last_model_path}")
                
            # 记录训练结果目录
            self.logger.info(f"Training results saved to: {results.save_dir}")
                
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}")
    
    def evaluate_model(self, results):
        """评估训练结果"""
        try:
            self.logger.info("\nTraining Results Summary:")
            self.logger.info("-" * 30)
            
            # 打印训练结果
            if hasattr(results, 'results_dict'):
                for key, value in results.results_dict.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"{key}: {value:.4f}")
            
            # 获取最佳指标
            metrics_file = results.save_dir / 'results.csv'
            if metrics_file.exists():
                import pandas as pd
                df = pd.read_csv(metrics_file)
                if len(df) > 0:
                    best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
                    self.logger.info(f"Best mAP@0.5: {best_row['metrics/mAP50(B)']:.4f}")
                    self.logger.info(f"Best mAP@0.5:0.95: {best_row['metrics/mAP50-95(B)']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
    
    def resume_training(self, checkpoint_path):
        """恢复训练"""
        try:
            self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            self.model = YOLO(checkpoint_path)
            return True
        except Exception as e:
            self.logger.error(f"Error resuming training: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11 Cattle Detection Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        # 创建训练器
        trainer = YOLOv11Trainer(args.config)
        
        # 更新配置（如果提供命令行参数）
        if args.device != 'auto':
            trainer.config.MODEL_CONFIG["device"] = args.device
        if args.batch_size:
            trainer.config.TRAINING_CONFIG["batch_size"] = args.batch_size
        if args.epochs:
            trainer.config.TRAINING_CONFIG["epochs"] = args.epochs
        if args.lr:
            trainer.config.TRAINING_CONFIG["learning_rate"] = args.lr
        
        # 恢复训练或开始新训练
        if args.resume:
            if trainer.resume_training(args.resume):
                success = trainer.train()
            else:
                success = False
        else:
            success = trainer.train()
        
        if success:
            print("Training completed successfully!")
            return 0
        else:
            print("Training failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
