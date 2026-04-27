#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
YOLOv11牛只检测项目配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
# print(f"Project root: {PROJECT_ROOT}")
DATA_ROOT = PROJECT_ROOT/ "data" / "yolodet_train_dataset"  # 修改这里
# print(f"Data root: {DATA_ROOT}")
YOLOV11_DIR = Path(__file__).parent.parent  # 修改这里，获取目录而不是文件
# print(f"YOLOV11 directory: {YOLOV11_DIR}")

# 预训练模型目录
PRETRAINED_MODELS_DIR = YOLOV11_DIR / "pretrained_models"
PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
# 数据集配置
# 数据集配置 - 使用你的原始格式
# 数据配置文件路径（唯一数据源）
DATA_CONFIG_YAML = DATA_ROOT/"dataset.yaml"

# 模型配置
MODEL_CONFIG = {
    "model_name": "yolo11s.pt",  # 可选: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    "pretrained_path": PRETRAINED_MODELS_DIR / "yolo11s.pt",  # 预训练模型完整路径
    "input_size": 832,
    "conf_threshold": 0.25,
    "iou_threshold": 0.65,
    "max_det": 1000,
}

# 训练配置
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 16,  # ↓ 减小batch增加更新频率（24→16）适配500张数据集
    "learning_rate": 0.0003,  # ↓↓ 进一步降低学习率（0.0005→0.0003）500张数据更需保守
    "momentum": 0.937,
    "weight_decay": 0.001,  # ↓ 降低权重衰减（5e-3→0.001）避免过度正则化
    "warmup_epochs": 20,  # ↑ 进一步延长warmup（15→20）数据少需更平稳启动
    "warmup_momentum": 0.5,  # ↓ 降低初始动量（0.8→0.5）
    "warmup_bias_lr": 0.05,  # ↓ 降低bias学习率（0.1→0.05）
    "box_loss_gain": 0.1,
    "cls_loss_gain": 0.5,
    "dfl_loss_gain": 1.0,  # 恢复原值
    "patience": 50,  # ↑ 增加patience，避免过早停止（30→50）
    "save_period": -1,
    "workers": 8,
    "device": "1",
    "project": "runs/train",
    "name": "cattle_detection",
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "AdamW",  # ✓ 使用AdamW优化器，比SGD更稳定
    "verbose": True,
    "seed": 42,  # ✓ 使用固定随机种子确保可复现
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": True,
    "close_mosaic": 20,  # ↓ 更早关闭mosaic，减少对俯瞰视角的破坏（50→20）
    "resume": False,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "freeze": None,
    "multi_scale": True,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.15,  # ↑ 增加dropout（0.1→0.15）防止小数据集过拟合
    "val": True,
    "plots": True,
    "save": True,
    "save_txt": False,
    "save_hybrid": False,
    "save_crop": False,
    "save_json": False,
    "conf": None,
    "iou": 0.8,  # 恢复原值
    "max_det": 300,
    "half": False,
    "dnn": False,
    "augment": False,
}

# 数据增强配置 - 针对俯瞰牧场场景优化
# 原则：简单场景用简单增强，保持俯瞰视角的一致性
AUGMENTATION_CONFIG = {
    # 颜色增强（适度，牧场光照变化不大）
    "hsv_h": 0.01,    # ↓ 降低色调变化（0.015→0.01）草地颜色相对固定
    "hsv_s": 0.4,     # ↓ 降低饱和度变化（0.5→0.4）
    "hsv_v": 0.3,     # ✓ 保持明度变化（模拟不同时间光照）
    
    # 几何增强（进一步减小，500张数据需更保守）
    "degrees": 2.0,   # ↓ 进一步降低旋转（3.0→2.0）
    "translate": 0.05,# ↓ 进一步降低平移（0.08→0.05）
    "scale": 0.15,    # ↓ 进一步降低缩放（0.2→0.15）
    "shear": 0.0,     # ✕ 关闭剪切（2→0）俯瞰视角不需要剪切变换
    "perspective": 0.0, # ✓ 保持关闭（正确）
    
    # 翻转增强（合理）
    "flipud": 0.0,    # ✓ 保持关闭（牛不会倒立）
    "fliplr": 0.5,    # ✓ 保持左右翻转（合理）
    
    # 高级增强（进一步降低强度，500张数据需更谨慎）
    "mosaic": 0.3,    # ↓ 进一步降低mosaic（0.5→0.3）小数据集避免过度增强
    "mixup": 0.0,     # ✕ 关闭mixup（0.1→0）不适合俯瞰场景
    "copy_paste": 0.0,# ✕ 关闭copy_paste（0.1→0）牛的姿态和位置有特定规律
}


# 验证配置
VALIDATION_CONFIG = {
    "val_split": 0.2,
    "shuffle": True,
    "save_txt": False,
    "save_hybrid": False,
    "save_conf": False,
    "save_json": True,
    "conf_thres": 0.001,
    "iou_thres": 0.8,
    "max_det": 300,
    "task": "val",
    "device": "",
    "workers": 8,
    "single_cls": False,
    "augment": False,
    "verbose": False,
    "save_txt": False,
    "save_hybrid": False,
    "save_conf": False,
    "save_json": True,
    "project": "runs/val",
    "name": "exp",
    "exist_ok": False,
    "half": True,
    "dnn": False,
}

# 推理配置
INFERENCE_CONFIG = {
    "conf_thres": 0.25,
    "iou_thres": 0.6,
    "max_det": 1000,
    "device": "",
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "nosave": False,
    "classes": None,
    "agnostic_nms": False,
    "augment": False,
    "visualize": False,
    "update": False,
    "project": "runs/detect",
    "name": "exp",
    "exist_ok": False,
    "line_thickness": 3,
    "hide_labels": False,
    "hide_conf": False,
    "half": False,
    "dnn": False,
    "vid_stride": 1,
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_dir": "logs",
    "max_files": 10,
}

# 路径配置
PATHS = {
    "project_root": PROJECT_ROOT,
    "data_root": DATA_ROOT,
    "pretrained_models_dir": PRETRAINED_MODELS_DIR,
    "logs_dir": YOLOV11_DIR / "logs",
    "output_dir": YOLOV11_DIR / "output",
}


# GPU配置检查
def get_device():
    """获取可用的设备"""
    import torch
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"

# 更新设备配置
TRAINING_CONFIG["device"] = get_device() if TRAINING_CONFIG["device"] == "0" else TRAINING_CONFIG["device"]

# 环境变量设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "cuda" in TRAINING_CONFIG["device"] else ""

class ProjectConfig:
    """项目配置类"""
    
    def __init__(self):
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAINING_CONFIG = TRAINING_CONFIG
        self.AUGMENTATION_CONFIG = AUGMENTATION_CONFIG
        self.PATHS = PATHS
        self.LOGGING_CONFIG = LOG_CONFIG
        self.device = TRAINING_CONFIG
        self.DATA_CONFIG_YAML = DATA_CONFIG_YAML
        
        # 创建必要目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        for path in self.PATHS.values():
            if isinstance(path, Path) and path.suffix == "":  # 只创建目录，不创建文件
                path.mkdir(parents=True, exist_ok=True)


