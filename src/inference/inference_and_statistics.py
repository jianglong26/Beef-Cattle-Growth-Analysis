#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_and_statistics.py
对未标注数据集进行YOLOv11推理并生成统计报告
推理置信度阈值：0.8
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List
import torch
from collections import defaultdict

# 导入YOLO
from ultralytics import YOLO


class InferenceStatistics:
    """推理统计器 - 对未标注数据进行推理并统计"""
    
    def __init__(self, model_path: str, data_root: str, output_dir: str, 
                 conf_threshold: float = 0.8, device: str = 'auto'):
        """初始化推理统计器
        
        Args:
            model_path: 模型checkpoint路径
            data_root: 数据集根目录 (如 data/15m_500_split)
            output_dir: 输出目录
            conf_threshold: 置信度阈值，默认0.8
            device: 设备选择
        """
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device.startswith('cuda'):
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 数据集根目录
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {data_root}")
        
        # 创建输出目录 - 直接在数据集根目录下
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.data_root / f"inference_statistics_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 置信度阈值
        self.conf_threshold = conf_threshold
        self.logger.info(f"Confidence threshold set to: {conf_threshold}")
        
        # 加载模型
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.logger.info(f"Loaded model from: {model_path}")
        
        # 类别名称 (根据模型训练时的定义)
        # 根据 temp_dataset.yaml: 0=laying, 1=standing
        self.class_names = {
            0: 'laying',
            1: 'standing'
        }
        
        # 统计结果存储
        self.statistics = {}
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "inference.log"),
                logging.StreamHandler()
            ]
        )
    
    def get_subdirectories(self) -> List[Path]:
        """获取数据集根目录下的所有子目录 (train, val, test)
        
        Returns:
            List[Path]: 子目录列表
        """
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        subdirs = sorted(subdirs)
        self.logger.info(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
        return subdirs
    
    def get_image_files(self, images_dir: Path) -> List[Path]:
        """获取images目录中的所有图像文件
        
        Args:
            images_dir: images目录路径
            
        Returns:
            List[Path]: 图像文件路径列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                           '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(images_dir.glob(f'*{ext}'))
        
        # 排序确保顺序一致
        image_paths = sorted(image_paths)
        
        return image_paths
    
    def inference_on_directory(self, subdir: Path) -> Dict:
        """对单个子目录进行推理并统计
        
        Args:
            subdir: 子目录路径 (如 train, val, test)
            
        Returns:
            Dict: 统计结果
        """
        subdir_name = subdir.name
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing directory: {subdir_name}")
        self.logger.info(f"{'='*60}")
        
        # 查找images目录
        images_dir = subdir / 'images'
        if not images_dir.exists():
            self.logger.warning(f"Images directory not found in {subdir_name}")
            return None
        
        # 获取所有图像文件
        image_paths = self.get_image_files(images_dir)
        
        if len(image_paths) == 0:
            self.logger.warning(f"No images found in {subdir_name}")
            return None
        
        self.logger.info(f"Found {len(image_paths)} images")
        
        # 统计变量
        total_images = len(image_paths)
        total_instances = 0
        standing_instances = 0
        laying_instances = 0
        
        # 对每张图像进行推理
        self.logger.info(f"Starting inference with confidence threshold = {self.conf_threshold}...")
        
        for idx, img_path in enumerate(image_paths, 1):
            if idx % 50 == 0 or idx == total_images:
                self.logger.info(f"Processing: {idx}/{total_images}")
            
            # 使用YOLO进行推理
            results = self.model.predict(
                source=str(img_path),
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )
            
            # 提取检测结果
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 统计实例数（0=laying, 1=standing）
                        total_instances += 1
                        
                        if class_id == 0:  # laying
                            laying_instances += 1
                        elif class_id == 1:  # standing
                            standing_instances += 1
        
        # 整理统计结果
        stats = {
            'directory': subdir_name,
            'total_images': total_images,
            'total_instances': total_instances,
            'standing_instances': standing_instances,
            'laying_instances': laying_instances,
            'avg_instances_per_image': total_instances / total_images if total_images > 0 else 0
        }
        
        self.logger.info(f"\nStatistics for {subdir_name}:")
        self.logger.info(f"  Total Images: {total_images}")
        self.logger.info(f"  Total Instances: {total_instances}")
        self.logger.info(f"  Standing Instances: {standing_instances}")
        self.logger.info(f"  Laying Instances: {laying_instances}")
        self.logger.info(f"  Avg Instances per Image: {stats['avg_instances_per_image']:.2f}")
        
        return stats
    
    def run_inference(self):
        """运行推理并生成统计报告"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Starting Inference and Statistics Generation")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Data Root: {self.data_root}")
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Confidence Threshold: {self.conf_threshold}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        
        # 获取所有子目录
        subdirs = self.get_subdirectories()
        
        if len(subdirs) == 0:
            self.logger.error("No subdirectories found!")
            return
        
        # 对每个子目录进行推理
        all_statistics = []
        
        for subdir in subdirs:
            stats = self.inference_on_directory(subdir)
            if stats:
                all_statistics.append(stats)
                self.statistics[subdir.name] = stats
        
        # 生成总体统计
        self.generate_summary_report(all_statistics)
        
        # 保存统计结果为JSON
        self.save_statistics_json()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Inference and Statistics Generation Completed!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def generate_summary_report(self, all_statistics: List[Dict]):
        """生成汇总统计报告
        
        Args:
            all_statistics: 所有子目录的统计结果列表
        """
        report_path = self.output_dir / "statistics_report.txt"
        
        # 计算总计
        total_images_all = sum(s['total_images'] for s in all_statistics)
        total_instances_all = sum(s['total_instances'] for s in all_statistics)
        standing_instances_all = sum(s['standing_instances'] for s in all_statistics)
        laying_instances_all = sum(s['laying_instances'] for s in all_statistics)
        
        # 生成报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("                  INFERENCE STATISTICS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data Root: {self.data_root}\n")
            f.write(f"Model: {self.model_path.name}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"Inference Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # 每个子目录的详细统计
            f.write("DETAILED STATISTICS BY DIRECTORY:\n")
            f.write("-"*80 + "\n\n")
            
            for stats in all_statistics:
                f.write(f"Directory: {stats['directory']}\n")
                f.write(f"  Total Images:          {stats['total_images']:>6}\n")
                f.write(f"  Total Instances:       {stats['total_instances']:>6}\n")
                f.write(f"  Standing Instances:    {stats['standing_instances']:>6}\n")
                f.write(f"  Laying Instances:      {stats['laying_instances']:>6}\n")
                f.write(f"  Avg Instances/Image:   {stats['avg_instances_per_image']:>6.2f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n\n")
            
            # 总计
            f.write("OVERALL SUMMARY:\n")
            f.write("-"*80 + "\n\n")
            f.write(f"  Total Images (All Directories):     {total_images_all:>6}\n")
            f.write(f"  Total Instances (All Directories):  {total_instances_all:>6}\n")
            f.write(f"  Standing Instances (All):           {standing_instances_all:>6}\n")
            f.write(f"  Laying Instances (All):             {laying_instances_all:>6}\n")
            
            if total_instances_all > 0:
                standing_ratio = standing_instances_all / total_instances_all * 100
                laying_ratio = laying_instances_all / total_instances_all * 100
                f.write(f"\n")
                f.write(f"  Standing Ratio:                     {standing_ratio:>5.1f}%\n")
                f.write(f"  Laying Ratio:                       {laying_ratio:>5.1f}%\n")
            
            if total_images_all > 0:
                avg_instances = total_instances_all / total_images_all
                f.write(f"  Average Instances per Image:        {avg_instances:>6.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            
            # 表格形式的汇总
            f.write("\n\nTABLE FORMAT SUMMARY:\n")
            f.write("-"*80 + "\n\n")
            
            # 表头
            f.write(f"{'Directory':<20} {'Images':<10} {'Instances':<12} {'Standing':<12} {'Laying':<12}\n")
            f.write("-"*80 + "\n")
            
            # 数据行
            for stats in all_statistics:
                f.write(f"{stats['directory']:<20} "
                       f"{stats['total_images']:<10} "
                       f"{stats['total_instances']:<12} "
                       f"{stats['standing_instances']:<12} "
                       f"{stats['laying_instances']:<12}\n")
            
            # 总计行
            f.write("-"*80 + "\n")
            f.write(f"{'Total':<20} "
                   f"{total_images_all:<10} "
                   f"{total_instances_all:<12} "
                   f"{standing_instances_all:<12} "
                   f"{laying_instances_all:<12}\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"\nStatistics report saved to: {report_path}")
        
        # 打印到控制台
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    
    def save_statistics_json(self):
        """保存统计结果为JSON格式"""
        json_path = self.output_dir / "statistics.json"
        
        output_data = {
            'metadata': {
                'data_root': str(self.data_root),
                'model_path': str(self.model_path),
                'confidence_threshold': self.conf_threshold,
                'inference_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'statistics': self.statistics
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Statistics JSON saved to: {json_path}")


def main():
    """主函数 - 参数直接在代码中配置"""
    
    # ==================== 配置参数 ====================
    # 模型路径
    MODEL_PATH = 'YOLOV11/output/yolov11s_cattle_20260109_205959_800/weights/best.pt'
    
    # 数据集根目录
    DATA_ROOT = 'data/15m_500_split'
    
    # 置信度阈值
    CONF_THRESHOLD = 0.8
    
    # 设备选择 ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1')
    DEVICE = 'auto'
    
    # 注意：输出目录会自动创建在数据集根目录下
    # 格式: {DATA_ROOT}/inference_statistics_{timestamp}
    # =================================================
    
    # 创建推理统计器并运行
    inferencer = InferenceStatistics(
        model_path=MODEL_PATH,
        data_root=DATA_ROOT,
        output_dir=DATA_ROOT,  # 输出到数据集目录
        conf_threshold=CONF_THRESHOLD,
        device=DEVICE
    )
    
    inferencer.run_inference()


if __name__ == '__main__':
    main()
