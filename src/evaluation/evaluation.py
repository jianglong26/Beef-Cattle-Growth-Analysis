#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_best_model.py
YOLOv11最佳模型测试集评估脚本
用于论文实验：在held-out测试集上评估训练过程中选择的最佳checkpoint
"""

# 设置matplotlib后端（服务器环境）
import matplotlib
matplotlib.use('Agg')

import os
import sys
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from collections import defaultdict

# 导入YOLO
from ultralytics import YOLO

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BestModelEvaluator:
    """最佳模型评估器 - 在测试集上评估训练选择的最佳checkpoint"""
    
    def __init__(self, model_path: str, output_dir: str, device: str = 'auto'):
        """初始化评估器
        
        Args:
            model_path: 最佳模型checkpoint路径 (基于验证集选择)
            output_dir: 输出目录
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

        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.logger.info(f"Loaded best model checkpoint from: {model_path}")
        
        # 结果存储
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': {},
            'inference_times': []
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "test_evaluation.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_test_dataset(self, dataset_yaml_path: str) -> Tuple[List[Path], List[Path]]:
        """加载测试数据集（held-out test set）
        
        Args:
            dataset_yaml_path: 数据集YAML配置文件路径
            
        Returns:
            Tuple[List[Path], List[Path]]: (图像路径列表, 标签路径列表)
        """
        with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_root = Path(dataset_config['path'])
        test_images_dir = dataset_root / 'test' / 'images'
        test_labels_dir = dataset_root / 'test' / 'labels'
        
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(test_images_dir.glob(f'*{ext}'))
            image_paths.extend(test_images_dir.glob(f'*{ext.upper()}'))
        
        # 获取对应的标签文件
        label_paths = []
        for img_path in image_paths:
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            label_paths.append(label_path if label_path.exists() else None)
        
        self.logger.info(f"Loaded {len(image_paths)} test images from held-out test set")
        self.logger.info(f"Found {sum(1 for lp in label_paths if lp is not None)} corresponding labels")
        
        return image_paths, label_paths
    
    def parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        """解析YOLO格式标签文件
        
        Args:
            label_path: 标签文件路径
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            List[Dict]: 标注信息列表
        """
        annotations = []
        
        if label_path is None or not label_path.exists():
            return annotations
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 转换为 [x1, y1, x2, y2]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 1.0
                })
        
        return annotations
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(self, predictions: List[Dict], ground_truths: List[Dict], 
                     iou_threshold: float = 0.5) -> Tuple[float, float, float]:
        """计算单个IoU阈值下的AP（Average Precision）
        
        Args:
            predictions: 所有预测结果（已按置信度排序）
            ground_truths: 所有真实标签
            iou_threshold: IoU阈值
            
        Returns:
            Tuple[float, float, float]: (precision, recall, AP)
        """
        if len(predictions) == 0 or len(ground_truths) == 0:
            return 0.0, 0.0, 0.0
        
        # 按置信度降序排列预测
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        matched_gt = set()
        
        # 匹配预测和真实框
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                # 检查类别是否匹配
                if pred['class_id'] != gt['class_id']:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算precision和recall
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP (使用11点插值法)
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # 使precisions单调递减
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # 计算曲线下面积
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        # 计算最终的precision和recall
        final_precision = precisions[-2] if len(precisions) > 2 else 0.0
        final_recall = recalls[-2] if len(recalls) > 2 else 0.0
        
        return final_precision, final_recall, ap
    
    def calculate_map_at_iou_range(self, all_predictions: List[List[Dict]], 
                                     all_ground_truths: List[List[Dict]]) -> Dict:
        """计算mAP@0.5:0.95 (COCO-style)
        
        Args:
            all_predictions: 所有图像的预测结果
            all_ground_truths: 所有图像的真实标签
            
        Returns:
            Dict: 包含各种mAP指标
        """
        # 展平所有预测和标签
        flat_predictions = []
        flat_ground_truths = []
        
        for preds, gts in zip(all_predictions, all_ground_truths):
            flat_predictions.extend(preds)
            flat_ground_truths.extend(gts)
        
        # 计算不同IoU阈值下的AP
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # 0.5, 0.55, 0.6, ..., 0.95
        aps = []
        
        results = {}
        
        for iou_thresh in iou_thresholds:
            precision, recall, ap = self.calculate_ap(
                flat_predictions, flat_ground_truths, iou_threshold=iou_thresh
            )
            aps.append(ap)
            
            # 保存特定阈值的结果
            if abs(iou_thresh - 0.5) < 0.01:  # mAP@0.5
                results['mAP@0.5'] = {
                    'precision': precision,
                    'recall': recall,
                    'ap': ap
                }
            elif abs(iou_thresh - 0.75) < 0.01:  # mAP@0.75
                results['mAP@0.75'] = {
                    'precision': precision,
                    'recall': recall,
                    'ap': ap
                }
        
        # 计算mAP@0.5:0.95 (平均所有IoU阈值)
        map_50_95 = np.mean(aps)
        results['mAP@0.5:0.95'] = {
            'map': map_50_95,
            'aps_per_iou': {f'{iou:.2f}': ap for iou, ap in zip(iou_thresholds, aps)}
        }
        
        return results
    
    def evaluate_on_test_set(self, dataset_yaml_path: str, conf_threshold: float = 0.25,
                             batch_size: int = 16, save_visualizations: bool = True,
                             max_vis_samples: int = 50):
        """在held-out测试集上评估最佳模型
        
        Args:
            dataset_yaml_path: 数据集配置文件路径
            conf_threshold: 置信度阈值
            batch_size: 批处理大小
            save_visualizations: 是否保存可视化
            max_vis_samples: 最大可视化样本数
        """
        self.logger.info("="*80)
        self.logger.info("EVALUATING BEST MODEL ON HELD-OUT TEST SET")
        self.logger.info("="*80)
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Confidence threshold: {conf_threshold}")
        self.logger.info("="*80)
        
        # 加载测试数据集
        image_paths, label_paths = self.load_test_dataset(dataset_yaml_path)
        
        if not image_paths:
            self.logger.error("No test images found!")
            return
        
        # 创建可视化目录
        vis_dir = self.output_dir / "visualizations"
        if save_visualizations:
            vis_dir.mkdir(exist_ok=True)
        
        all_predictions = []
        all_ground_truths = []
        
        self.logger.info(f"Processing {len(image_paths)} test images...")
        
        # 批量处理
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batch_labels = label_paths[batch_start:batch_end]
            
            progress = (batch_end / len(image_paths)) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({batch_end}/{len(image_paths)})")
            
            # 批量推理
            start_time = time.time()
            batch_results = self.model([str(path) for path in batch_paths], 
                                      device=self.device, verbose=False)
            batch_inference_time = time.time() - start_time
            
            # 处理批量结果
            for i, (img_path, label_path, result) in enumerate(zip(batch_paths, batch_labels, batch_results)):
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                img_height, img_width = image.shape[:2]
                
                # 解析真实标签
                ground_truths = self.parse_yolo_label(label_path, img_width, img_height)
                
                # 处理预测结果
                predictions = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if confidence >= conf_threshold:
                            predictions.append({
                                'class_id': class_id,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence)
                            })
                
                all_predictions.append(predictions)
                all_ground_truths.append(ground_truths)
                
                # 记录推理时间
                self.results['inference_times'].append(batch_inference_time / len(batch_paths))
                
                # 保存可视化（前N个样本）
                if save_visualizations and (batch_start + i) < max_vis_samples:
                    vis_image = self.draw_detections(image, predictions, ground_truths)
                    vis_path = vis_dir / f"{img_path.stem}_result.jpg"
                    cv2.imwrite(str(vis_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 保存结果
        self.results['predictions'] = all_predictions
        self.results['ground_truths'] = all_ground_truths
        
        # 计算评估指标
        self.logger.info("\nCalculating evaluation metrics...")
        metrics = self.calculate_map_at_iou_range(all_predictions, all_ground_truths)
        
        # 添加推理速度统计
        metrics['inference_stats'] = {
            'mean_time': np.mean(self.results['inference_times']),
            'std_time': np.std(self.results['inference_times']),
            'fps': 1.0 / np.mean(self.results['inference_times']),
            'total_images': len(image_paths)
        }
        
        self.results['metrics'] = metrics
        
        # ===== 新增：生成可视化分析 =====
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING VISUALIZATION ANALYSIS...")
        self.logger.info("="*80)
        
        # 1. 置信度阈值分析（找到最优conf）
        conf_analysis = self.analyze_confidence_thresholds(all_predictions, all_ground_truths)
        best_conf, best_f1 = self.plot_confidence_analysis(conf_analysis)
        
        # 保存置信度分析结果到CSV
        conf_df = pd.DataFrame(conf_analysis)
        conf_csv = self.output_dir / "confidence_analysis.csv"
        conf_df.to_csv(conf_csv, index=False)
        self.logger.info(f"Confidence analysis data saved to: {conf_csv}")
        
        # 2. PR曲线
        self.plot_pr_curve(all_predictions, all_ground_truths)
        
        # 3. 置信度分布
        self.plot_confidence_distribution(all_predictions)
        
        # 4. 保存最优置信度建议
        recommendation_file = self.output_dir / "confidence_recommendation.txt"
        with open(recommendation_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CONFIDENCE THRESHOLD RECOMMENDATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Optimal Confidence Threshold: {best_conf:.2f}\n")
            f.write(f"Best F1-Score: {best_f1:.3f}\n\n")
            
            # 查找该阈值下的precision和recall
            best_result = [r for r in conf_analysis if abs(r['conf_threshold'] - best_conf) < 0.001][0]
            f.write(f"At this threshold:\n")
            f.write(f"  - Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.2f}%)\n")
            f.write(f"  - Recall: {best_result['recall']:.3f} ({best_result['recall']*100:.2f}%)\n")
            f.write(f"  - F1-Score: {best_result['f1_score']:.3f} ({best_result['f1_score']*100:.2f}%)\n\n")
            
            f.write("="*60 + "\n")
            f.write("ANALYSIS FOR DIFFERENT SCENARIOS:\n")
            f.write("="*60 + "\n\n")
            
            # 找到高精度点
            high_precision_idx = np.argmax([r['precision'] for r in conf_analysis])
            high_prec_result = conf_analysis[high_precision_idx]
            f.write("1. HIGH PRECISION (minimize false positives):\n")
            f.write(f"   Recommended conf: {high_prec_result['conf_threshold']:.2f}\n")
            f.write(f"   Precision: {high_prec_result['precision']:.3f}, Recall: {high_prec_result['recall']:.3f}\n")
            f.write(f"   Use when: False detections are costly\n\n")
            
            # 找到高召回点
            high_recall_idx = np.argmax([r['recall'] for r in conf_analysis])
            high_rec_result = conf_analysis[high_recall_idx]
            f.write("2. HIGH RECALL (minimize misses):\n")
            f.write(f"   Recommended conf: {high_rec_result['conf_threshold']:.2f}\n")
            f.write(f"   Precision: {high_rec_result['precision']:.3f}, Recall: {high_rec_result['recall']:.3f}\n")
            f.write(f"   Use when: Missing cattle is unacceptable\n\n")
            
            # 平衡点
            f.write("3. BALANCED (best F1-score):\n")
            f.write(f"   Recommended conf: {best_conf:.2f}\n")
            f.write(f"   Precision: {best_result['precision']:.3f}, Recall: {best_result['recall']:.3f}\n")
            f.write(f"   Use when: Need accurate counting with precise localization\n\n")
            
            f.write("="*60 + "\n")
            f.write("PRACTICAL RECOMMENDATION:\n")
            f.write("="*60 + "\n")
            f.write(f"For your aerial cattle monitoring application:\n")
            f.write(f"  → Use conf = {best_conf:.2f} for best overall performance\n")
            f.write(f"  → This gives {best_result['precision']*100:.1f}% precision and {best_result['recall']*100:.1f}% recall\n")
            f.write(f"  → Provides accurate counting with precise cattle localization\n")
        
        self.logger.info(f"✓ Confidence recommendation saved to: {recommendation_file}")
        
        # 打印和保存结果
        self.print_results()
        self.save_results()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*80)
    
    def draw_detections(self, image: np.ndarray, predictions: List[Dict], 
                       ground_truths: List[Dict]) -> np.ndarray:
        """在图像上绘制检测框"""
        img_vis = image.copy()
        height, width = img_vis.shape[:2]
        line_thickness = max(2, width // 500)
        font_scale = max(0.5, width / 2000)
        
        class_names = self.model.names
        
        # 绘制真实框（绿色）
        for gt in ground_truths:
            x1, y1, x2, y2 = [int(coord) for coord in gt['bbox']]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
            class_name = class_names.get(gt['class_id'], f"Class_{gt['class_id']}")
            label = f'GT: {class_name}'
            cv2.putText(img_vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thickness)
        
        # 绘制预测框（红色）
        for pred in predictions:
            x1, y1, x2, y2 = [int(coord) for coord in pred['bbox']]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
            
            class_name = class_names.get(pred['class_id'], f"Class_{pred['class_id']}")
            label = f'{class_name}: {pred["confidence"]:.2f}'
            cv2.putText(img_vis, label, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
        
        return img_vis
    
    def print_results(self):
        """打印评估结果"""
        metrics = self.results['metrics']
        
        print("\n" + "="*80)
        print("TEST SET EVALUATION RESULTS")
        print("="*80)
        
        # mAP@0.5
        map_50 = metrics['mAP@0.5']
        print(f"\nmAP@0.5 Metrics:")
        print(f"  Precision: {map_50['precision']*100:.2f}%")
        print(f"  Recall: {map_50['recall']*100:.2f}%")
        print(f"  mAP@0.5: {map_50['ap']*100:.2f}%")
        
        # mAP@0.75
        map_75 = metrics['mAP@0.75']
        print(f"\nmAP@0.75 Metrics:")
        print(f"  Precision: {map_75['precision']*100:.2f}%")
        print(f"  Recall: {map_75['recall']*100:.2f}%")
        print(f"  mAP@0.75: {map_75['ap']*100:.2f}%")
        
        # mAP@0.5:0.95 (COCO-style)
        map_50_95 = metrics['mAP@0.5:0.95']
        print(f"\nmAP@0.5:0.95 (COCO-style) Metrics:")
        print(f"  mAP@0.5:0.95: {map_50_95['map']*100:.2f}%")
        
        # 推理速度
        inference_stats = metrics['inference_stats']
        print(f"\nInference Speed:")
        print(f"  Mean time per image: {inference_stats['mean_time']:.4f}s")
        print(f"  FPS: {inference_stats['fps']:.2f}")
        print(f"  Total images evaluated: {inference_stats['total_images']}")
        
        print("\n" + "="*80)
        print("PAPER-READY METRICS SUMMARY:")
        print("="*80)
        print(f"Precision: {map_50['precision']*100:.2f}%")
        print(f"Recall: {map_50['recall']*100:.2f}%")
        print(f"mAP@0.5: {map_50['ap']*100:.2f}%")
        print(f"mAP@0.5:0.95: {map_50_95['map']*100:.2f}%")
        print("="*80)
    
    def analyze_confidence_thresholds(self, all_predictions: List[List[Dict]], 
                                       all_ground_truths: List[List[Dict]]) -> Dict:
        """分析不同置信度阈值下的性能
        
        Args:
            all_predictions: 所有图像的预测结果（包含所有置信度）
            all_ground_truths: 所有图像的真实标签
            
        Returns:
            Dict: 不同阈值下的性能指标
        """
        self.logger.info("\n分析不同置信度阈值的影响...")
        
        # 测试的置信度阈值范围
        conf_thresholds = np.arange(0.1, 1.0, 0.05)
        results_by_conf = []
        
        for conf_thresh in conf_thresholds:
            # 过滤预测
            filtered_preds = []
            for preds in all_predictions:
                filtered = [p for p in preds if p['confidence'] >= conf_thresh]
                filtered_preds.append(filtered)
            
            # 计算指标
            precision, recall, ap = self.calculate_ap(
                [p for preds in filtered_preds for p in preds],
                [gt for gts in all_ground_truths for gt in gts],
                iou_threshold=0.5
            )
            
            # 计算F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results_by_conf.append({
                'conf_threshold': conf_thresh,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ap': ap
            })
        
        return results_by_conf
    
    def plot_confidence_analysis(self, conf_analysis: List[Dict]):
        """绘制置信度阈值分析图表
        
        Args:
            conf_analysis: 不同置信度下的性能分析结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        conf_values = [r['conf_threshold'] for r in conf_analysis]
        precisions = [r['precision'] for r in conf_analysis]
        recalls = [r['recall'] for r in conf_analysis]
        f1_scores = [r['f1_score'] for r in conf_analysis]
        aps = [r['ap'] for r in conf_analysis]
        
        # 1. Precision vs Confidence
        axes[0, 0].plot(conf_values, precisions, 'b-', linewidth=2.5, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Precision', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Precision vs Confidence Threshold', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # 2. Recall vs Confidence  
        axes[0, 1].plot(conf_values, recalls, 'g-', linewidth=2.5, marker='s', markersize=4)
        axes[0, 1].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Recall', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Recall vs Confidence Threshold', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # 3. F1-Score vs Confidence (最重要的图)
        axes[1, 0].plot(conf_values, f1_scores, 'r-', linewidth=2.5, marker='^', markersize=4)
        # 标记最佳F1点
        best_f1_idx = np.argmax(f1_scores)
        best_conf = conf_values[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        axes[1, 0].scatter([best_conf], [best_f1], color='red', s=200, zorder=5, 
                          marker='*', edgecolors='black', linewidth=2)
        axes[1, 0].annotate(f'Best F1: {best_f1:.3f}\nConf: {best_conf:.2f}',
                           xy=(best_conf, best_f1),
                           xytext=(best_conf + 0.1, best_f1 - 0.05),
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', lw=2))
        axes[1, 0].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('F1-Score vs Confidence Threshold (Optimal Point)', 
                             fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])
        
        # 4. Precision-Recall综合曲线
        axes[1, 1].plot(conf_values, precisions, 'b-', linewidth=2.5, marker='o', 
                       markersize=4, label='Precision', alpha=0.7)
        axes[1, 1].plot(conf_values, recalls, 'g-', linewidth=2.5, marker='s', 
                       markersize=4, label='Recall', alpha=0.7)
        axes[1, 1].plot(conf_values, f1_scores, 'r-', linewidth=2.5, marker='^', 
                       markersize=4, label='F1-Score', alpha=0.7)
        axes[1, 1].axvline(x=best_conf, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 1].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('All Metrics vs Confidence Threshold', fontsize=14, fontweight='bold')
        axes[1, 1].legend(loc='best', fontsize=11, frameon=True, shadow=True)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / "confidence_threshold_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confidence analysis plot saved to: {save_path}")
        self.logger.info(f"✓ Optimal confidence threshold: {best_conf:.2f} (F1={best_f1:.3f})")
        
        return best_conf, best_f1
    
    def plot_pr_curve(self, all_predictions: List[List[Dict]], 
                      all_ground_truths: List[List[Dict]]):
        """绘制Precision-Recall曲线
        
        Args:
            all_predictions: 所有预测结果
            all_ground_truths: 所有真实标签
        """
        # 展平预测和标签
        flat_preds = [p for preds in all_predictions for p in preds]
        flat_gts = [gt for gts in all_ground_truths for gt in gts]
        
        if not flat_preds or not flat_gts:
            return
        
        # 按置信度排序
        flat_preds = sorted(flat_preds, key=lambda x: x['confidence'], reverse=True)
        
        # 计算每个点的precision和recall
        precisions = []
        recalls = []
        
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred in flat_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(flat_gts):
                if gt_idx in matched_gt or pred['class_id'] != gt['class_id']:
                    continue
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(flat_gts) if len(flat_gts) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        # 绘制PR曲线
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recalls, precisions, 'b-', linewidth=2.5, alpha=0.8)
        ax.fill_between(recalls, precisions, alpha=0.2)
        
        # 计算AP (曲线下面积)
        ap = np.trapz(precisions, recalls) if recalls else 0
        
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve (AP={ap:.3f})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"PR curve saved to: {save_path}")
    
    def plot_confidence_distribution(self, all_predictions: List[List[Dict]]):
        """绘制预测置信度分布图
        
        Args:
            all_predictions: 所有预测结果
        """
        confidences = [p['confidence'] for preds in all_predictions for p in preds]
        
        if not confidences:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 直方图
        axes[0].hist(confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(confidences), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].axvline(x=np.median(confidences), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        axes[0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. 累积分布
        sorted_conf = np.sort(confidences)
        cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        axes[1].plot(sorted_conf, cumulative, 'b-', linewidth=2.5)
        axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].axhline(y=0.9, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        axes[1].set_title('Cumulative Confidence Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "confidence_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confidence distribution plot saved to: {save_path}")
    
    def save_results(self):
        """保存评估结果"""
        # 保存JSON格式的详细结果
        results_file = self.output_dir / "test_evaluation_results.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_converted = convert_numpy(self.results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Detailed results saved to: {results_file}")
        
        # 保存CSV格式的指标摘要
        metrics = self.results['metrics']
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        summary_data['Metric'].extend([
            'Precision @ IoU=0.5',
            'Recall @ IoU=0.5',
            'mAP@0.5',
            'mAP@0.75',
            'mAP@0.5:0.95',
            'Mean Inference Time (s)',
            'FPS'
        ])
        
        summary_data['Value'].extend([
            f"{metrics['mAP@0.5']['precision']*100:.2f}%",
            f"{metrics['mAP@0.5']['recall']*100:.2f}%",
            f"{metrics['mAP@0.5']['ap']*100:.2f}%",
            f"{metrics['mAP@0.75']['ap']*100:.2f}%",
            f"{metrics['mAP@0.5:0.95']['map']*100:.2f}%",
            f"{metrics['inference_stats']['mean_time']:.4f}",
            f"{metrics['inference_stats']['fps']:.2f}"
        ])
        
        df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / "test_metrics_summary.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Metrics summary saved to: {csv_file}")
        
        # 保存LaTeX格式的表格
        latex_file = self.output_dir / "test_metrics_latex.txt"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% LaTeX table for paper\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{YOLOv11s Detection Performance on Held-out Test Set}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            f.write(f"Precision (\\%) & {metrics['mAP@0.5']['precision']*100:.2f} \\\\\n")
            f.write(f"Recall (\\%) & {metrics['mAP@0.5']['recall']*100:.2f} \\\\\n")
            f.write(f"mAP@0.5 (\\%) & {metrics['mAP@0.5']['ap']*100:.2f} \\\\\n")
            f.write(f"mAP@0.5:0.95 (\\%) & {metrics['mAP@0.5:0.95']['map']*100:.2f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:yolo_test_results}\n")
            f.write("\\end{table}\n")
        
        self.logger.info(f"LaTeX table saved to: {latex_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate best YOLOv11 checkpoint on held-out test set for paper'
    )
    parser.add_argument('--model', type=str, default="YOLOV11/output/yolov11s_cattle_20260109_223121_500/weights/best.pt", 
                        help='Path to trained model file')
    parser.add_argument('--data', type=str, default='data/yolodet_train_dataset/dataset.yaml',
                       help='Dataset YAML file')
    parser.add_argument('--output', type=str, default="YOLOV11/output/yolov11s_cattle_20260109_223121_500/test_results_50",
                        help='Output directory for evaluation results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization saving')
    parser.add_argument('--max-vis', type=int, default=50,
                       help='Maximum number of visualization samples to save')
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = BestModelEvaluator(
            model_path=args.model,
            output_dir=args.output,
            device=args.device
        )
        
        # 在测试集上评估
        evaluator.evaluate_on_test_set(
            dataset_yaml_path=args.data,
            conf_threshold=args.conf,
            batch_size=args.batch_size,
            save_visualizations=not args.no_vis,
            max_vis_samples=args.max_vis
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
