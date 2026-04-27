"""
牛实例数据收集脚本 (Cattle Instance Data Collection Script)
==========================================================

目的：收集所有YOLO检测到的实例数据，不做任何过滤
用途：为后续参数敏感性分析提供完整的原始数据

数据收集策略：
- 固定conf_threshold=0.5（可配置但建议较低）
- 不做ROI位置过滤（保存bbox中心坐标供后续分析）
- 不做WL_Ratio过滤（保存所有长宽比）
- Standing和Laying分别处理和保存
- 保存mask图像 + 完整元数据CSV
- 可选保存可视化图像（节省空间）

后续分析流程：
1. 使用本脚本收集所有实例数据（运行一次，耗时数小时）
2. 使用敏感性分析脚本快速探索不同参数组合（耗时几秒到几分钟）
"""

import sys
import cv2
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
import math
import torch
import glob
from tqdm import tqdm
import datetime
from typing import List, Tuple, Dict, Optional
import time

try:
    np.int0
except AttributeError:
    np.int0 = np.int32
    
try:
    np.float
except AttributeError:
    np.float = float

from ultralytics import YOLO, SAM
import re
from datetime import datetime as dt


def extract_date_info(folder_path):
    """
    从文件夹名提取日期和高度信息
    
    Args:
        folder_path: 文件夹路径，如 'data/yolo_sam_w_15m/20240710_15m'
    
    Returns:
        tuple: (date_str, height, day_index)
            - date_str: '2024-07-10'
            - height: '15m'
            - day_index: 相对第一天的天数（需要在外部计算）
    """
    folder_name = Path(folder_path).name
    
    # 匹配格式：20240710_15m
    match = re.match(r'(\d{8})_(\d+m)', folder_name)
    if match:
        date_str = match.group(1)
        height = match.group(2)
        # 转换日期格式：20240710 → 2024-07-10
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_formatted, height, folder_name
    
    return None, None, folder_name


class YOLODetector:
    """YOLOv11检测器模块"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型文件路径
            config_path: YOLO配置文件路径
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """设置推理设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ 成功加载YOLO模型: {self.model_path}")
            print(f"✓ 使用设备: {self.device}")
        except Exception as e:
            print(f"✗ 加载YOLO模型失败: {e}")
            raise
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        使用YOLO进行目标检测（不过滤类别）
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
        
        Returns:
            检测结果列表，每个元素包含 bbox, confidence, class_id
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            results = self.model(image, conf=conf_threshold, device=self.device, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id)
                        })
            
            return detections
            
        except Exception as e:
            print(f"✗ YOLO检测失败: {e}")
            return []


class SAMSegmenter:
    """SAM分割器模块 - 使用ultralytics集成的SAM"""
    
    def __init__(self, model_path: str = "sam2.1_l.pt", device: str = 'auto'):
        """
        初始化SAM分割器
        
        Args:
            model_path: SAM模型文件路径或名称
            device: 设备类型
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """设置推理设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """加载SAM模型"""
        try:
            print(f"正在加载SAM模型: {self.model_path}")
            
            # 尝试不同的模型名称
            model_names_to_try = [
                self.model_path,
                "sam2.1_l.pt",
                "sam2_l.pt", 
                "sam_l.pt"
            ]
            
            model_loaded = False
            for model_name in model_names_to_try:
                try:
                    self.model = SAM(model_name)
                    self.model_path = model_name
                    model_loaded = True
                    print(f"✓ 成功加载SAM模型: {model_name}")
                    print(f"✓ 使用设备: {self.device}")
                    break
                except Exception as e:
                    continue
            
            if not model_loaded:
                raise RuntimeError("所有SAM模型加载尝试都失败")
                
        except Exception as e:
            print(f"✗ 加载SAM模型失败: {e}")
            self.model = None
            raise

    def segment_from_boxes(self, image: np.ndarray, boxes: List[List[int]]) -> List[np.ndarray]:
        """
        基于边界框进行分割
        
        Args:
            image: 输入图像
            boxes: 边界框列表 [[x1, y1, x2, y2], ...]
        
        Returns:
            掩膜列表
        """
        if self.model is None:
            raise RuntimeError("SAM模型未加载")
        
        try:
            results = self.model(image, bboxes=boxes, device=self.device, verbose=False)
            
            masks = []
            for result in results:
                if result.masks is not None:
                    masks_data = result.masks.data.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        if i < len(masks_data):
                            mask = masks_data[i]
                            mask = (mask * 255).astype(np.uint8)
                            masks.append(mask)
                        else:
                            h, w = image.shape[:2]
                            empty_mask = np.zeros((h, w), dtype=np.uint8)
                            masks.append(empty_mask)
                else:
                    h, w = image.shape[:2]
                    for _ in boxes:
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        masks.append(empty_mask)
            
            while len(masks) < len(boxes):
                h, w = image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                masks.append(empty_mask)
            
            return masks[:len(boxes)]
            
        except Exception as e:
            print(f"✗ SAM分割失败: {e}")
            h, w = image.shape[:2]
            masks = []
            for _ in boxes:
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                masks.append(empty_mask)
            return masks


class CowProcessor:
    """牛处理器模块 - 整合检测和分割结果（无过滤版本）"""
    
    def __init__(self, yolo_detector: YOLODetector, sam_segmenter: SAMSegmenter):
        """
        初始化牛处理器
        
        Args:
            yolo_detector: YOLO检测器实例
            sam_segmenter: SAM分割器实例
        """
        self.yolo_detector = yolo_detector
        self.sam_segmenter = sam_segmenter
    
    def process_image(self, image_path: str, conf_threshold: float = 0.5) -> Dict[str, List]:
        """
        处理单张图像：检测 + 分割（不过滤类别）
        
        Args:
            image_path: 图像路径
            conf_threshold: 检测置信度阈值
        
        Returns:
            {'standing': [(mask, bbox, conf), ...], 'laying': [(mask, bbox, conf), ...]}
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # YOLO检测（所有类别）
        detections = self.yolo_detector.detect(image, conf_threshold)
        
        if not detections:
            return {'standing': [], 'laying': []}
        
        # 按类别分组
        standing_detections = [det for det in detections if det['class_id'] == 1]
        laying_detections = [det for det in detections if det['class_id'] == 0]
        
        results = {'standing': [], 'laying': []}
        
        # 处理standing类别
        if standing_detections:
            boxes = [det['bbox'] for det in standing_detections]
            confidences = [det['confidence'] for det in standing_detections]
            masks = self.sam_segmenter.segment_from_boxes(image, boxes)
            
            for mask, bbox, conf in zip(masks, boxes, confidences):
                results['standing'].append((mask, bbox, conf))
        
        # 处理laying类别
        if laying_detections:
            boxes = [det['bbox'] for det in laying_detections]
            confidences = [det['confidence'] for det in laying_detections]
            masks = self.sam_segmenter.segment_from_boxes(image, boxes)
            
            for mask, bbox, conf in zip(masks, boxes, confidences):
                results['laying'].append((mask, bbox, conf))
        
        return results


def calculate_cow_dimensions(mask):
    """计算牛的尺寸信息"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    rect_width = rect[1][0]
    rect_height = rect[1][1]
    
    length = max(rect_width, rect_height)
    width = min(rect_width, rect_height)
    aspect_ratio = round(length / width, 3) if width > 0 else 0
    wl_ratio = round(width / length, 3) if length > 0 else 0  # 添加W/L ratio
    perimeter = round(cv2.arcLength(largest_contour, True), 3)
    
    def get_max_distance_points(contour):
        max_dist = 0
        max_points = None
        points = contour.reshape(-1, 2)
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    max_points = (points[i], points[j])
        
        return max_points, max_dist
    
    head_tail_points, head_tail_distance = get_max_distance_points(largest_contour)
    
    return {
        'area': round(area, 3),
        'bbox_width': w,
        'bbox_height': h,
        'min_rect_length': round(length, 3),
        'min_rect_width': round(width, 3),
        'aspect_ratio': aspect_ratio,
        'wl_ratio': wl_ratio,  # W/L = Width/Length
        'perimeter': perimeter,
        'head_tail_distance': round(head_tail_distance, 3),
        'head_tail_points': head_tail_points,
        'min_rect_box': box,
        'min_rect_angle': round(rect[2], 3)
    }


def create_cow_specific_mask(full_mask, box, image_shape):
    """创建特定牛的掩膜"""
    cow_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cow_mask[full_mask > 127] = 255
    
    x_min, y_min, x_max, y_max = box
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_shape[1], x_max + padding)
    y_max = min(image_shape[0], y_max + padding)
    
    bbox_mask = cow_mask[y_min:y_max, x_min:x_max].copy()
    
    if np.sum(bbox_mask) == 0:
        return bbox_mask, (x_min, y_min, x_max, y_max)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bbox_mask, connectivity=8)
    
    if num_labels <= 1:
        return bbox_mask, (x_min, y_min, x_max, y_max)
    
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    clean_mask = np.zeros_like(bbox_mask)
    clean_mask[labels == largest_component] = 255
    
    return clean_mask, (x_min, y_min, x_max, y_max)


def process_single_instance(color_image, mask, bbox, confidence, class_name, 
                           instance_id, image_name, output_dir,
                           date_info=None):
    """
    处理单个实例：计算尺寸并保存数据（保存3种图像：mask, crop, annotated）
    
    Args:
        color_image: 原始图像
        mask: 分割掩膜
        bbox: 边界框 [x1, y1, x2, y2]
        confidence: 置信度
        class_name: 类别名称 ('standing' or 'laying')
        instance_id: 实例ID
        image_name: 图像名称
        output_dir: 输出目录
        date_info: 日期信息字典 {'date': '2024-07-10', 'height': '15m', 'folder_name': '20240710_15m'}
    
    Returns:
        dict: 实例数据字典
    """
    clean_mask, adjusted_box = create_cow_specific_mask(mask, bbox, color_image.shape)
    x_min, y_min, x_max, y_max = adjusted_box
    
    # 检查掩膜是否为空
    if np.sum(clean_mask) == 0:
        return None
    
    # 计算尺寸信息
    dimensions = calculate_cow_dimensions(clean_mask)
    if dimensions is None:
        return None
    
    # 计算bbox中心坐标（用于后续ROI判断）
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    
    # 准备实例数据（添加日期和时间维度字段）
    instance_data = {
        'Image_Name': image_name,
        'Instance_ID': instance_id,
        'Class': class_name,
        'Confidence': confidence,
        'BBox_x1': bbox[0],
        'BBox_y1': bbox[1],
        'BBox_x2': bbox[2],
        'BBox_y2': bbox[3],
        'BBox_Center_X': bbox_center_x,
        'BBox_Center_Y': bbox_center_y,
        'Area_pixels': dimensions['area'],
        'BBox_Width_pixels': dimensions['bbox_width'],
        'BBox_Height_pixels': dimensions['bbox_height'],
        'MinRect_Length_pixels': dimensions['min_rect_length'],
        'MinRect_Width_pixels': dimensions['min_rect_width'],
        'Aspect_Ratio': dimensions['aspect_ratio'],  # L/W (Length/Width)
        'WL_Ratio': dimensions['wl_ratio'],  # W/L (Width/Length)
        'Perimeter_pixels': dimensions['perimeter'],
        'Head_Tail_Distance_pixels': dimensions['head_tail_distance'],
        'MinRect_Angle_degrees': dimensions['min_rect_angle']
    }
    
    # 添加日期和时间维度信息
    if date_info:
        instance_data['Date'] = date_info.get('date', 'Unknown')
        instance_data['Height'] = date_info.get('height', 'Unknown')
        instance_data['Source_Folder'] = date_info.get('folder_name', 'Unknown')
        instance_data['Day_Index'] = date_info.get('day_index', -1)
    
    # 提取实例区域（crop）
    color_crop = color_image[y_min:y_max, x_min:x_max]
    
    # 生成文件名前缀（包含日期信息）
    if date_info and date_info.get('folder_name'):
        filename_prefix = f"{date_info['folder_name']}_{image_name}_inst{instance_id:04d}"
    else:
        filename_prefix = f"{image_name}_inst{instance_id:04d}"
    
    # 1. 保存纯黑白mask
    mask_dir = os.path.join(output_dir, 'masks', class_name)
    os.makedirs(mask_dir, exist_ok=True)
    mask_filename = f"{filename_prefix}_mask.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    cv2.imwrite(mask_path, clean_mask)
    instance_data['Mask_Path'] = os.path.relpath(mask_path, output_dir)
    
    # 2. 保存原始crop图像
    crop_dir = os.path.join(output_dir, 'crops_original', class_name)
    os.makedirs(crop_dir, exist_ok=True)
    crop_filename = f"{filename_prefix}_crop.png"
    crop_path = os.path.join(crop_dir, crop_filename)
    cv2.imwrite(crop_path, color_crop)
    instance_data['Crop_Path'] = os.path.relpath(crop_path, output_dir)
    
    # 3. 保存简洁标注图（原始crop + 绿色掩膜叠加 + 顶部文本）
    annotated_dir = os.path.join(output_dir, 'annotated', class_name)
    os.makedirs(annotated_dir, exist_ok=True)
    
    # 创建标注图
    annotated = color_crop.copy()
    mask_indices = clean_mask > 0
    
    # 绿色掩膜叠加（半透明）
    overlay = annotated.copy()
    overlay[mask_indices] = (0, 255, 0)  # 绿色
    annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)
    
    # 添加顶部文本：ID:0001 | Conf:0.95 | W/L:0.285
    text = f"ID:{instance_id:04d} | Conf:{confidence:.3f} | W/L:{dimensions['wl_ratio']:.3f}"
    
    # 文本背景（黑色半透明矩形）
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(annotated, (0, 0), (text_size[0] + 10, text_size[1] + 10), 
                 (0, 0, 0), -1)
    cv2.rectangle(annotated, (0, 0), (text_size[0] + 10, text_size[1] + 10), 
                 (255, 255, 255), 1)
    
    # 白色文本
    cv2.putText(annotated, text, (5, text_size[1] + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    ann_filename = f"{filename_prefix}_ann.png"
    ann_path = os.path.join(annotated_dir, ann_filename)
    cv2.imwrite(ann_path, annotated)
    instance_data['Annotated_Path'] = os.path.relpath(ann_path, output_dir)
    
    return instance_data


def process_image_all_instances(image_path: str, output_dir: str,
                                cow_processor: CowProcessor,
                                conf_threshold: float = 0.5,
                                date_info: dict = None):
    """
    处理单张图像，提取所有实例数据
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        cow_processor: CowProcessor实例
        conf_threshold: 置信度阈值
        date_info: 日期信息字典
    
    Returns:
        list: 实例数据列表
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ 无法读取图像: {image_path}")
        return []
    
    image_name = Path(image_path).stem
    image_height, image_width = image.shape[:2]
    
    # 记录YOLO+SAM推理时间（不包括保存时间）
    inference_start = time.time()
    results = cow_processor.process_image(image_path, conf_threshold)
    inference_time = time.time() - inference_start  # 单位：秒
    
    all_instances_data = []
    global_instance_id = 1
    
    # 处理standing实例
    for mask, bbox, conf in results['standing']:
        instance_data = process_single_instance(
            image, mask, bbox, conf, 'standing',
            global_instance_id, image_name, output_dir,
            date_info
        )
        if instance_data:
            instance_data['Image_Width'] = image_width
            instance_data['Image_Height'] = image_height
            instance_data['Inference_Time_Seconds'] = round(inference_time, 3)  # YOLO+SAM推理时间
            all_instances_data.append(instance_data)
            global_instance_id += 1
    
    # 处理laying实例
    for mask, bbox, conf in results['laying']:
        instance_data = process_single_instance(
            image, mask, bbox, conf, 'laying',
            global_instance_id, image_name, output_dir,
            date_info
        )
        if instance_data:
            instance_data['Image_Width'] = image_width
            instance_data['Image_Height'] = image_height
            instance_data['Inference_Time_Seconds'] = round(inference_time, 3)  # YOLO+SAM推理时间
            all_instances_data.append(instance_data)
            global_instance_id += 1
    
    return all_instances_data


def process_folder_all_instances(input_folder: str,
                                 output_dir: str,
                                 cow_processor: CowProcessor,
                                 conf_threshold: float = 0.5,
                                 all_dates_info: dict = None):
    """
    处理文件夹中的所有图像，收集所有实例数据（返回数据而不是保存）
    
    Args:
        input_folder: 输入文件夹路径
        output_dir: 统一输出目录（根目录下的instances_results）
        cow_processor: CowProcessor实例（共享）
        conf_threshold: 置信度阈值
        all_dates_info: 所有日期信息字典（用于计算day_index）
    
    Returns:
        tuple: (instances_data_list, stats_dict)
    """
    print(f"\n{'='*80}")
    print(f"牛实例数据收集脚本 - 开始处理文件夹")
    print(f"{'='*80}")
    print(f"输入文件夹: {input_folder}")
    print(f"置信度阈值: {conf_threshold}")
    
    # 提取当前文件夹的日期信息
    date_str, height, folder_name = extract_date_info(input_folder)
    print(f"日期信息: {date_str} | 高度: {height}")
    
    # 计算day_index（相对于最早日期的天数）
    day_index = -1
    if all_dates_info and date_str and date_str in all_dates_info:
        day_index = all_dates_info[date_str]
    
    date_info = {
        'date': date_str,
        'height': height,
        'folder_name': folder_name,
        'day_index': day_index
    }
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"✗ 输入文件夹不存在: {input_folder}")
        return [], {}
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"✗ 未找到图像文件")
        return [], {}
    
    print(f"✓ 找到 {len(image_files)} 个图像文件")
    
    # 处理所有图像
    folder_instances_data = []
    stats = {
        'folder_name': folder_name,
        'date': date_str,
        'total_images': len(image_files),
        'processed_images': 0,
        'total_standing': 0,
        'total_laying': 0,
        'failed_images': 0,
        'total_inference_time': 0.0,  # 总推理时间
        'avg_inference_time': 0.0  # 平均推理时间
    }
    
    print("开始处理图像...")
    for image_file in tqdm(image_files, desc=f"Processing {folder_name}"):
        try:
            instances_data = process_image_all_instances(
                image_file, output_dir, cow_processor,
                conf_threshold, date_info
            )
            
            if instances_data:
                folder_instances_data.extend(instances_data)
                standing_count = sum(1 for d in instances_data if d['Class'] == 'standing')
                laying_count = sum(1 for d in instances_data if d['Class'] == 'laying')
                stats['total_standing'] += standing_count
                stats['total_laying'] += laying_count
                stats['processed_images'] += 1
                # 累积推理时间（从第一个实例获取，因为同一图像的所有实例共享推理时间）
                if instances_data:
                    stats['total_inference_time'] += instances_data[0].get('Inference_Time_Seconds', 0)
            
        except Exception as e:
            print(f"\n✗ 处理图像失败 {Path(image_file).name}: {e}")
            stats['failed_images'] += 1
            continue
    
    # 计算平均推理时间
    if stats['processed_images'] > 0:
        stats['avg_inference_time'] = round(stats['total_inference_time'] / stats['processed_images'], 3)
    
    # 打印当前文件夹统计
    print(f"\n{'='*60}")
    print(f"文件夹 {folder_name} 处理完成")
    print(f"{'='*60}")
    print(f"处理图像: {stats['processed_images']}/{stats['total_images']}")
    print(f"实例数: Standing={stats['total_standing']}, Laying={stats['total_laying']}")
    print(f"推理时间: 总计={stats['total_inference_time']:.3f}s, 平均={stats['avg_inference_time']:.3f}s/图像")
    print(f"{'='*60}\n")
    
    return folder_instances_data, stats


def main():
    """主函数"""
    # ========== 配置参数 ==========
    
    # 数据路径
    base_input_folder = "data/yolo_sam_w_10m"
    
    # 模型路径
    yolo_model_path = "YOLOV11/output/yolov11s_cattle_20260109_223121_500/weights/best.pt"
    yolo_config_path = "YOLOV11/configs/config_s.py"
    sam_model_path = "YOLOV11/sam2.1/checkpoints/sam2.1_l.pt"  # 使用ultralytics官方SAM
    
    # 处理参数
    conf_threshold = 0.5  # 置信度阈值（建议0.3-0.5，越低收集越全）
    device = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # ========== 处理文件夹 ==========
    
    # 获取所有子文件夹（只保留日期格式的文件夹，如20240710_15m）
    all_folders = [f for f in Path(base_input_folder).iterdir() if f.is_dir()]
    
    # 过滤出符合日期格式的文件夹（排除results、instances_results等）
    date_pattern = re.compile(r'^\d{8}_\d+m$')  # 匹配如：20240710_15m
    subfolders = [f for f in all_folders if date_pattern.match(f.name)]
    
    if len(subfolders) < len(all_folders):
        excluded = [f.name for f in all_folders if f not in subfolders]
        print(f"排除非日期文件夹: {', '.join(excluded)}")
    
    # 提取所有日期信息并计算day_index
    all_dates = []
    for folder in subfolders:
        date_str, _, _ = extract_date_info(str(folder))
        if date_str:
            all_dates.append(date_str)
    
    # 按日期排序，计算每个日期相对于最早日期的天数
    all_dates_info = {}
    if all_dates:
        all_dates.sort()
        base_date = dt.strptime(all_dates[0], '%Y-%m-%d')
        for date_str in all_dates:
            current_date = dt.strptime(date_str, '%Y-%m-%d')
            days_diff = (current_date - base_date).days
            all_dates_info[date_str] = days_diff
        
        print(f"\n{'='*80}")
        print("时间序列信息:")
        print(f"{'='*80}")
        print(f"基准日期（Day 0）: {all_dates[0]}")
        print(f"结束日期: {all_dates[-1]}")
        print(f"总时间跨度: {max(all_dates_info.values())} 天")
        print(f"采样日期数: {len(all_dates)}")
        print(f"{'='*80}\n")
    
    # 创建统一的输出目录（在根目录下）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_output_dir = os.path.join(base_input_folder, f"instances_results_{timestamp}")
    os.makedirs(unified_output_dir, exist_ok=True)
    print(f"\n统一输出目录: {unified_output_dir}\n")
    
    # 初始化模型（只初始化一次，所有文件夹共享）
    try:
        print("初始化模型...")
        yolo_detector = YOLODetector(yolo_model_path, yolo_config_path, device)
        sam_segmenter = SAMSegmenter(sam_model_path, device)
        cow_processor = CowProcessor(yolo_detector, sam_segmenter)
        print("✓ 模型初始化完成\n")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return
    
    # 累积所有文件夹的数据
    all_instances_data = []
    all_folder_stats = []
    
    if not subfolders:
        print(f"在 {base_input_folder} 中未找到子文件夹，将处理主文件夹")
        instances_data, stats = process_folder_all_instances(
            input_folder=base_input_folder,
            output_dir=unified_output_dir,
            cow_processor=cow_processor,
            conf_threshold=conf_threshold,
            all_dates_info=all_dates_info
        )
        all_instances_data.extend(instances_data)
        all_folder_stats.append(stats)
    else:
        print(f"找到 {len(subfolders)} 个子文件夹\n")
        for i, folder in enumerate(subfolders, 1):
            print(f"\n{'#'*80}")
            print(f"处理文件夹 [{i}/{len(subfolders)}]: {folder.name}")
            print(f"{'#'*80}")
            
            instances_data, stats = process_folder_all_instances(
                input_folder=str(folder),
                output_dir=unified_output_dir,
                cow_processor=cow_processor,
                conf_threshold=conf_threshold,
                all_dates_info=all_dates_info
            )
            all_instances_data.extend(instances_data)
            all_folder_stats.append(stats)
    
    # 保存汇总数据
    print(f"\n{'='*80}")
    print("保存汇总数据")
    print(f"{'='*80}")
    
    if all_instances_data:
        # 完整数据集
        df_all = pd.DataFrame(all_instances_data)
        csv_path_all = os.path.join(unified_output_dir, "instances_metadata_all.csv")
        df_all.to_csv(csv_path_all, index=False, encoding='utf-8-sig')
        print(f"✓ 完整元数据已保存: {csv_path_all}")
        print(f"  总实例数: {len(df_all)}")
        
        # 按类别分别保存
        df_standing = df_all[df_all['Class'] == 'standing']
        df_laying = df_all[df_all['Class'] == 'laying']
        
        if len(df_standing) > 0:
            csv_standing = os.path.join(unified_output_dir, "instances_metadata_standing.csv")
            df_standing.to_csv(csv_standing, index=False, encoding='utf-8-sig')
            print(f"✓ Standing元数据已保存: {csv_standing}")
            print(f"  Standing实例数: {len(df_standing)}")
        
        if len(df_laying) > 0:
            csv_laying = os.path.join(unified_output_dir, "instances_metadata_laying.csv")
            df_laying.to_csv(csv_laying, index=False, encoding='utf-8-sig')
            print(f"✓ Laying元数据已保存: {csv_laying}")
            print(f"  Laying实例数: {len(df_laying)}")
        
        # 保存文件夹级别的统计
        df_folder_stats = pd.DataFrame(all_folder_stats)
        folder_stats_path = os.path.join(unified_output_dir, "folder_statistics.csv")
        df_folder_stats.to_csv(folder_stats_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件夹统计已保存: {folder_stats_path}")
        
        # 打印总体统计
        print(f"\n{'='*80}")
        print("总体统计")
        print(f"{'='*80}")
        print(f"处理文件夹数: {len(all_folder_stats)}")
        print(f"总图像数: {sum(s['total_images'] for s in all_folder_stats)}")
        print(f"成功处理: {sum(s['processed_images'] for s in all_folder_stats)}")
        print(f"总实例数: {len(all_instances_data)}")
        print(f"  - Standing: {len(df_standing)}")
        print(f"  - Laying: {len(df_laying)}")
        
        # 推理时间统计
        total_inference_time = sum(s.get('total_inference_time', 0) for s in all_folder_stats)
        total_processed = sum(s['processed_images'] for s in all_folder_stats)
        if total_processed > 0:
            avg_inference_time = total_inference_time / total_processed
            print(f"\n推理时间统计 (YOLO+SAM):")
            print(f"  总推理时间: {total_inference_time:.3f} 秒 ({total_inference_time/60:.3f} 分钟)")
            print(f"  平均推理时间: {avg_inference_time:.3f} 秒/图像")
            print(f"  推理吞吐量: {1/avg_inference_time:.3f} 图像/秒")
        
        # 按日期统计
        if 'Date' in df_standing.columns and len(df_standing) > 0:
            print(f"\n按日期统计（Standing）:")
            date_stats = df_standing.groupby('Date').agg({
                'Instance_ID': 'count',
                'WL_Ratio': ['mean', 'std'],
                'Confidence': 'mean'
            }).round(3)
            print(date_stats.to_string())
        
        # 保存汇总统计
        total_inference_time = sum(s.get('total_inference_time', 0) for s in all_folder_stats)
        total_processed = sum(s['processed_images'] for s in all_folder_stats)
        avg_inference_time = total_inference_time / total_processed if total_processed > 0 else 0
        
        summary = {
            'Collection_Timestamp': timestamp,
            'Total_Folders': len(all_folder_stats),
            'Total_Images': sum(s['total_images'] for s in all_folder_stats),
            'Processed_Images': sum(s['processed_images'] for s in all_folder_stats),
            'Total_Instances': len(all_instances_data),
            'Standing_Instances': len(df_standing),
            'Laying_Instances': len(df_laying),
            'Confidence_Threshold': conf_threshold,
            'Dates_Processed': sorted(df_all['Date'].unique().tolist()) if 'Date' in df_all.columns else [],
            'Inference_Statistics': {
                'Total_Inference_Time_Seconds': round(total_inference_time, 3),
                'Total_Inference_Time_Minutes': round(total_inference_time / 60, 3),
                'Average_Inference_Time_Seconds': round(avg_inference_time, 3),
                'Throughput_Images_Per_Second': round(1/avg_inference_time, 3) if avg_inference_time > 0 else 0
            }
        }
        
        summary_path = os.path.join(unified_output_dir, "collection_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 汇总统计已保存: {summary_path}")
    else:
        print("\n✗ 未收集到任何实例数据")
    
    print("\n" + "="*80)
    print("所有文件夹处理完成！")
    print("="*80)


if __name__ == "__main__":
    main()
