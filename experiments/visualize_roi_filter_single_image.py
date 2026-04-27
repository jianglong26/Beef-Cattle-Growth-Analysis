"""
单张图像ROI过滤可视化脚本
功能：
1. 对指定图像进行YOLOv11推理
2. 只保留standing类别的检测
3. 计算每个检测框的质心坐标
4. 根据center_70 ROI规则过滤
5. 可视化：ROI红框 + 保留的检测框 + 质心点
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


class ROIFilterVisualizer:
    def __init__(self, model_path, confidence_threshold=0.8):
        """
        初始化可视化器
        
        Args:
            model_path: YOLOv11模型路径
            confidence_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.class_names = {0: 'laying', 1: 'standing'}
        
        # center_70 ROI配置（归一化坐标）- 与cattle_sensitivity_analysis_15m.py一致
        self.roi_config = {
            'name': 'center_70',
            'coords': [0.15, 0.15, 0.85, 0.85],  # [x1, y1, x2, y2] - 正方形ROI
            'description': 'Center 70% × 70% square region'
        }
    
    def compute_centroid(self, bbox, img_width, img_height):
        """
        计算边界框的质心坐标
        
        Args:
            bbox: [x1, y1, x2, y2] 像素坐标
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            (cx_norm, cy_norm): 归一化质心坐标
            (cx_pixel, cy_pixel): 像素质心坐标
        """
        x1, y1, x2, y2 = bbox
        cx_pixel = (x1 + x2) / 2
        cy_pixel = (y1 + y2) / 2
        
        cx_norm = cx_pixel / img_width
        cy_norm = cy_pixel / img_height
        
        return (cx_norm, cy_norm), (cx_pixel, cy_pixel)
    
    def is_in_roi(self, centroid_norm):
        """
        判断质心是否在ROI内
        
        Args:
            centroid_norm: (cx_norm, cy_norm) 归一化质心坐标
            
        Returns:
            bool: True if in ROI, False otherwise
        """
        cx_norm, cy_norm = centroid_norm
        x1, y1, x2, y2 = self.roi_config['coords']
        
        return (x1 <= cx_norm <= x2) and (y1 <= cy_norm <= y2)
    
    def process_image(self, image_path):
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 包含检测结果和统计信息
        """
        print(f"\n{'='*80}")
        print(f"处理图像: {image_path}")
        print(f"{'='*80}")
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        print(f"图像尺寸: {img_width} × {img_height}")
        
        # YOLOv11推理
        print(f"\n执行YOLOv11推理（置信度阈值={self.conf_threshold}）...")
        results = self.model(img_rgb, conf=self.conf_threshold, verbose=False)
        
        # 解析检测结果
        detections = []
        total_standing = 0
        roi_standing = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                # 只保留standing类别
                if cls_id == 1:  # standing
                    total_standing += 1
                    
                    # 计算质心
                    centroid_norm, centroid_pixel = self.compute_centroid(
                        bbox, img_width, img_height
                    )
                    
                    # 判断是否在ROI内
                    in_roi = self.is_in_roi(centroid_norm)
                    
                    if in_roi:
                        roi_standing += 1
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'centroid_norm': centroid_norm,
                        'centroid_pixel': centroid_pixel,
                        'in_roi': in_roi
                    })
        
        print(f"\n检测结果统计:")
        print(f"  总standing检测数: {total_standing}")
        print(f"  ROI内standing数: {roi_standing}")
        print(f"  ROI外standing数: {total_standing - roi_standing}")
        print(f"  保留比例: {roi_standing/total_standing*100:.1f}%" if total_standing > 0 else "  保留比例: N/A")
        
        return {
            'image': img_rgb,
            'detections': detections,
            'img_width': img_width,
            'img_height': img_height,
            'total_standing': total_standing,
            'roi_standing': roi_standing
        }
    
    def visualize_results(self, results, save_path):
        """
        可视化检测结果
        
        Args:
            results: process_image返回的结果字典
            save_path: 保存路径
        """
        img = results['image']
        detections = results['detections']
        img_width = results['img_width']
        img_height = results['img_height']
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(img)
        ax.axis('off')
        
        # 绘制ROI红色边框
        x1, y1, x2, y2 = self.roi_config['coords']
        roi_pixel = [
            x1 * img_width,
            y1 * img_height,
            (x2 - x1) * img_width,
            (y2 - y1) * img_height
        ]
        
        roi_rect = patches.Rectangle(
            (roi_pixel[0], roi_pixel[1]),
            roi_pixel[2],
            roi_pixel[3],
            linewidth=5,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            label=f'ROI: {self.roi_config["name"]}'
        )
        ax.add_patch(roi_rect)
        
        # 在右上角添加ROI标签（更醒目）- 使用LaTeX格式
        ax.text(
            roi_pixel[0] + roi_pixel[2] - 20,
            roi_pixel[1] + 20,
            r'$S_{\mathrm{ROI}}=70\%$',
            fontsize=20,
            color='red',
            fontweight='bold',
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='red', linewidth=3)
        )
        
        # 绘制检测结果
        for idx, det in enumerate(detections):
            if det['in_roi']:  # 只绘制ROI内的检测框
                bbox = det['bbox']
                cx_pixel, cy_pixel = det['centroid_pixel']
                cx_norm, cy_norm = det['centroid_norm']
                conf = det['confidence']
                
                # 绘制检测框（绿色，加粗）
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 绘制质心点（圆圈+十字架）
                # 外圈：橙色圆点
                ax.plot(cx_pixel, cy_pixel, 'o', 
                       markersize=20, 
                       markerfacecolor='orange', 
                       markeredgecolor='white',
                       markeredgewidth=3,
                       alpha=0.95)
                
                # 十字架标记（白色）
                cross_size = 12
                ax.plot([cx_pixel-cross_size, cx_pixel+cross_size], [cy_pixel, cy_pixel], 
                       'w-', linewidth=2.5)
                ax.plot([cx_pixel, cx_pixel], [cy_pixel-cross_size, cy_pixel+cross_size], 
                       'w-', linewidth=2.5)
                
                # 简化标注信息（只在检测框左上角显示置信度）
                label_text = f'Conf: {conf:.2f}'
                ax.text(
                    x1,
                    y1 - 5,
                    label_text,
                    fontsize=8,
                    color='white',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7, edgecolor='lime', linewidth=1)

                )
        
        # # 添加统计信息（已注释）
        # stats_text = (
        #     f"Total Standing: {results['total_standing']}\n"
        #     f"In ROI: {results['roi_standing']}\n"
        #     f"Filtered Out: {results['total_standing'] - results['roi_standing']}"
        # )
        # ax.text(
        #     0.02, 0.98,
        #     stats_text,
        #     transform=ax.transAxes,
        #     fontsize=14,
        #     color='white',
        #     fontweight='bold',
        #     verticalalignment='top',
        #     bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.7, edgecolor='white', linewidth=2)
        # )
        
        # # 添加ROI公式说明（显示像素坐标范围）（已注释）
        # x1_norm, y1_norm, x2_norm, y2_norm = self.roi_config['coords']
        # roi_formula = (
        #     f"ROI Region (pixels):\n"
        #     f"x: [{int(roi_pixel[0])} - {int(roi_pixel[0] + roi_pixel[2])}]\n"
        #     f"y: [{int(roi_pixel[1])} - {int(roi_pixel[1] + roi_pixel[3])}]\n"
        #     f"\n"
        #     f"Normalized: [{x1_norm:.2f}, {y1_norm:.2f}, {x2_norm:.2f}, {y2_norm:.2f}]"
        # )
        # ax.text(
        #     0.98, 0.98,
        #     roi_formula,
        #     transform=ax.transAxes,
        #     fontsize=12,
        #     color='white',
        #     fontweight='bold',
        #     verticalalignment='top',
        #     horizontalalignment='right',
        #     bbox=dict(boxstyle='round,pad=0.8', facecolor='darkred', alpha=0.7, edgecolor='red', linewidth=2)
        # )
        
        plt.tight_layout(pad=0)
        # 保存PNG版本（无白边）
        png_path = str(save_path).replace('.pdf', '.png') if str(save_path).endswith('.pdf') else save_path
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"\n✓ PNG可视化结果已保存: {png_path}")
        
        # 保存PDF版本（高清矢量格式，无白边）
        pdf_path = str(save_path).replace('.png', '.pdf') if str(save_path).endswith('.png') else save_path
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"✓ PDF可视化结果已保存: {pdf_path}")
        plt.close()


def main():
    """主函数"""
    # 配置参数
    IMAGE_PATH = "data/yolo_sam_w_15m/20240710_15m/20240710_15m_0338.JPG"
    MODEL_PATH = "YOLOV11/output/yolov11s_cattle_20260109_205959_800/weights/best.pt"
    CONFIDENCE_THRESHOLD = 0.7
    
    # 创建输出目录
    output_dir = Path("YOLOV11/experiments/roi_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名（PDF格式）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"roi_filter_visualization_{timestamp}.pdf"
    
    print("\n" + "="*80)
    print("ROI过滤可视化工具")
    print("="*80)
    print(f"输入图像: {IMAGE_PATH}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"ROI配置: center_70 [0.15, 0.20, 0.85, 0.80]")
    print(f"输出路径: {output_path}")
    
    # 初始化可视化器
    visualizer = ROIFilterVisualizer(MODEL_PATH, CONFIDENCE_THRESHOLD)
    
    # 处理图像
    results = visualizer.process_image(IMAGE_PATH)
    
    # 可视化结果
    visualizer.visualize_results(results, output_path)
    
    print("\n" + "="*80)
    print("处理完成！")
    print("="*80)


if __name__ == "__main__":
    main()
