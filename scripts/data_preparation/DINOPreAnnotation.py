import os
import json
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import shutil
from tqdm import tqdm
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image

class CattleDatasetProcessor:
    def __init__(self, text_prompt, dataset_root: str, model_name: str = "IDEA-Research/grounding-dino-base"):
        """
        初始化处理器
        
        Args:
            dataset_root: 图像文件夹路径
            model_name: DINO模型名称，默认使用grounding-dino-base
        """
        self.dataset_root = Path(dataset_root)
        
        # 初始化DINO模型和处理器
        print(f"正在加载DINO模型: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)
        
        # 设置GPU加速
        if torch.cuda.is_available():
            self.model.to('cuda')
            print(f"使用GPU加速，显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("GPU不可用，使用CPU处理")
        
        # 设置文本提示词（用于目标检测）
        # self.text_prompt = "cow. cattle. bull. animal."  # DINO模型使用的文本提示
        self.text_prompt = text_prompt
        
        
        # 创建输出目录
        self.output_dir = self.dataset_root / "processed_annotations"
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.json_dir = self.output_dir / "json_results"
        self.txt_dir = self.output_dir / "txt_results"
        self.xml_dir = self.output_dir / "xml_results"
        self.visualized_dir = self.output_dir / "visualized_results"  # 可视化结果目录
        
        for dir_path in [self.json_dir, self.txt_dir, self.xml_dir, self.visualized_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 统一类别设置 - 所有检测目标都标记为cattle
        self.unified_class_name = 'cattle'
        self.unified_class_id = 0  # 统一使用0作为类别ID
        
        # 颜色映射（用于可视化）
        self.class_color = (0, 255, 0)  # 绿色
    
    def scan_dataset_structure(self) -> List[Dict]:
        """
        扫描文件夹，返回所有图像路径信息
        
        Returns:
            包含图像路径信息的列表
        """
        image_paths = []
        
        print(f"处理文件夹: {self.dataset_root}")
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 递归搜索所有图像文件
        for img_file in self.dataset_root.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                image_paths.append({
                    'image_path': img_file,
                    'relative_path': img_file.relative_to(self.dataset_root)
                })
        
        print(f"总共找到 {len(image_paths)} 张图像")
        return image_paths
    
    def detect_targets(self, image_path: Path, conf_threshold: float = 0.1) -> List[Dict]:
        """
        使用DINO检测图像中的所有目标
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        try:
            # 读取图像
            image = Image.open(image_path).convert("RGB")
            
            print(f"检测图像: {image_path.name}")
            
            # 处理图像和文本
            inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt")
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # 进行推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 处理结果
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            if torch.cuda.is_available():
                target_sizes = target_sizes.to('cuda')
            
            # 修复：正确访问input_ids
            results = self.processor.post_process_grounded_object_detection(
                outputs, 
                input_ids=inputs["input_ids"],  # 修改这里，使用字典访问方式
                box_threshold=conf_threshold,
                text_threshold=0.3,
                target_sizes=target_sizes
            )[0]
            
            detections = []
            
            if len(results["scores"]) > 0:
                print(f"  原始检测框数量: {len(results['scores'])}")
                
                for i, (score, label, box) in enumerate(zip(
                    results["scores"], results["labels"], results["boxes"]
                )):
                    confidence = float(score)
                    class_name = str(label)  # Grounding DINO返回的是文本标签
                    
                    # 打印所有检测结果用于调试
                    print(f"    检测 {i+1}: 检测标签='{class_name}', 置信度={confidence:.3f}")
                    
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.cpu().numpy()
                    
                    # 所有检测目标都统一标记为cattle
                    detection = {
                        'class_id': self.unified_class_id,  # 统一使用0
                        'class_name': self.unified_class_name,  # 统一使用cattle
                        'original_label': class_name,  # 保留原始检测标签用于调试
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
                    print(f"      保存为: {self.unified_class_name}, 置信度={confidence:.3f}")
            else:
                print(f"  未检测到任何目标")
            
            print(f"  最终保留检测数: {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"检测图像 {image_path.name} 时出错: {str(e)}")
            # 添加调试信息
            if 'inputs' in locals():
                print(f"  调试信息: inputs keys = {list(inputs.keys())}")
            return []
    
    def visualize_detections(self, image_path: Path, detections: List[Dict], output_path: Path):
        """
        在图像上绘制检测框并保存（无论是否有检测结果都保存图像）
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果
            output_path: 输出图像路径
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 如果有检测结果，绘制检测框
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                original_label = detection['original_label']
                
                # 使用统一颜色
                color = self.class_color
                
                # 绘制边界框（加粗线条）
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # 线条粗细从2改为4
                
                # 绘制标签（显示统一类别名和原始检测标签）
                label = f"{class_name}: {confidence:.2f} (orig:{original_label})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]  # 字体大小从0.6改为0.8
                
                # 绘制标签背景
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 15), 
                             (int(x1) + label_size[0], int(y1)), color, -1)
                
                # 绘制标签文字
                cv2.putText(img, label, (int(x1), int(y1) - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存图像（无论是否有检测结果）
        cv2.imwrite(str(output_path), img)
    
        
    def save_json_results(self, image_info: Dict, detections: List[Dict], output_path: Path):
        """
        保存LabelMe格式的JSON检测结果
        """
        img = cv2.imread(str(image_info['image_path']))
        height, width = img.shape[:2]
        
        # LabelMe格式
        result = {
            "version": "5.8.3",  # LabelMe版本
            "flags": {},
            "shapes": [],
            "imagePath": image_info['image_path'].name,
            "imageData": None,  # LabelMe通常不存储图像数据
            "imageHeight": height,
            "imageWidth": width
        }
        
        # 添加检测框
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            shape = {
                "label": self.unified_class_name,
                "points": [
                    [float(x1), float(y1)],  # 左上角
                    [float(x2), float(y2)]   # 右下角
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            result["shapes"].append(shape)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_txt_results(self, image_info: Dict, detections: List[Dict], output_path: Path):
        """
        保存LabelImg支持的YOLO格式TXT检测结果
        """
        img = cv2.imread(str(image_info['image_path']))
        img_height, img_width = img.shape[:2]
        
        with open(output_path, 'w') as f:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # 转换为YOLO格式 (中心点x, 中心点y, 宽度, 高度)，归一化到[0,1]
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # 确保坐标在有效范围内
                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                # LabelImg YOLO格式: class_id center_x center_y width height
                f.write(f"{self.unified_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    def save_xml_results(self, image_info: Dict, detections: List[Dict], output_path: Path):
        """
        保存LabelImg支持的Pascal VOC格式XML检测结果
        """
        img = cv2.imread(str(image_info['image_path']))
        img_height, img_width, img_depth = img.shape
        
        # 创建标准Pascal VOC XML结构
        annotation = ET.Element('annotation')
        
        # 添加基本信息
        ET.SubElement(annotation, 'folder').text = str(image_info['image_path'].parent.name)
        ET.SubElement(annotation, 'filename').text = image_info['image_path'].name
        ET.SubElement(annotation, 'path').text = str(image_info['image_path'])
        
        # 添加数据源信息（Pascal VOC标准格式）
        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        
        # 添加图像尺寸信息
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(img_width)
        ET.SubElement(size, 'height').text = str(img_height)
        ET.SubElement(size, 'depth').text = str(img_depth)
        
        # 添加分割信息（LabelImg需要）
        ET.SubElement(annotation, 'segmented').text = '0'
        
        # 添加检测结果
        for detection in detections:
            obj = ET.SubElement(annotation, 'object')
            
            # 使用统一的类别名称
            ET.SubElement(obj, 'name').text = self.unified_class_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            # 添加边界框
            bndbox = ET.SubElement(obj, 'bndbox')
            x1, y1, x2, y2 = detection['bbox']
            
            # 确保坐标为整数且在图像范围内
            xmin = max(0, min(img_width - 1, int(round(x1))))
            ymin = max(0, min(img_height - 1, int(round(y1))))
            xmax = max(0, min(img_width - 1, int(round(x2))))
            ymax = max(0, min(img_height - 1, int(round(y2))))
            
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        
        # 格式化XML并保存
        from xml.dom import minidom
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  ")[23:])  # 去掉第一行的XML声明重复
    
    def process_all_images(self, confidence_threshold: float = 0.3, dino_conf_threshold: float = 0.1):
        """
        处理所有图像
        
        Args:
            confidence_threshold: 过滤置信度阈值（用于最终保存）
            dino_conf_threshold: DINO模型内部置信度阈值（用于检测）
        """
        # 扫描数据集
        image_paths = self.scan_dataset_structure()
        
        # 创建统计信息
        stats = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'total_detections': 0,
            'images_with_detections': 0,
            'original_label_counts': {}  # 统计原始标签分布
        }
        
        print(f"开始处理 {len(image_paths)} 张图像...")
        print(f"所有检测目标将统一标记为: {self.unified_class_name}")
        print(f"使用文本提示: {self.text_prompt}")
        print(f"DINO内部置信度阈值: {dino_conf_threshold}")
        print(f"过滤置信度阈值: {confidence_threshold}")
        
        # 处理每张图像
        for image_info in tqdm(image_paths, desc="处理图像"):
            try:
                # 检测目标
                detections = self.detect_targets(image_info['image_path'], dino_conf_threshold)
                
                # 过滤低置信度的检测
                filtered_detections = [
                    d for d in detections 
                    if d['confidence'] >= confidence_threshold
                ]
                
                # 更新统计信息
                stats['processed_images'] += 1
                stats['total_detections'] += len(filtered_detections)
                if filtered_detections:
                    stats['images_with_detections'] += 1
                
                # 统计原始标签分布
                for detection in filtered_detections:
                    orig_label = detection['original_label']
                    stats['original_label_counts'][orig_label] = stats['original_label_counts'].get(orig_label, 0) + 1
                
                # 生成输出文件名
                base_name = image_info['image_path'].stem
                
                # 保存结果（无论是否有检测结果都保存）
                # 保存JSON
                json_path = self.json_dir / f"{base_name}.json"
                self.save_json_results(image_info, filtered_detections, json_path)
                
                # 保存TXT
                txt_path = self.txt_dir / f"{base_name}.txt"
                self.save_txt_results(image_info, filtered_detections, txt_path)
                
                # 保存XML（LabelImg格式）
                xml_path = self.xml_dir / f"{base_name}.xml"
                self.save_xml_results(image_info, filtered_detections, xml_path)
                
                # 生成可视化结果（无论是否有检测结果都保存图像）
                if filtered_detections:
                    vis_path = self.visualized_dir / f"{base_name}_detected.jpg"
                else:
                    vis_path = self.visualized_dir / f"{base_name}_no_detection.jpg"
                self.visualize_detections(image_info['image_path'], filtered_detections, vis_path)
                
            except Exception as e:
                print(f"处理图像 {image_info['image_path']} 时出错: {str(e)}")
                continue
        
        # 打印统计信息
        print("\n处理完成!")
        print(f"总图像数: {stats['total_images']}")
        print(f"成功处理: {stats['processed_images']}")
        print(f"有检测结果的图像: {stats['images_with_detections']}")
        print(f"无检测结果的图像: {stats['total_images'] - stats['images_with_detections']}")
        print(f"总检测数: {stats['total_detections']}")
        print(f"平均每张图像检测数: {stats['total_detections'] / max(stats['images_with_detections'], 1):.2f}")
        
        print(f"\n所有目标已统一标记为: {self.unified_class_name}")
        print("\n原始标签分布统计:")
        for label, count in sorted(stats['original_label_counts'].items()):
            print(f"  原始标签 '{label}': {count} 个检测框")
        
        # 生成处理报告
        self.generate_report(stats)
    
    def generate_report(self, stats: Dict):
        """
        生成处理报告
        """
        report_path = self.output_dir / "processing_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("目标检测数据集处理报告 (Grounding DINO模型)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总图像数: {stats['total_images']}\n")
            f.write(f"成功处理: {stats['processed_images']}\n")
            f.write(f"有检测结果的图像: {stats['images_with_detections']}\n")
            f.write(f"无检测结果的图像: {stats['total_images'] - stats['images_with_detections']}\n")
            f.write(f"总检测数: {stats['total_detections']}\n")
            f.write(f"平均每张图像检测数: {stats['total_detections'] / max(stats['images_with_detections'], 1):.2f}\n\n")
            
            f.write(f"统一类别标记: {self.unified_class_name}\n")
            f.write(f"使用文本提示: {self.text_prompt}\n\n")
            
            f.write("原始标签分布统计:\n")
            for label, count in sorted(stats['original_label_counts'].items()):
                f.write(f"  原始标签 '{label}': {count} 个检测框\n")
            f.write("\n")
            
            f.write("输出文件说明:\n")
            f.write("- json_results/: JSON格式检测结果（包含原始标签信息）\n")
            f.write("- txt_results/: YOLO格式检测结果（所有类别统一为0）\n")
            f.write("- xml_results/: Pascal VOC格式检测结果（所有类别统一为cattle）\n")
            f.write("- visualized_results/: 检测结果可视化图像（所有图像都保存）\n")
            f.write("  * *_detected.jpg: 有检测结果的图像\n")
            f.write("  * *_no_detection.jpg: 无检测结果的图像\n")
        
        print(f"处理报告已保存至: {report_path}")


def main():
    """
    主函数
    """
    # 设置图像文件夹路径
    image_folder = "/home/u11719919/project/CowDetection/DataAnnotation/data/renamed_images"  # 修改为你的图像文件夹路径
    
    # 创建处理器 (使用Grounding DINO模型)
    text_prompt = "cow. cattle. animal."  # 更具体的提示词
    # text_prompt = "animal"  # 更具体的提示词

    processor = CattleDatasetProcessor(text_prompt, image_folder)
    
    # 处理所有图像 - 使用更低的置信度阈值
    processor.process_all_images(
        confidence_threshold=0.3,      # 最终保存的置信度阈值
        dino_conf_threshold=0.05       # DINO模型内部置信度阈值，设置很低以获取更多原始检测
    )
    
    print("\n下一步操作指南:")
    print("1. 检查 processed_annotations/visualized_results/ 查看所有检测结果")
    print("2. 检查 processed_annotations/xml_results/ 目录中的XML文件")
    print("3. 使用LabelImg打开原始图像文件夹")
    print("4. 设置LabelImg的标注保存路径为 processed_annotations/xml_results/")
    print("5. 手动删除或修正错误的目标标注")
    print("6. 对遗漏的牛进行补充标注")

if __name__ == "__main__":
    main()