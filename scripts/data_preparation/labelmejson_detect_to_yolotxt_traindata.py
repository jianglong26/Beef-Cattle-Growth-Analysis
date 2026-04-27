import json
import os
import shutil
from pathlib import Path
import random

def convert_labelme_to_yolo(source_dir, output_dir):
    """
    将 LabelMe 格式的 JSON 标注文件转换为 YOLO 格式，并复制图像文件
    
    Args:
        source_dir: 包含图像和 JSON 标注文件的源目录
        output_dir: YOLO 格式数据集输出目录
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建 YOLO 数据集目录结构
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 收集所有类别 - 先遍历所有 JSON 文件
    all_classes = set()
    json_files = list(source_path.glob('*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            for shape in labelme_data.get('shapes', []):
                all_classes.add(shape['label'])
        except Exception as e:
            print(f"读取 {json_file} 时出错: {e}")
            continue
    
    # 创建类别到索引的映射
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"检测到类别: {sorted(all_classes)}")
    
    # 保存类别到根目录下的 classes.txt
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in sorted(all_classes):
            f.write(f"{cls}\n")
    
    print(f"类别信息已保存到: {classes_file}")
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_path.glob(ext)))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像文件
    processed_count = 0
    # tqdm()
    # for idx, image_file in enumerate(tqdm image_files):
    for image_file in image_files:
        try:
            # 查找对应的 JSON 文件
            json_file = source_path / (image_file.stem + '.json')
            
            if not json_file.exists():
                print(f"警告: 找不到与 {image_file.name} 对应的 JSON 标注文件")
                continue
            
            # 读取 JSON 标注文件
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            # 获取图像尺寸
            image_width = labelme_data['imageWidth']
            image_height = labelme_data['imageHeight']
            
            # 复制图像文件到 images 目录
            target_image = images_dir / image_file.name
            shutil.copy2(image_file, target_image)
            
            # 准备对应的 YOLO 标注文件
            txt_filename = image_file.stem + '.txt'
            txt_path = labels_dir / txt_filename
            
            with open(txt_path, 'w') as f:
                for shape in labelme_data.get('shapes', []):
                    if shape['shape_type'] == 'rectangle':
                        label = shape['label']
                        class_idx = class_to_idx[label]
                        
                        # 获取矩形坐标点
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # 确保坐标顺序正确
                        x_min = min(x1, x2)
                        x_max = max(x1, x2)
                        y_min = min(y1, y2)
                        y_max = max(y1, y2)
                        
                        # 转换为 YOLO 格式 (归一化的中心点坐标和宽高)
                        x_center = (x_min + x_max) / 2 / image_width
                        y_center = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height
                        
                        # 写入 YOLO 格式的标注
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {image_file} 时出错: {e}")
            continue
    
    print(f"转换完成! 成功处理了 {processed_count} 个文件")
    print(f"数据集保存到: {output_path}")
    print(f"图像文件: {images_dir}")
    print(f"标注文件: {labels_dir}")
    print(f"类别文件: {classes_file}")

def split_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将 YOLO 格式数据集按比例划分为训练集、验证集和测试集
    
    Args:
        dataset_dir: 原始 YOLO 数据集目录
        output_dir: 划分后数据集的输出目录
        train_ratio: 训练集比例 (默认 0.8)
        val_ratio: 验证集比例 (默认 0.1)
        test_ratio: 测试集比例 (默认 0.1)
        random_seed: 随机种子，确保结果可重现
    """
    # 检查比例是否合理
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集和测试集的比例之和必须等于1")
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # 检查源数据集是否存在
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    classes_file = dataset_path / 'classes.txt'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"找不到 YOLO 数据集目录: {images_dir} 或 {labels_dir}")
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
        image_files.extend(list(images_dir.glob(ext)))
    
    if not image_files:
        raise FileNotFoundError(f"在 {images_dir} 中找不到任何图像文件")
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 设置随机种子
    random.seed(random_seed)
    random.shuffle(image_files)
    
    # 计算各个数据集的大小
    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count
    
    print(f"数据集划分:")
    print(f"  训练集: {train_count} 个文件 ({train_count/total_count:.1%})")
    print(f"  验证集: {val_count} 个文件 ({val_count/total_count:.1%})")
    print(f"  测试集: {test_count} 个文件 ({test_count/total_count:.1%})")
    
    # 划分文件列表
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # 创建输出目录结构
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, file_list in splits.items():
        if not file_list:  # 跳过空的划分
            continue
            
        split_images_dir = output_path / split_name / 'images'
        split_labels_dir = output_path / split_name / 'labels'
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        # 复制图像和标注文件
        for image_file in file_list:
            # 复制图像文件
            target_image = split_images_dir / image_file.name
            shutil.copy2(image_file, target_image)
            
            # 复制对应的标注文件
            label_file = labels_dir / (image_file.stem + '.txt')
            if label_file.exists():
                target_label = split_labels_dir / label_file.name
                shutil.copy2(label_file, target_label)
            else:
                print(f"警告: 找不到 {image_file.name} 对应的标注文件")
        
        print(f"{split_name} 数据集: {len(file_list)} 个文件已保存到 {output_path / split_name}")
    
    # 复制类别文件到根目录
    if classes_file.exists():
        target_classes = output_path / 'classes.txt'
        shutil.copy2(classes_file, target_classes)
        print(f"类别文件已复制到: {target_classes}")
    
    # 创建数据集配置文件 (YAML格式，用于YOLO训练)
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_path.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (optional, relative to 'path')

# Classes
nc: {len(open(classes_file).readlines()) if classes_file.exists() else 0}  # number of classes
names: 
"""
    
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        for i, class_name in enumerate(classes):
            yaml_content += f"  {i}: {class_name}\n"
    
    yaml_file = output_path / 'dataset.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已保存到: {yaml_file}")
    print(f"数据集划分完成! 输出目录: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 第一步: 转换 LabelMe 到 YOLO 格式
    source_dir = 'data/renamed_images'
    output_dir = 'data/yolo_det_dataset_500'
    # convert_labelme_to_yolo(source_dir, output_dir)
    
    # # 第二步: 划分数据集
    split_output_dir = 'data/yolodet_train_dataset'
    split_yolo_dataset(output_dir, split_output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)    