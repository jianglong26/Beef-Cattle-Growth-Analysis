#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_subset_dataset.py
从总数据集中随机抽取指定数量的图像及标签，创建子数据集
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse


class DatasetSubsetCreator:
    """数据集子集创建器"""
    
    def __init__(self, source_dir: str, target_dir: str, num_samples: int, seed: int = 42):
        """初始化
        
        Args:
            source_dir: 源数据集目录路径
            target_dir: 目标数据集目录路径
            num_samples: 要抽取的样本数量
            seed: 随机种子，保证可复现性
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_samples = num_samples
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        
        # 定义源路径
        self.source_images_dir = self.source_dir / "images"
        self.source_labels_dir = self.source_dir / "labels"
        self.source_classes_file = self.source_dir / "classes.txt"
        
        # 定义目标路径
        self.target_images_dir = self.target_dir / "images"
        self.target_labels_dir = self.target_dir / "labels"
        self.target_classes_file = self.target_dir / "classes.txt"
        
    def validate_source_dataset(self) -> bool:
        """验证源数据集是否有效
        
        Returns:
            bool: 数据集是否有效
        """
        print("="*60)
        print("验证源数据集...")
        print("="*60)
        
        # 检查目录是否存在
        if not self.source_dir.exists():
            print(f"❌ 错误：源数据集目录不存在: {self.source_dir}")
            return False
        
        if not self.source_images_dir.exists():
            print(f"❌ 错误：源图像目录不存在: {self.source_images_dir}")
            return False
        
        if not self.source_labels_dir.exists():
            print(f"❌ 错误：源标签目录不存在: {self.source_labels_dir}")
            return False
        
        print(f"✓ 源数据集目录: {self.source_dir}")
        print(f"✓ 源图像目录: {self.source_images_dir}")
        print(f"✓ 源标签目录: {self.source_labels_dir}")
        
        return True
    
    def get_image_files(self) -> List[Path]:
        """获取所有图像文件
        
        Returns:
            List[Path]: 图像文件路径列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_images_dir.glob(f'*{ext}'))
        
        # 按文件名排序，保证可复现性
        image_files = sorted(image_files)
        
        return image_files
    
    def get_label_path(self, image_path: Path) -> Path:
        """根据图像路径获取对应的标签路径
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Path: 对应的标签文件路径
        """
        label_name = image_path.stem + '.txt'
        label_path = self.source_labels_dir / label_name
        return label_path
    
    def create_target_directories(self):
        """创建目标目录结构"""
        print("\n" + "="*60)
        print("创建目标数据集目录...")
        print("="*60)
        
        # 如果目标目录已存在，询问是否覆盖
        if self.target_dir.exists():
            response = input(f"⚠️  目标目录已存在: {self.target_dir}\n是否删除并重新创建? (y/n): ")
            if response.lower() == 'y':
                print(f"删除现有目录: {self.target_dir}")
                shutil.rmtree(self.target_dir)
            else:
                print("❌ 操作已取消")
                return False
        
        # 创建目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.target_images_dir.mkdir(parents=True, exist_ok=True)
        self.target_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 创建目标目录: {self.target_dir}")
        print(f"✓ 创建图像目录: {self.target_images_dir}")
        print(f"✓ 创建标签目录: {self.target_labels_dir}")
        
        return True
    
    def copy_files(self, selected_images: List[Path]) -> Tuple[int, int]:
        """复制选中的图像和标签文件
        
        Args:
            selected_images: 选中的图像文件列表
            
        Returns:
            Tuple[int, int]: (成功复制的图像数, 成功复制的标签数)
        """
        print("\n" + "="*60)
        print("复制文件...")
        print("="*60)
        
        copied_images = 0
        copied_labels = 0
        missing_labels = []
        
        for i, image_path in enumerate(selected_images, 1):
            # 显示进度
            if i % 50 == 0 or i == len(selected_images):
                print(f"进度: {i}/{len(selected_images)} ({i/len(selected_images)*100:.1f}%)")
            
            # 复制图像
            target_image_path = self.target_images_dir / image_path.name
            try:
                shutil.copy2(image_path, target_image_path)
                copied_images += 1
            except Exception as e:
                print(f"❌ 复制图像失败: {image_path.name} - {e}")
                continue
            
            # 复制对应的标签
            label_path = self.get_label_path(image_path)
            if label_path.exists():
                target_label_path = self.target_labels_dir / label_path.name
                try:
                    shutil.copy2(label_path, target_label_path)
                    copied_labels += 1
                except Exception as e:
                    print(f"❌ 复制标签失败: {label_path.name} - {e}")
            else:
                missing_labels.append(image_path.name)
        
        # 报告缺失的标签
        if missing_labels:
            print(f"\n⚠️  警告: {len(missing_labels)} 个图像没有对应的标签文件:")
            for name in missing_labels[:5]:  # 只显示前5个
                print(f"  - {name}")
            if len(missing_labels) > 5:
                print(f"  ... 还有 {len(missing_labels)-5} 个")
        
        return copied_images, copied_labels
    
    def copy_classes_file(self):
        """复制classes.txt文件"""
        if self.source_classes_file.exists():
            shutil.copy2(self.source_classes_file, self.target_classes_file)
            print(f"✓ 复制classes.txt文件")
        else:
            print(f"⚠️  警告: classes.txt文件不存在于源目录")
    
    def create_subset(self):
        """创建子数据集的主流程"""
        print("\n" + "="*60)
        print("开始创建子数据集")
        print("="*60)
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        print(f"抽取数量: {self.num_samples}")
        print(f"随机种子: {self.seed}")
        print("="*60)
        
        # 1. 验证源数据集
        if not self.validate_source_dataset():
            return False
        
        # 2. 获取所有图像文件
        print("\n" + "="*60)
        print("读取图像文件列表...")
        print("="*60)
        
        all_images = self.get_image_files()
        print(f"✓ 找到 {len(all_images)} 张图像")
        
        if len(all_images) == 0:
            print("❌ 错误：源数据集中没有图像文件")
            return False
        
        # 3. 检查抽取数量是否合理
        if self.num_samples > len(all_images):
            print(f"⚠️  警告：请求抽取 {self.num_samples} 张，但只有 {len(all_images)} 张")
            print(f"将抽取所有 {len(all_images)} 张图像")
            self.num_samples = len(all_images)
        
        # 4. 随机选择图像
        print(f"\n随机选择 {self.num_samples} 张图像...")
        selected_images = random.sample(all_images, self.num_samples)
        print(f"✓ 已选择 {len(selected_images)} 张图像")
        
        # 5. 创建目标目录
        if not self.create_target_directories():
            return False
        
        # 6. 复制文件
        copied_images, copied_labels = self.copy_files(selected_images)
        
        # 7. 复制classes.txt
        self.copy_classes_file()
        
        # 8. 生成统计报告
        self.print_summary(len(all_images), copied_images, copied_labels)
        
        # 9. 保存选择的文件列表（用于记录）
        self.save_file_list(selected_images)
        
        print("\n" + "="*60)
        print("✓ 子数据集创建完成！")
        print("="*60)
        
        return True
    
    def print_summary(self, total_images: int, copied_images: int, copied_labels: int):
        """打印统计摘要
        
        Args:
            total_images: 总图像数
            copied_images: 已复制图像数
            copied_labels: 已复制标签数
        """
        print("\n" + "="*60)
        print("统计摘要")
        print("="*60)
        print(f"源数据集总图像数: {total_images}")
        print(f"目标抽取数量: {self.num_samples}")
        print(f"实际复制图像数: {copied_images}")
        print(f"实际复制标签数: {copied_labels}")
        print(f"标签匹配率: {copied_labels/copied_images*100:.1f}%")
        print("="*60)
    
    def save_file_list(self, selected_images: List[Path]):
        """保存选择的文件列表
        
        Args:
            selected_images: 选择的图像列表
        """
        list_file = self.target_dir / "selected_files.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write(f"# 子数据集文件列表\n")
            f.write(f"# 源目录: {self.source_dir}\n")
            f.write(f"# 创建时间: {Path(__file__).stat().st_mtime}\n")
            f.write(f"# 随机种子: {self.seed}\n")
            f.write(f"# 总数: {len(selected_images)}\n\n")
            
            for img_path in sorted(selected_images):
                f.write(f"{img_path.name}\n")
        
        print(f"✓ 保存文件列表: {list_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从总数据集中随机抽取子数据集'
    )
    parser.add_argument('--source', type=str, 
                       default='data/yolo_det_dataset_total_1997',
                       help='源数据集目录路径')
    parser.add_argument('--target', type=str,
                       default='data/yolo_det_dataset',
                       help='目标数据集目录路径')
    parser.add_argument('--num', type=int, default=500,
                       help='要抽取的样本数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，保证可复现性')
    
    args = parser.parse_args()
    
    try:
        # 创建子数据集创建器
        creator = DatasetSubsetCreator(
            source_dir=args.source,
            target_dir=args.target,
            num_samples=args.num,
            seed=args.seed
        )
        
        # 创建子数据集
        success = creator.create_subset()
        
        if success:
            print(f"\n✓ 成功创建子数据集: {args.target}")
            print(f"包含 {args.num} 张图像及对应标签")
            return 0
        else:
            print("\n❌ 创建子数据集失败")
            return 1
            
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
