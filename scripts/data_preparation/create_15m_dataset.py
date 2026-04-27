#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_15m_dataset.py
第一步：从原始数据集中复制所有15m文件夹的图像，保持时间文件夹结构
第二步：从15m数据集中随机抽取500张，按8:1:1划分为训练、验证、测试集
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


class Dataset15mCreator:
    """15m数据集创建器"""
    
    def __init__(self, source_root: str, output_15m_dir: str, output_split_dir: str, seed: int = 42):
        """初始化
        
        Args:
            source_root: 源数据集根目录 (如: F:/Datasets/CattleDetection/images_original)
            output_15m_dir: 第一步输出目录 (所有15m图像)
            output_split_dir: 第二步输出目录 (划分后的500张)
            seed: 随机种子
        """
        self.source_root = Path(source_root)
        self.output_15m_dir = Path(output_15m_dir)
        self.output_split_dir = Path(output_split_dir)
        self.seed = seed
        
        # 排除20240911（与20240831重复）
        self.periods = [
            '20240710', '20240720', '20240731', '20240811', 
            '20240821', '20240831', '20240921', '20240928', 
            '20241005', '20241012', '20241019', '20241027'
        ]
        
        # 设置随机种子
        random.seed(seed)
    
    def step1_copy_all_15m_images(self) -> Dict[str, int]:
        """第一步：复制所有15m文件夹的图像，保持时间文件夹结构
        
        Returns:
            Dict[str, int]: 每个时期复制的图像数量统计
        """
        print("="*80)
        print("第一步：复制所有15m文件夹的图像（重命名添加时期标识）")
        print("="*80)
        print(f"源目录: {self.source_root}")
        print(f"目标目录: {self.output_15m_dir}")
        print(f"排除时期: 20240911 (与20240831重复)")
        print()
        
        # 创建输出目录
        self.output_15m_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {}
        total_copied = 0
        
        for period in self.periods:
            source_15m_path = self.source_root / period / "15m"
            target_period_path = self.output_15m_dir / period
            
            if not source_15m_path.exists():
                print(f"⚠️  跳过 {period}: 15m文件夹不存在")
                stats[period] = 0
                continue
            
            # 创建目标时期目录
            target_period_path.mkdir(parents=True, exist_ok=True)
            
            # 获取所有图像文件（递归）
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(source_15m_path.rglob(f'*{ext}'))
            
            # 去重（避免符号链接或其他原因导致的重复）
            image_files = list(set(image_files))
            
            # 复制图像并重命名（添加时期标识避免冲突）
            # 时期标识格式：0710_15m_原文件名
            period_prefix = period[4:] + "_15m_"  # 提取月日，如"0710_15m_"
            copied_count = 0
            
            for img_file in image_files:
                # 重命名：添加时期标识
                new_filename = period_prefix + img_file.name
                target_file = target_period_path / new_filename
                
                # 创建目标目录
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制文件（如果目标文件已存在则跳过）
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                    copied_count += 1
            
            # 重新统计实际复制的文件数
            actual_count = len(list(target_period_path.rglob('*.jpg'))) + \
                          len(list(target_period_path.rglob('*.jpeg'))) + \
                          len(list(target_period_path.rglob('*.png'))) + \
                          len(list(target_period_path.rglob('*.JPG'))) + \
                          len(list(target_period_path.rglob('*.JPEG'))) + \
                          len(list(target_period_path.rglob('*.PNG')))
            
            stats[period] = actual_count
            total_copied += copied_count
            print(f"✓ {period}: 复制 {copied_count} 张图像")
        
        print()
        print("="*80)
        print("第一步完成！")
        print("="*80)
        print(f"总共复制: {total_copied} 张图像")
        print()
        
        # 打印统计表
        print("各时期图像数量统计:")
        print("-" * 40)
        for period in self.periods:
            print(f"  {period}: {stats[period]:4d} 张")
        print("-" * 40)
        print(f"  总计:     {total_copied:4d} 张")
        print()
        
        return stats
    
    def step2_create_train_val_test_split(self, num_samples: int = 500, 
                                          train_ratio: float = 0.8,
                                          val_ratio: float = 0.1,
                                          test_ratio: float = 0.1):
        """第二步：从15m数据集随机抽取样本，按比例划分为训练、验证、测试集
        
        Args:
            num_samples: 抽取的样本数量
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        print("="*80)
        print("第二步：创建训练、验证、测试集划分")
        print("="*80)
        print(f"从 {self.output_15m_dir} 随机抽取 {num_samples} 张图像")
        print(f"划分比例: 训练集={train_ratio:.0%}, 验证集={val_ratio:.0%}, 测试集={test_ratio:.0%}")
        print()
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(self.output_15m_dir.rglob(f'*{ext}'))
        
        print(f"15m数据集总图像数: {len(all_images)} 张")
        
        if len(all_images) < num_samples:
            print(f"⚠️  警告: 可用图像数({len(all_images)})少于请求数量({num_samples})")
            num_samples = len(all_images)
            print(f"将使用所有 {num_samples} 张图像")
        
        # 随机抽取
        selected_images = random.sample(all_images, num_samples)
        
        # 计算各集合的数量
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val  # 剩余的分配给测试集
        
        print(f"\n数据集划分:")
        print(f"  训练集: {num_train} 张 ({num_train/num_samples:.1%})")
        print(f"  验证集: {num_val} 张 ({num_val/num_samples:.1%})")
        print(f"  测试集: {num_test} 张 ({num_test/num_samples:.1%})")
        print()
        
        # 划分数据
        train_images = selected_images[:num_train]
        val_images = selected_images[num_train:num_train+num_val]
        test_images = selected_images[num_train+num_val:]
        
        # 创建目标目录结构
        splits = {
            'train': (train_images, self.output_split_dir / 'train' / 'images'),
            'val': (val_images, self.output_split_dir / 'val' / 'images'),
            'test': (test_images, self.output_split_dir / 'test' / 'images')
        }
        
        # 复制文件
        for split_name, (images, target_dir) in splits.items():
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"复制 {split_name} 集...")
            
            for img_path in images:
                # 使用原始文件名
                target_file = target_dir / img_path.name
                shutil.copy2(img_path, target_file)
            
            print(f"✓ {split_name}: {len(images)} 张图像已复制")
        
        # 保存文件列表
        self._save_file_lists(splits)
        
        print()
        print("="*80)
        print("第二步完成！")
        print("="*80)
        print(f"输出目录: {self.output_split_dir}")
        print(f"  - train/images: {num_train} 张")
        print(f"  - val/images: {num_val} 张")
        print(f"  - test/images: {num_test} 张")
        print()
    
    def _save_file_lists(self, splits: Dict):
        """保存文件列表到txt文件"""
        list_dir = self.output_split_dir / 'file_lists'
        list_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, (images, _) in splits.items():
            list_file = list_dir / f'{split_name}_files.txt'
            with open(list_file, 'w', encoding='utf-8') as f:
                for img_path in sorted(images):
                    # 记录相对于15m数据集的路径
                    relative_path = img_path.relative_to(self.output_15m_dir)
                    f.write(f"{relative_path}\n")
            print(f"✓ 保存文件列表: {list_file}")
    
    def run_full_pipeline(self, num_samples: int = 500):
        """运行完整的两步流程
        
        Args:
            num_samples: 第二步抽取的样本数量
        """
        print("\n" + "🐄"*40)
        print("肉牛检测数据集创建流程")
        print("🐄"*40 + "\n")
        
        # 第一步：复制所有15m图像
        stats = self.step1_copy_all_15m_images()
        
        # 第二步：创建训练集划分
        self.step2_create_train_val_test_split(num_samples=num_samples)
        
        print("\n" + "🎉"*40)
        print("所有步骤完成！")
        print("🎉"*40 + "\n")


def main():
    parser = argparse.ArgumentParser(description='创建15m数据集并划分训练、验证、测试集')
    parser.add_argument('--source', type=str, 
                        default=r'F:\Datasets\CattleDetection\images_original',
                        help='源数据集根目录')
    parser.add_argument('--output-15m', type=str,
                        default=r'F:\Datasets\CattleDetection\15m_dataset',
                        help='第一步输出目录（所有15m图像）')
    parser.add_argument('--output-split', type=str,
                        default=r'F:\Datasets\CattleDetection\15m_500_split',
                        help='第二步输出目录（划分后的500张）')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='第二步抽取的样本数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--step', type=int, choices=[1, 2, 0], default=0,
                        help='执行步骤: 0=全部, 1=仅第一步, 2=仅第二步')
    
    args = parser.parse_args()
    
    creator = Dataset15mCreator(
        source_root=args.source,
        output_15m_dir=args.output_15m,
        output_split_dir=args.output_split,
        seed=args.seed
    )
    
    if args.step == 0:
        # 运行完整流程
        creator.run_full_pipeline(num_samples=args.num_samples)
    elif args.step == 1:
        # 仅执行第一步
        creator.step1_copy_all_15m_images()
    elif args.step == 2:
        # 仅执行第二步
        creator.step2_create_train_val_test_split(num_samples=args.num_samples)


if __name__ == '__main__':
    main()
