import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    解析文件名，提取时间、高度和图像号
    格式: 20240710_10m_0475.json
    
    Args:
        filename: 文件名
        
    Returns:
        (时间, 高度, 图像号) 元组
    """
    try:
        # 移除扩展名
        base_name = filename.replace('.json', '').replace('.jpg', '').replace('.png', '')
        
        # 按下划线分割
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            date = parts[0]  # 时间
            height = parts[1]  # 高度（如10m）
            image_num = parts[2]  # 图像号
            return date, height, image_num
        else:
            return "unknown", "unknown", "unknown"
    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return "unknown", "unknown", "unknown"

def analyze_labelme_annotations(folder_path: str) -> Dict:
    """
    分析文件夹中所有labelme JSON文件的类别分布
    
    Args:
        folder_path: 包含JSON文件的文件夹路径
        
    Returns:
        包含统计信息的字典
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return {}
    
    # 统计数据结构
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "total_annotations": 0,
        "overall_class_counts": defaultdict(int),  # 整个数据集的类别统计
        "date_height_stats": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),  # 时间-高度-类别统计
        "date_stats": defaultdict(lambda: defaultdict(int)),  # 时间-类别统计
        "height_stats": defaultdict(lambda: defaultdict(int)),  # 高度-类别统计
        "file_details": []  # 详细文件信息
    }
    
    # 查找所有JSON文件
    json_files = list(folder.glob("*.json"))
    stats["total_files"] = len(json_files)
    
    if not json_files:
        print(f"在 {folder_path} 中没有找到JSON文件")
        return stats
    
    print(f"找到 {len(json_files)} 个JSON文件，开始分析...")
    
    for json_file in json_files:
        try:
            # 解析文件名
            date, height, image_num = parse_filename(json_file.name)
            
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats["processed_files"] += 1
            
            # 检查是否为labelme格式
            if 'shapes' not in data:
                print(f"跳过非labelme格式文件: {json_file.name}")
                continue
            
            # 统计当前文件的类别
            file_class_counts = defaultdict(int)
            
            for shape in data['shapes']:
                if shape.get('shape_type') == 'rectangle' and 'label' in shape:
                    label = shape['label']
                    
                    # 更新各种统计
                    stats["overall_class_counts"][label] += 1
                    stats["date_height_stats"][date][height][label] += 1
                    stats["date_stats"][date][label] += 1
                    stats["height_stats"][height][label] += 1
                    file_class_counts[label] += 1
                    stats["total_annotations"] += 1
            
            # 记录文件详细信息
            stats["file_details"].append({
                "filename": json_file.name,
                "date": date,
                "height": height,
                "image_num": image_num,
                "class_counts": dict(file_class_counts),
                "total_objects": sum(file_class_counts.values())
            })
            
        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
            continue
    
    return stats

def print_statistics(stats: Dict):
    """
    打印统计结果
    
    Args:
        stats: 统计数据字典
    """
    print("\n" + "="*80)
    print("LABELME 标注类别分布统计报告")
    print("="*80)
    
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理文件数: {stats['processed_files']}")
    print(f"总标注数: {stats['total_annotations']}")
    
    # 1. 整个数据集的类别统计
    print(f"\n{'='*60}")
    print("1. 整个数据集类别分布:")
    print(f"{'='*60}")
    
    if stats['overall_class_counts']:
        for label, count in sorted(stats['overall_class_counts'].items()):
            percentage = (count / stats['total_annotations']) * 100 if stats['total_annotations'] > 0 else 0
            print(f"  {label}: {count} 个 ({percentage:.2f}%)")
    else:
        print("  未找到任何标注")
    
    # 2. 按时间分组的统计
    print(f"\n{'='*60}")
    print("2. 按时间分组的类别分布:")
    print(f"{'='*60}")
    
    for date in sorted(stats['date_stats'].keys()):
        print(f"\n时间: {date}")
        print("-" * 40)
        date_total = sum(stats['date_stats'][date].values())
        for label, count in sorted(stats['date_stats'][date].items()):
            percentage = (count / date_total) * 100 if date_total > 0 else 0
            print(f"  {label}: {count} 个 ({percentage:.2f}%)")
        print(f"  小计: {date_total} 个标注")
    
    # 3. 按高度分组的统计
    print(f"\n{'='*60}")
    print("3. 按高度分组的类别分布:")
    print(f"{'='*60}")
    
    for height in sorted(stats['height_stats'].keys()):
        print(f"\n高度: {height}")
        print("-" * 40)
        height_total = sum(stats['height_stats'][height].values())
        for label, count in sorted(stats['height_stats'][height].items()):
            percentage = (count / height_total) * 100 if height_total > 0 else 0
            print(f"  {label}: {count} 个 ({percentage:.2f}%)")
        print(f"  小计: {height_total} 个标注")
    
    # 4. 按时间-高度组合的详细统计
    print(f"\n{'='*60}")
    print("4. 按时间-高度组合的详细分布:")
    print(f"{'='*60}")
    
    for date in sorted(stats['date_height_stats'].keys()):
        print(f"\n时间: {date}")
        print("-" * 50)
        
        for height in sorted(stats['date_height_stats'][date].keys()):
            print(f"  高度 {height}:")
            combo_total = sum(stats['date_height_stats'][date][height].values())
            
            for label, count in sorted(stats['date_height_stats'][date][height].items()):
                percentage = (count / combo_total) * 100 if combo_total > 0 else 0
                print(f"    {label}: {count} 个 ({percentage:.2f}%)")
            
            print(f"    小计: {combo_total} 个标注")

def save_to_csv(stats: Dict, output_dir: str):
    """
    将统计结果保存为CSV文件
    
    Args:
        stats: 统计数据字典
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 保存整体类别统计
    overall_df = pd.DataFrame([
        {"类别": label, "数量": count, "百分比": f"{(count/stats['total_annotations'])*100:.2f}%"}
        for label, count in sorted(stats['overall_class_counts'].items())
    ])
    overall_df.to_csv(output_path / "overall_class_distribution.csv", index=False, encoding='utf-8-sig')
    
    # 2. 保存按时间-高度组合的详细统计
    detail_rows = []
    for date in stats['date_height_stats']:
        for height in stats['date_height_stats'][date]:
            for label, count in stats['date_height_stats'][date][height].items():
                detail_rows.append({
                    "时间": date,
                    "高度": height,
                    "类别": label,
                    "数量": count
                })
    
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(output_path / "date_height_class_distribution.csv", index=False, encoding='utf-8-sig')
    
    # 3. 保存文件详细信息
    file_detail_rows = []
    for file_info in stats['file_details']:
        for label, count in file_info['class_counts'].items():
            file_detail_rows.append({
                "文件名": file_info['filename'],
                "时间": file_info['date'],
                "高度": file_info['height'],
                "图像号": file_info['image_num'],
                "类别": label,
                "数量": count
            })
    
    if file_detail_rows:
        file_detail_df = pd.DataFrame(file_detail_rows)
        file_detail_df.to_csv(output_path / "file_detail_distribution.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n统计结果已保存到: {output_path}")
    print("- overall_class_distribution.csv: 整体类别分布")
    print("- date_height_class_distribution.csv: 时间-高度组合分布")
    print("- file_detail_distribution.csv: 文件详细信息")

def main():
    """主函数"""
    # 设置数据路径
    data_path = r"D:\Datasets\CowDetection\renamed_images_few_nega"
    
    print(f"开始分析数据集: {data_path}")
    print("文件命名格式: 20240710_10m_0475 (时间_高度_图像号)")
    print("-" * 80)
    
    # 执行分析
    stats = analyze_labelme_annotations(data_path)
    
    if not stats:
        print("分析失败或没有找到数据")
        return
    
    # 打印统计结果
    print_statistics(stats)
    
    # 保存到CSV文件
    output_dir = os.path.join(data_path, "statistics_output")
    save_to_csv(stats, output_dir)

if __name__ == "__main__":
    main()