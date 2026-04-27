import os
import json
import math

# 设置阈值，认为坐标"几乎相同"的像素距离
THRESHOLD = 10.0

def are_points_close(p1, p2, threshold=THRESHOLD):
    """检查两个点是否足够接近"""
    return abs(p1[0] - p2[0]) < threshold and abs(p1[1] - p2[1]) < threshold

def are_boxes_similar(box1_points, box2_points, threshold=THRESHOLD):
    """检查两个标注框是否相似（位置接近）"""
    if len(box1_points) != 2 or len(box2_points) != 2:
        return False
    
    # 标准化坐标顺序
    def normalize_box(points):
        x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
        x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])
        return [(x1, y1), (x2, y2)]
    
    norm_box1 = normalize_box(box1_points)
    norm_box2 = normalize_box(box2_points)
    
    # 检查两个角点是否都足够接近
    return (are_points_close(norm_box1[0], norm_box2[0], threshold) and
            are_points_close(norm_box1[1], norm_box2[1], threshold))

def remove_duplicate_annotations(input_dir, threshold=THRESHOLD):
    """
    检查指定文件夹下所有JSON文件中的重复和近似重复标注框并去除
    
    Args:
        input_dir (str): 包含JSON文件的文件夹路径
        threshold (float): 判断坐标接近的阈值（像素）
    
    Returns:
        dict: 统计信息，包括处理的文件数、发现的重复框数等
    """
    stats = {
        "processed_files": 0,
        "files_with_duplicates": 0,
        "total_duplicates_removed": 0,
        "duplicate_details": []
    }
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                stats["processed_files"] += 1
                
                # 检查是否为labelme格式的JSON文件
                if "shapes" not in data:
                    print(f"跳过非labelme格式文件: {filename}")
                    continue
                
                original_count = len(data["shapes"])
                shapes = data["shapes"]
                
                # 找出重复或近似重复的标注框
                duplicates_indices = set()
                rectangles = []  # 存储矩形标注框的信息
                
                # 首先收集所有矩形标注框
                for i, shape in enumerate(shapes):
                    if shape.get("shape_type") == "rectangle" and "points" in shape:
                        if len(shape["points"]) == 2:
                            rectangles.append((i, shape))
                
                # 比较每对矩形标注框
                for i in range(len(rectangles)):
                    for j in range(i + 1, len(rectangles)):
                        idx1, shape1 = rectangles[i]
                        idx2, shape2 = rectangles[j]
                        
                        # 检查是否为相同标签且位置接近
                        if (shape1.get("label", "") == shape2.get("label", "") and
                            are_boxes_similar(shape1["points"], shape2["points"], threshold)):
                            
                            # 标记第二个为重复（保留第一个）
                            duplicates_indices.add(idx2)
                            
                            # 输出详细信息
                            box1 = shape1["points"]
                            box2 = shape2["points"]
                            print(f"发现近似重复标注框在 {filename}:")
                            print(f"  框1 (索引{idx1}): {box1}, 标签: {shape1.get('label', '')}")
                            print(f"  框2 (索引{idx2}): {box2}, 标签: {shape2.get('label', '')}")
                
                # 移除重复的标注框
                unique_shapes = [shape for i, shape in enumerate(shapes) if i not in duplicates_indices]
                duplicates_in_file = len(duplicates_indices)
                
                # 如果发现重复，更新文件
                if duplicates_in_file > 0:
                    stats["files_with_duplicates"] += 1
                    stats["total_duplicates_removed"] += duplicates_in_file
                    stats["duplicate_details"].append({
                        "filename": filename,
                        "original_count": original_count,
                        "duplicates_removed": duplicates_in_file,
                        "final_count": len(unique_shapes)
                    })
                    
                    # 更新数据并保存
                    data["shapes"] = unique_shapes
                    
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    print(f"✓ 文件 {filename}: 移除了 {duplicates_in_file} 个重复/近似重复标注框 "
                          f"({original_count} -> {len(unique_shapes)})")
                else:
                    print(f"✓ 文件 {filename}: 没有发现重复或近似重复标注框")
                    
            except json.JSONDecodeError:
                print(f"错误: 无法解析JSON文件 {filename}")
            except Exception as e:
                print(f"错误: 处理文件 {filename} 时发生异常: {str(e)}")
    
    return stats

def print_summary(stats):
    """打印处理结果摘要"""
    print("\n" + "="*50)
    print("重复标注框检查完成!")
    print("="*50)
    print(f"处理的文件总数: {stats['processed_files']}")
    print(f"包含重复标注框的文件数: {stats['files_with_duplicates']}")
    print(f"移除的重复标注框总数: {stats['total_duplicates_removed']}")
    
    if stats["duplicate_details"]:
        print("\n详细信息:")
        print("-"*50)
        for detail in stats["duplicate_details"]:
            print(f"文件: {detail['filename']}")
            print(f"  原始标注框数: {detail['original_count']}")
            print(f"  移除重复数: {detail['duplicates_removed']}")
            print(f"  最终标注框数: {detail['final_count']}")
            print()

# 主程序
if __name__ == "__main__":
    input_dir = "D:\\Datasets\\CowDetection\\renamed_images"
    
    print(f"开始检查文件夹: {input_dir}")
    print(f"使用阈值: {THRESHOLD} 像素")
    print("正在扫描JSON文件中的重复和近似重复标注框...")
    print("-"*50)
    
    # 执行重复标注框检查和移除
    stats = remove_duplicate_annotations(input_dir, THRESHOLD)
    
    # 打印结果摘要
    print_summary(stats)