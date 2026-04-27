import json
import os
import shutil
from pathlib import Path

def check_and_fix_labels(folder_path, old_label, new_label):
    """
    检查文件夹下所有labelme的json文件，将指定标签转换为新标签
    只有修改后的文件才保存到新文件夹，原文件不变
    
    Args:
        folder_path: 包含json文件的文件夹路径
        old_label: 需要被替换的原标签
        new_label: 新的标签名称
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 创建输出文件夹
    output_folder = folder / "fixed_labels"
    output_folder.mkdir(exist_ok=True)
    print(f"输出文件夹: {output_folder}")
    
    # 创建记录文件
    record_filename = f"{old_label}_{new_label}.txt"
    record_path = output_folder / record_filename
    
    # 查找所有json文件
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"在 {folder_path} 中没有找到json文件")
        return
    
    print(f"找到 {len(json_files)} 个json文件")
    print(f"将标签 '{old_label}' 替换为 '{new_label}'")
    
    modified_count = 0
    total_labels_changed = 0
    modified_files = []  # 记录被修改的文件名
    
    for json_file in json_files:
        try:
            # 读取json文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有shapes字段
            if 'shapes' not in data:
                print(f"文件 {json_file.name}: 没有shapes字段，跳过")
                continue
            
            file_modified = False
            labels_changed_in_file = 0
            
            # 遍历所有标注
            for shape in data['shapes']:
                if 'label' in shape:
                    original_label = shape['label']
                    # 如果标签匹配要替换的标签，则修改
                    if original_label.lower() == old_label.lower():
                        shape['label'] = new_label
                        print(f"文件 {json_file.name}: 标签 '{original_label}' -> '{new_label}'")
                        file_modified = True
                        labels_changed_in_file += 1
            
            # 只保存修改过的文件
            if file_modified:
                output_path = output_folder / json_file.name
                
                # 将imageData字段设置为null
                if 'imageData' in data:
                    data['imageData'] = None
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                modified_count += 1
                total_labels_changed += labels_changed_in_file
                modified_files.append(json_file.name)
                print(f"已保存修改到: {output_path}")
            else:
                print(f"文件 {json_file.name}: 没有需要修改的标签，跳过")
        
        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
    
    # 将修改的文件名写入txt文件
    if modified_files:
        with open(record_path, 'w', encoding='utf-8') as f:
            f.write(f"Label change record: {old_label} -> {new_label}\n")
            f.write(f"Total modified files: {len(modified_files)}\n")
            f.write(f"Total labels changed: {total_labels_changed}\n")
            f.write("-" * 50 + "\n")
            for filename in modified_files:
                f.write(f"{filename}\n")
        print(f"修改记录已保存到: {record_path}")
    else:
        print(f"没有找到包含标签 '{old_label}' 的文件")
    
    print(f"\n处理完成!")
    print(f"修改的文件数: {modified_count}")
    print(f"修改的标签总数: {total_labels_changed}")
    if modified_count > 0:
        print(f"修改后的文件已保存到: {output_folder}")

if __name__ == "__main__":
    # 设置你的json文件所在的文件夹路径
    folder_path = r"D:\Datasets\CowDetection\sam2_hq_seg_large"
    
    # 设置要替换的标签
    print("请输入要替换的标签信息:")
    old_label = input("请输入原标签名称: ").strip()
    new_label = input("请输入新标签名称: ").strip()
    
    if not old_label or not new_label:
        print("标签名称不能为空!")
        exit()
    
    # 也可以直接在代码中设置，注释掉上面的输入部分
    # old_label = "cow"
    # new_label = "cattle"
    
    check_and_fix_labels(folder_path, old_label, new_label)