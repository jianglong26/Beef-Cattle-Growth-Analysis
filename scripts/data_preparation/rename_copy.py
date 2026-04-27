import os
import shutil
import re

def rename_and_copy_images(source_folder, target_folder):
    # 源文件夹和目标文件夹路径
    # source_folder = r"D:\Datasets\CowDetection\images\20240821\15m"
    # target_folder = r"D:\Datasets\CowDetection\yolo_sam_w_15m\20240821_15m"
    
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取源文件夹中的所有文件
    if not os.path.exists(source_folder):
        print(f"源文件夹不存在: {source_folder}")
        return
    
    files = os.listdir(source_folder)
    
    # 过滤出符合格式的图片文件
    pattern = re.compile(r"DJI_(\d+)\.jpg", re.IGNORECASE)
    
    copied_count = 0
    skipped_count = 0
    
    for file in files:
        match = pattern.match(file)
        if match:
            # 提取数字部分
            number = match.group(1)
            
            # 构造新的文件名
            new_filename = f"20240821_15m_{number}.jpg"
            
            # 源文件完整路径
            source_path = os.path.join(source_folder, file)
            
            # 目标文件完整路径
            target_path = os.path.join(target_folder, new_filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(target_path):
                print(f"跳过 (文件已存在): {new_filename}")
                skipped_count += 1
                continue
            
            try:
                # 复制文件
                shutil.copy2(source_path, target_path)
                print(f"已复制: {file} -> {new_filename}")
                copied_count += 1
            except Exception as e:
                print(f"复制失败 {file}: {e}")
    
    print(f"\n处理完成！")
    print(f"成功复制: {copied_count} 个文件")
    print(f"跳过文件: {skipped_count} 个文件")

if __name__ == "__main__":
    # 源文件夹和目标文件夹路径
    source_folder = r"D:\Datasets\CowDetection\images\20240821\15m"
    target_folder = r"D:\Datasets\CowDetection\yolo_sam_w_15m\20240821_15m"
    rename_and_copy_images(source_folder, target_folder)