import os
import json
import glob
from pathlib import Path

def check_and_delete_empty_annotations(folder_path):
    """
    检查文件夹下的所有labelme json标注文件，
    如果没有任何标签就删除json文件和对应的图像文件
    """
    # 获取所有json文件
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    deleted_count = 0
    
    for json_file in json_files:
        try:
            # 读取json文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有shapes（标注）
            shapes = data.get('shapes', [])
            
            if len(shapes) == 0:
                print(f"发现空标注文件: {json_file}")
                
                # 获取对应的图像文件路径
                json_path = Path(json_file)
                image_name = data.get('imagePath', '')
                
                # 如果json中有imagePath，使用该路径
                if image_name:
                    # 如果是相对路径，与json文件在同一目录
                    if not os.path.isabs(image_name):
                        image_file = os.path.join(json_path.parent, image_name)
                    else:
                        image_file = image_name
                else:
                    # 如果没有imagePath，根据json文件名推测图像文件名
                    base_name = json_path.stem
                    # 常见的图像文件扩展名
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                    image_file = None
                    
                    for ext in image_extensions:
                        potential_image = json_path.parent / (base_name + ext)
                        if potential_image.exists():
                            image_file = str(potential_image)
                            break
                
                # 删除json文件
                os.remove(json_file)
                print(f"已删除json文件: {json_file}")
                
                # 删除对应的图像文件（如果存在）
                if image_file and os.path.exists(image_file):
                    os.remove(image_file)
                    print(f"已删除图像文件: {image_file}")
                else:
                    print(f"未找到对应的图像文件: {image_file}")
                
                deleted_count += 1
                print("-" * 50)
                
        except json.JSONDecodeError:
            print(f"JSON文件格式错误，跳过: {json_file}")
        except Exception as e:
            print(f"处理文件时出错 {json_file}: {str(e)}")
    
    print(f"\n处理完成！共删除了 {deleted_count} 对文件")

def main():

    folder_path = r"D:\Datasets\CowDetection\renamed_images_few_nega"
    
    check_and_delete_empty_annotations(folder_path)

if __name__ == "__main__":
    main()