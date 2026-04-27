import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from datetime import timedelta
import re
import time


def extract_info_from_path(video_path, root_dir):
    """
    从视频路径中提取信息并生成命名前缀
    :param video_path: 视频文件完整路径
    :param root_dir: 根目录路径
    :return: naming_prefix (命名前缀)
    """
    video_path = Path(video_path)
    root_dir = Path(root_dir)
    
    # 获取相对于根目录的路径
    try:
        relative_path = video_path.relative_to(root_dir)
        path_parts = relative_path.parts[:-1]  # 排除文件名本身
    except ValueError:
        # 如果不在根目录下，使用绝对路径
        path_parts = video_path.parent.parts
    
    print(f"  调试信息 - 路径部分: {path_parts}")  # 调试信息
    
    # 情况1: 直接在根目录下的视频
    if len(path_parts) == 0:
        # 从视频文件名中去掉DJI前缀
        video_name = video_path.stem
        if video_name.startswith("DJI_"):
            return video_name[4:]  # 去掉"DJI_"
        else:
            return video_name
    
    # 情况2: 在子文件夹中的视频
    date_str = "unknown"
    location_str = "unknown"
    
    # 查找日期信息 (8位数字)
    for part in path_parts:
        date_match = re.search(r'(\d{8})', str(part))
        if date_match:
            full_date = date_match.group(1)
            date_str = full_date[4:8]  # 取后4位，如0811
            print(f"  找到日期: {full_date} -> {date_str}")  # 调试信息
            break
    
    # 查找位置信息 - 优先匹配特定模式
    for part in path_parts:
        part_str = str(part).lower()
        print(f"  检查位置部分: {part_str}")  # 调试信息
        
        if re.search(r'^\d+m$', part_str):  # 优先匹配如15m这样的模式
            height_match = re.search(r'^(\d+)m$', part_str)
            if height_match:
                location_str = height_match.group(1) + "m"
                print(f"  找到高度: {part_str} -> {location_str}")
                break
        elif 'cocho' in part_str:
            location_str = "co"
            print(f"  找到cocho -> {location_str}")
            break
        elif part_str == 'mating':
            location_str = "ma"
            print(f"  找到mating -> {location_str}")
            break
    
    # 如果上面没有找到位置信息，再用其他方式
    if location_str == "unknown":
        for part in path_parts:
            part_str = str(part).lower()
            if part_str not in ['resize', 'original', 'videos', 'images'] and not re.search(r'\d{8}', part_str):  # 排除常见文件夹名和日期
                # 其他情况使用前2个字符
                if len(part_str) >= 2:
                    location_str = part_str[:2]
                    print(f"  其他位置: {part_str} -> {location_str}")
                    break
    
    # 生成命名前缀
    if date_str != "unknown" and location_str != "unknown":
        prefix = f"{date_str}_{location_str}"
    elif date_str != "unknown":
        prefix = date_str
    elif location_str != "unknown":
        prefix = location_str
    else:
        prefix = "unknown"
    
    print(f"  最终前缀: {prefix}")  # 调试信息
    return prefix


def resize_frame_to_1080p(frame):
    """
    将帧resize到1080P (1920x1080)
    :param frame: 输入帧
    :return: resize后的帧
    """
    height, width = frame.shape[:2]
    
    # 如果已经是1080P或更高，则不需要resize
    if height >= 1080 and width >= 1920:
        return frame
    
    # 计算新的尺寸，保持宽高比
    target_height = 1080
    target_width = 1920
    
    # 计算缩放比例
    scale_h = target_height / height
    scale_w = target_width / width
    scale = max(scale_h, scale_w)  # 使用较大的缩放比例确保至少达到1080P
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize图像
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 如果超过目标尺寸，进行裁剪到中心区域
    if new_height > target_height or new_width > target_width:
        start_y = (new_height - target_height) // 2
        start_x = (new_width - target_width) // 2
        resized_frame = resized_frame[start_y:start_y+target_height, start_x:start_x+target_width]
    
    return resized_frame


def extract_frames(video_path, frame_interval):
    """
    从视频中按指定间隔抽取帧
    :param video_path: 视频文件路径
    :param frame_interval: 每隔多少帧抽取一次
    :return: (frame, minutes, seconds, sub_frame, progress) 生成器
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"无法获取视频帧率: {video_path}")
        cap.release()
        return
        
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息 - 总帧数: {total_frames}, 帧率: {fps:.2f}, 分辨率: {width}x{height}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 检查图像质量
            if is_frame_valid(frame):
                # 如果是720P则resize到1080P
                if height <= 720:
                    frame = resize_frame_to_1080p(frame)
                
                # 计算时间戳
                seconds = frame_idx / fps
                td = timedelta(seconds=seconds)
                total_seconds = int(td.total_seconds())
                minutes = total_seconds // 60
                sec = total_seconds % 60
                sub_frame = int((seconds - total_seconds) * fps) + 1
                
                # 计算进度
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                
                yield frame, minutes, sec, sub_frame, progress

        frame_idx += 1

    cap.release()


def is_frame_valid(frame):
    """
    检查帧是否适合作为数据集
    :param frame: OpenCV BGR 图像
    :return: True/False
    """
    if frame is None:
        return False
    
    # 检查图像尺寸
    height, width = frame.shape[:2]
    if height < 100 or width < 100:
        return False
    
    # 检查图像是否过暗或过亮
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness < 20 or mean_brightness > 235:
        return False
    
    # 检查图像是否过于模糊
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:  # 阈值可调整
        return False
    
    return True


def has_cow(frame, model, confidence_threshold=0.9):
    """
    检测图片中是否有牛(standing或laying)并检查可信度
    :param frame: OpenCV BGR 图像
    :param model: YOLO模型
    :param confidence_threshold: 可信度阈值，默认0.9
    :return: True/False
    """
    try:
        results = model(frame, verbose=False)
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_name = model.names[int(cls_id)]
                    confidence = result.boxes.conf[i].item()  # 获取可信度
                    
                    # 检查是否为standing或laying牛，且可信度大于阈值
                    if (cls_name.lower() in ["standing", "laying"] and 
                        confidence >= confidence_threshold):
                        return True
        return False
    except Exception as e:
        print(f"YOLO检测出错: {e}")
        return False


def process_video(video_path, save_dir, naming_prefix, frame_interval, model, video_index, total_videos, root_dir):
    """
    处理单个视频，抽取帧并检测是否有牛
    """
    video_path = Path(video_path)
    video_name = video_path.stem  # DJI_0253
    
    # 获取视频ID
    if video_name.startswith("DJI_"):
        video_id = video_name[4:]  # 去掉DJI_前缀，得到0253
    else:
        video_id = video_name
    
    save_count = 0
    total_extracted = 0
    
    # 显示相对路径
    try:
        rel_path = video_path.relative_to(Path(root_dir))
        display_path = str(rel_path)
    except ValueError:
        display_path = video_name
    
    print(f"\n[{video_index}/{total_videos}] 开始处理视频: {display_path}")
    print(f"  视频ID: {video_id}")
    
    start_time = time.time()

    for frame, minutes, sec, sub_frame, progress in extract_frames(video_path, frame_interval):
        total_extracted += 1
        
        if has_cow(frame, model, confidence_threshold=0.9):
            # 修改文件名格式: 前缀_视频ID_分钟秒帧数.jpg
            filename = f"{naming_prefix}_{video_id}_{minutes:02d}{sec:02d}{sub_frame:02d}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # 保存高质量图片
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_count += 1
            
            # 显示第一个保存的文件名作为示例
            if save_count == 1:
                print(f"  文件名示例: {filename}")
        
        # 显示进度（每处理一定数量帧显示一次）
        if total_extracted % max(1, frame_interval * 10) == 0:
            elapsed_time = time.time() - start_time
            print(f"  进度: {progress:.1f}% | 已抽取: {total_extracted} | 已保存: {save_count} | 用时: {elapsed_time:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"✓ 视频处理完成")
    print(f"  共抽取 {total_extracted} 帧，保存 {save_count} 张包含牛的图片，用时 {elapsed_time:.1f}s")
    return save_count


def find_all_videos(root_dir):
    """
    递归查找所有视频文件，避免重复
    :param root_dir: 根目录
    :return: 视频文件路径列表
    """
    video_extensions = ['.MOV', '.mov', '.MP4', '.mp4', '.AVI', '.avi', '.mkv', '.MKV']
    video_files = set()  # 使用set避免重复
    
    # 使用rglob查找所有视频文件
    root_path = Path(root_dir)
    for ext in video_extensions:
        for video_file in root_path.rglob(f"*{ext}"):
            video_files.add(video_file)  # set会自动去重
    
    return sorted(list(video_files))


def main(ROOT_VIDEO_DIR, SAVE_DIR, MODEL_PATH, FRAME_INTERVAL):
    print("=" * 60)
    print("             牛检测图像提取工具 (递归版)")
    print("=" * 60)
    
    # 创建保存路径
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"保存目录: {SAVE_DIR}")
    
    # 加载 YOLO 模型
    try:
        print("正在加载YOLO模型...")
        model = YOLO(MODEL_PATH)
        print(f"✓ 成功加载模型: {MODEL_PATH}")
        print(f"模型类别: {list(model.names.values())}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 递归查找所有视频文件
    print("正在搜索视频文件...")
    video_files = find_all_videos(ROOT_VIDEO_DIR)
    
    if not video_files:
        print(f"✗ 在目录 {ROOT_VIDEO_DIR} 中未找到视频文件")
        return
    
    print(f"✓ 找到 {len(video_files)} 个视频文件")
    for i, vf in enumerate(video_files, 1):
        rel_path = vf.relative_to(Path(ROOT_VIDEO_DIR)) if Path(ROOT_VIDEO_DIR) in vf.parents else vf.name
        print(f"  {i}. {rel_path}")
    
    print(f"帧提取间隔: 每 {FRAME_INTERVAL} 帧提取一次")
    print(f"自动resize: 720P及以下 → 1080P")
    print("-" * 60)
    
    total_saved = 0
    start_time = time.time()
    
    for i, video_file in enumerate(video_files, 1):
        # 为每个视频生成命名前缀
        naming_prefix = extract_info_from_path(video_file, ROOT_VIDEO_DIR)
        
        saved_count = process_video(
            video_file, SAVE_DIR, naming_prefix, 
            FRAME_INTERVAL, model, i, len(video_files), ROOT_VIDEO_DIR
        )
        total_saved += saved_count
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("                处理完成!")
    print("=" * 60)
    print(f"总共处理: {len(video_files)} 个视频")
    print(f"总共保存: {total_saved} 张包含牛的图片")
    print(f"保存路径: {SAVE_DIR}")
    print(f"总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print("=" * 60)


# ========== 配置参数 (放在最下面方便修改) ==========
if __name__ == "__main__":
    # 视频根目录 - 会递归搜索所有子文件夹
    ROOT_VIDEO_DIR = r"D:/Datasets/CowDetection/images_original/videos/mating"
    
    # 保存目录
    SAVE_DIR = r"D:/Datasets/CowDetection/images_v_m"  
    
    # 模型路径
    MODEL_PATH = r"YOLOV11/output/yolo11s_20250807_183015/weights/best.pt"
    
    # 帧提取间隔（每隔多少帧取一帧，30fps取30帧即1秒）
    FRAME_INTERVAL = 3
    
    # 可信度阈值（可以在这里调整）
    CONFIDENCE_THRESHOLD = 0.9
    
    main(ROOT_VIDEO_DIR, SAVE_DIR, MODEL_PATH, FRAME_INTERVAL)