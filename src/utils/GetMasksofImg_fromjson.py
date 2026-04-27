import cv2
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
import math

def load_labelme_json(json_path):
    """加载labelme格式的JSON标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载JSON文件: {json_path}")
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None

def define_red_box_region(image_shape):
    """
    定义红色框区域（根据经验设置）
    基于图像中红色框的大致位置
    """
    height, width = image_shape[:2]
    
    # 根据经验定义红色框区域（可调整这些参数）
    # 假设红色框大致在图像的中央偏下区域
    box_width_ratio = 0.7  # 框宽度占图像宽度的比例
    box_height_ratio = 0.6  # 框高度占图像高度的比例
    
    # 计算红色框的坐标
    box_width = int(width * box_width_ratio)
    box_height = int(height * box_height_ratio)
    
    # 框的左上角坐标（居中偏下）
    x_start = int((width - box_width) / 2)
    y_start = int(height * 0.25)  # 从图像25%高度开始
    
    x_end = x_start + box_width
    y_end = y_start + box_height
    
    # 确保坐标在图像范围内
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_end)
    y_end = min(height, y_end)
    
    return (x_start, y_start, x_end, y_end)

def is_cow_in_red_box(bbox, red_box, tolerance=50):
    """
    判断牛的边界框是否在红色框区域内或边缘上
    
    Args:
        bbox: 牛的边界框 [x_min, y_min, x_max, y_max]
        red_box: 红色框区域 (x_start, y_start, x_end, y_end)
        tolerance: 边缘容忍度（像素）
    """
    cow_x_min, cow_y_min, cow_x_max, cow_y_max = bbox
    box_x_start, box_y_start, box_x_end, box_y_end = red_box
    
    # 扩展红色框区域（加上容忍度）
    expanded_x_start = box_x_start - tolerance
    expanded_y_start = box_y_start - tolerance
    expanded_x_end = box_x_end + tolerance
    expanded_y_end = box_y_end + tolerance
    
    # 检查牛的中心点是否在扩展的红色框内
    cow_center_x = (cow_x_min + cow_x_max) / 2
    cow_center_y = (cow_y_min + cow_y_max) / 2
    
    is_inside = (expanded_x_start <= cow_center_x <= expanded_x_end and 
                 expanded_y_start <= cow_center_y <= expanded_y_end)
    
    return is_inside

def calculate_cow_dimensions(mask):
    """
    计算牛的尺寸信息（长度、宽度、面积、长宽比）
    
    Args:
        mask: 二值掩膜
    
    Returns:
        dict: 包含尺寸信息的字典
    """
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算面积
    area = cv2.contourArea(largest_contour)
    
    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 计算最小外接矩形（可以旋转）
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 从最小外接矩形获取长度和宽度
    rect_width = rect[1][0]
    rect_height = rect[1][1]
    
    # 长度是较长的边，宽度是较短的边
    length = max(rect_width, rect_height)
    width = min(rect_width, rect_height)
    
    # 计算长宽比
    aspect_ratio = length / width if width > 0 else 0
    
    # 计算轮廓的长度（周长）
    perimeter = cv2.arcLength(largest_contour, True)
    
    # 计算头尾距离（轮廓上最远的两点）
    # 使用rotating calipers算法找最远点对
    def get_max_distance_points(contour):
        max_dist = 0
        max_points = None
        points = contour.reshape(-1, 2)
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    max_points = (points[i], points[j])
        
        return max_points, max_dist
    
    head_tail_points, head_tail_distance = get_max_distance_points(largest_contour)
    
    return {
        'area': area,
        'bbox_width': w,
        'bbox_height': h,
        'min_rect_length': length,
        'min_rect_width': width,
        'aspect_ratio': aspect_ratio,
        'perimeter': perimeter,
        'head_tail_distance': head_tail_distance,
        'head_tail_points': head_tail_points,
        'min_rect_box': box,
        'min_rect_angle': rect[2]
    }

def draw_dimensions_on_image(image, mask, dimensions, cow_id):
    """
    在图像上绘制牛的尺寸信息
    
    Args:
        image: 输入图像
        mask: 二值掩膜
        dimensions: 尺寸信息字典
        cow_id: 牛的ID
    
    Returns:
        带有尺寸标注的图像
    """
    result_image = image.copy()
    
    if dimensions is None:
        return result_image
    
    # 绘制最小外接矩形
    cv2.drawContours(result_image, [dimensions['min_rect_box']], 0, (255, 0, 0), 2)
    
    # 绘制头尾连线
    if dimensions['head_tail_points']:
        pt1, pt2 = dimensions['head_tail_points']
        cv2.line(result_image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 255), 3)
        
        # 在头尾点画圆
        cv2.circle(result_image, tuple(pt1.astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(result_image, tuple(pt2.astype(int)), 5, (0, 255, 0), -1)
    
    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 获取掩膜的边界框位置来放置文字
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # 在掩膜上方显示信息
        text_y = max(y - 10, 20)
        cv2.putText(result_image, f"Cow {cow_id}", (x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(result_image, f"L: {dimensions['min_rect_length']:.1f}px", (x, text_y + 25), font, font_scale-0.1, (255, 255, 0), thickness-1)
        cv2.putText(result_image, f"W: {dimensions['min_rect_width']:.1f}px", (x, text_y + 45), font, font_scale-0.1, (255, 255, 0), thickness-1)
        cv2.putText(result_image, f"H2T: {dimensions['head_tail_distance']:.1f}px", (x, text_y + 65), font, font_scale-0.1, (0, 255, 255), thickness-1)
    
    return result_image

def create_mask_from_polygon(polygon_points, image_shape):
    """根据多边形点创建掩膜"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_bounding_box_from_polygon(polygon_points):
    """从多边形点获取边界框"""
    points = np.array(polygon_points)
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return [x_min, y_min, x_max, y_max]

def parse_labelme_annotations(labelme_data, image_shape, red_box):
    """解析labelme标注数据，只处理红色框区域内的standing标注"""
    masks_and_boxes = []
    
    if 'shapes' not in labelme_data:
        print("JSON文件中没有找到'shapes'字段")
        return masks_and_boxes
    
    for i, shape in enumerate(labelme_data['shapes']):
        if shape['shape_type'] != 'polygon':
            print(f"跳过非多边形标注: {shape['shape_type']}")
            continue
        
        # 只处理标签为"standing"的标注
        label = shape.get('label', '').lower()
        if label != 'standing':
            print(f"跳过非standing标注: {shape.get('label', 'unknown')}")
            continue
        
        # 获取多边形点
        polygon_points = shape['points']
        if len(polygon_points) < 3:
            print(f"多边形点数不足: {len(polygon_points)}")
            continue
        
        # 创建掩膜
        mask = create_mask_from_polygon(polygon_points, image_shape)
        
        # 获取边界框
        bbox = get_bounding_box_from_polygon(polygon_points)
        
        # 验证边界框坐标
        x_min, y_min, x_max, y_max = bbox
        if x_min >= 0 and y_min >= 0 and x_max <= image_shape[1] and y_max <= image_shape[0]:
            # 检查是否在红色框区域内
            if is_cow_in_red_box(bbox, red_box):
                # 设置默认置信度为1.0（因为是人工标注）
                confidence = 1.0
                masks_and_boxes.append((mask, bbox, confidence))
                print(f"添加红色框内standing标注 {i+1}，标签: {shape.get('label', 'unknown')}")
            else:
                print(f"跳过红色框外的standing标注 {i+1}")
        else:
            print(f"边界框坐标无效: {bbox}")
    
    return masks_and_boxes

def create_cow_specific_mask(full_mask, box, image_shape):
    """创建特定牛的掩膜，只保留该牛的部分，其他部分用背景填充"""
    # 首先在整个图像上创建该牛的掩膜
    cow_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # 将该牛的掩膜区域复制到新掩膜中
    cow_mask[full_mask > 127] = 255
    
    # 在边界框区域内寻找连通组件
    x_min, y_min, x_max, y_max = box
    
    # 扩展边界框
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_shape[1], x_max + padding)
    y_max = min(image_shape[0], y_max + padding)
    
    # 在边界框区域内寻找连通组件
    bbox_mask = cow_mask[y_min:y_max, x_min:x_max].copy()
    
    if np.sum(bbox_mask) == 0:
        return bbox_mask, (x_min, y_min, x_max, y_max)
    
    # 寻找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bbox_mask, connectivity=8)
    
    if num_labels <= 1:
        return bbox_mask, (x_min, y_min, x_max, y_max)
    
    # 找到最大的连通组件（排除背景）
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    
    # 创建只包含最大连通组件的掩膜
    clean_mask = np.zeros_like(bbox_mask)
    clean_mask[labels == largest_component] = 255
    
    return clean_mask, (x_min, y_min, x_max, y_max)

def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """在图像上叠加彩色掩膜"""
    overlay = image.copy()
    mask_indices = mask > 0
    overlay[mask_indices] = color
    return cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

def process_single_cow(color_image, mask, box, cow_id, confidence, output_dir, cow_data_list, aspect_ratio_threshold=2.9):
    """处理单头牛：生成四种可视化图像并计算尺寸"""
    # 创建特定牛的掩膜
    clean_mask, adjusted_box = create_cow_specific_mask(mask, box, color_image.shape)
    x_min, y_min, x_max, y_max = adjusted_box
    
    # 裁剪彩色图像
    color_crop = color_image[y_min:y_max, x_min:x_max]
    
    # 检查掩膜是否有效
    if np.sum(clean_mask) == 0:
        print(f"牛 {cow_id} 掩膜为空，跳过")
        return None
    
    # 验证尺寸匹配
    if clean_mask.shape != color_crop.shape[:2]:
        print(f"牛 {cow_id} 掩膜尺寸 {clean_mask.shape} 与图像尺寸 {color_crop.shape[:2]} 不匹配")
        return None

    # 计算牛的尺寸信息
    dimensions = calculate_cow_dimensions(clean_mask)

    if dimensions is None:
        print(f"牛 {cow_id} 无法计算尺寸信息，跳过")
        return None
    
    # 检查长宽比阈值
    aspect_ratio = dimensions['aspect_ratio']
    if aspect_ratio <= aspect_ratio_threshold:
        print(f"牛 {cow_id} 长宽比 {aspect_ratio:.2f} 小于阈值 {aspect_ratio_threshold}，跳过")
        return None
    
    print(f"牛 {cow_id} 长宽比 {aspect_ratio:.2f} 通过阈值过滤")
    
    if dimensions:
        # 添加到数据列表
        cow_data_list.append({
            'Cow_ID': cow_id,
            'Area_pixels': dimensions['area'],
            'BBox_Width_pixels': dimensions['bbox_width'],
            'BBox_Height_pixels': dimensions['bbox_height'],
            'MinRect_Length_pixels': dimensions['min_rect_length'],
            'MinRect_Width_pixels': dimensions['min_rect_width'],
            'Aspect_Ratio': dimensions['aspect_ratio'],
            'Perimeter_pixels': dimensions['perimeter'],
            'Head_Tail_Distance_pixels': dimensions['head_tail_distance'],
            'MinRect_Angle_degrees': dimensions['min_rect_angle'],
            'Confidence': confidence
        })

    print(f"牛 {cow_id} 置信度: {confidence:.3f}, 原始尺寸: {clean_mask.shape}, 彩色图尺寸: {color_crop.shape}")

    # 创建叠加了绿色掩膜的图像
    color_with_mask = overlay_mask_on_image(color_crop, clean_mask, alpha=0.4)
    
    # 创建去掉背景的牛实例（只保留牛的部分，其他部分为黑色）
    cow_only = color_crop.copy()
    cow_only[clean_mask == 0] = 0  # 非牛部分设为黑色

    # 在牛实例上绘制尺寸信息
    cow_with_dimensions = draw_dimensions_on_image(color_crop.copy(), clean_mask, dimensions, cow_id)

    # 创建五列可视化（原始、绿色掩膜、去掉背景的牛、尺寸标注、二值掩膜）
    max_h = max(color_crop.shape[0], clean_mask.shape[0])
    max_w = max(color_crop.shape[1], clean_mask.shape[1])
    
    vis_height, vis_width = max_h, max_w * 5  # 1行5列
    visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    # 调整图像尺寸以适应网格
    def resize_to_fit(img, target_h, target_w):
        if len(img.shape) == 2:
            resized = np.zeros((target_h, target_w), dtype=img.dtype)
            h_start = (target_h - img.shape[0]) // 2
            w_start = (target_w - img.shape[1]) // 2
            h_end = h_start + img.shape[0]
            w_end = w_start + img.shape[1]
            resized[h_start:h_end, w_start:w_end] = img
        else:
            resized = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            h_start = (target_h - img.shape[0]) // 2
            w_start = (target_w - img.shape[1]) // 2
            h_end = h_start + img.shape[0]
            w_end = w_start + img.shape[1]
            resized[h_start:h_end, w_start:w_end] = img
        return resized

    # 调整所有图像尺寸
    color_crop_resized = resize_to_fit(color_crop, max_h, max_w)
    color_with_mask_resized = resize_to_fit(color_with_mask, max_h, max_w)
    cow_only_resized = resize_to_fit(cow_only, max_h, max_w)
    cow_with_dimensions_resized = resize_to_fit(cow_with_dimensions, max_h, max_w)
    clean_mask_resized = resize_to_fit(clean_mask, max_h, max_w)
    
    # 将二值掩膜转换为RGB
    clean_mask_rgb = cv2.cvtColor(clean_mask_resized, cv2.COLOR_GRAY2RGB)

    # 转换为RGB
    color_crop_rgb = cv2.cvtColor(color_crop_resized, cv2.COLOR_BGR2RGB)
    color_with_mask_rgb = cv2.cvtColor(color_with_mask_resized, cv2.COLOR_BGR2RGB)
    cow_only_rgb = cv2.cvtColor(cow_only_resized, cv2.COLOR_BGR2RGB)
    cow_with_dimensions_rgb = cv2.cvtColor(cow_with_dimensions_resized, cv2.COLOR_BGR2RGB)

    # 填充一行五列
    visualization[0:max_h, 0:max_w] = color_crop_rgb                           # 原始图像
    visualization[0:max_h, max_w:2*max_w] = color_with_mask_rgb                # 绿色掩膜
    visualization[0:max_h, 2*max_w:3*max_w] = cow_only_rgb                     # 去掉背景的牛
    visualization[0:max_h, 3*max_w:4*max_w] = cow_with_dimensions_rgb          # 尺寸标注
    visualization[0:max_h, 4*max_w:5*max_w] = clean_mask_rgb                   # 二值掩膜

    # 添加编号标签和置信度
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # 白色
    thickness = 2
    text = f"Cow {cow_id} ({confidence:.3f})"

    # 在每张子图上添加标签
    positions = [
        (10, 25), (max_w+10, 25), (2*max_w+10, 25), (3*max_w+10, 25), (4*max_w+10, 25)
    ]
    labels = [
        "Original", "With Mask", "Cow Only", "With Dimensions", "Binary Mask"
    ]
    
    for pos, label in zip(positions, labels):
        cv2.putText(visualization, f"{text} - {label}", pos, 
                    font, font_scale, color, thickness, cv2.LINE_AA)

    # 保存可视化图像
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, f"cow_{cow_id}_visualization.png")
    cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"保存牛 {cow_id} 的可视化图像到 {vis_path}")

    return clean_mask, color_crop, cow_only

def process_cows(color_image_path, masks_and_boxes, output_dir, red_box, aspect_ratio_threshold=2.9):
    """处理所有牛，生成整体图像和单牛可视化"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取彩色图像
    color_image = cv2.imread(color_image_path)
    if color_image is None:
        raise ValueError("无法读取彩色图像")

    # 复制图像用于绘制整体检测框和掩膜
    full_image = color_image.copy()
    full_image_with_masks = color_image.copy()
    
    # 在图像上绘制红色框
    box_x_start, box_y_start, box_x_end, box_y_end = red_box
    cv2.rectangle(full_image, (box_x_start, box_y_start), (box_x_end, box_y_end), (0, 0, 255), 3)
    cv2.rectangle(full_image_with_masks, (box_x_start, box_y_start), (box_x_end, box_y_end), (0, 0, 255), 3)
    
    # 添加红色框标签
    cv2.putText(full_image, "Detection Region", (box_x_start, box_y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(full_image_with_masks, "Detection Region", (box_x_start, box_y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # 用于存储牛的数据
    cow_data_list = []

    # 计数器：用于给通过阈值过滤的牛重新编号
    valid_cow_id = 1
    total_cows = len(masks_and_boxes)
    filtered_out_count = 0

    # 处理每头牛
    for idx, (mask, box, confidence) in enumerate(masks_and_boxes, 1):
        # 处理单头牛
        result = process_single_cow(color_image, mask, box, valid_cow_id, confidence, output_dir, cow_data_list, aspect_ratio_threshold)
        if result is None:
            filtered_out_count += 1
            continue

        # 在整体图像上绘制检测框和编号（包含置信度）
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(full_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(full_image, f"Cow {idx} ({confidence:.3f})", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 增加有效牛的计数
        valid_cow_id += 1

        # 在整体图像上叠加绿色掩膜
        full_image_with_masks = overlay_mask_on_image(full_image_with_masks, mask, color=(0, 255, 0), alpha=0.3)

    # 保存整体图像（带检测框和红色区域）
    full_image_path = os.path.join(output_dir, "full_image_with_boxes_and_region.png")
    cv2.imwrite(full_image_path, full_image)
    print(f"保存整体图像（带检测框和红色区域）到 {full_image_path}")
    
    # 保存整体图像（带掩膜和红色区域）
    full_image_masks_path = os.path.join(output_dir, "full_image_with_masks_and_region.png")
    cv2.imwrite(full_image_masks_path, full_image_with_masks)
    print(f"保存整体图像（带掩膜和红色区域）到 {full_image_masks_path}")
    
    # 保存牛的尺寸数据到CSV
    if cow_data_list:
        image_name = Path(color_image_path).stem
        csv_path = os.path.join(output_dir, f"{image_name}_cow_measurements.csv")
        
        # 如果文件存在，先删除
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                print(f"删除现有文件: {csv_path}")
            except PermissionError:
                print(f"无法删除文件 {csv_path}，可能正在被其他程序使用")
                # 尝试使用不同的文件名
                import time
                timestamp = int(time.time())
                csv_path = os.path.join(output_dir, f"{image_name}_cow_measurements.csv")
                print(f"使用新文件名: {csv_path}")
        
        # 保存CSV文件
        try:
            df = pd.DataFrame(cow_data_list)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"保存牛的尺寸数据到 {csv_path}")
        except PermissionError:
            print(f"权限错误：无法写入 {csv_path}")
            print("请检查：")
            print("1. 文件是否正在被Excel或其他程序打开")
            print("2. 目录是否有写入权限")
            print("3. 尝试关闭相关程序后重新运行")
            return
        except Exception as e:
            print(f"保存CSV文件时发生错误: {e}")
            return
        
        # 打印统计信息
        print(f"\n牛的尺寸统计信息:")
        print(f"总共检测到的牛数量: {total_cows}")
        print(f"长宽比过滤掉的牛数量: {filtered_out_count}")
        print(f"通过长宽比阈值({aspect_ratio_threshold})的牛数量: {len(cow_data_list)}")
        print(f"平均面积: {df['Area_pixels'].mean():.1f} 像素")
        print(f"平均长度: {df['MinRect_Length_pixels'].mean():.1f} 像素")
        print(f"平均宽度: {df['MinRect_Width_pixels'].mean():.1f} 像素")
        print(f"平均长宽比: {df['Aspect_Ratio'].mean():.2f}")
        print(f"平均头尾距离: {df['Head_Tail_Distance_pixels'].mean():.1f} 像素")
        print(f"长宽比范围: {df['Aspect_Ratio'].min():.2f} - {df['Aspect_Ratio'].max():.2f}")
    else:
        print(f"\n没有牛通过长宽比阈值({aspect_ratio_threshold})过滤")

def process_image_with_json(image_path, json_path, output_dir, aspect_ratio_threshold=2.9):
    """使用图像和对应的JSON标注文件生成掩膜和可视化"""
    print(f"开始处理图像: {image_path}")
    print(f"使用JSON标注: {json_path}")
    print(f"输出目录: {output_dir}")
    print(f"长宽比阈值: {aspect_ratio_threshold}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取输入图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape}")
    
    # 定义红色框区域
    red_box = define_red_box_region(image.shape)
    print(f"红色框区域: {red_box}")
    
    # 加载JSON标注文件
    labelme_data = load_labelme_json(json_path)
    if labelme_data is None:
        return
    
    # 解析标注数据，获取掩膜和边界框（只处理红色框内的standing标注）
    masks_and_boxes = parse_labelme_annotations(labelme_data, image.shape, red_box)
    
    if not masks_and_boxes:
        print("没有找到红色框区域内的有效standing标注数据")
        return
    
    print(f"找到 {len(masks_and_boxes)} 个红色框区域内的有效standing标注")
    
    # 处理所有牛
    process_cows(image_path, masks_and_boxes, output_dir, red_box, aspect_ratio_threshold)
    
    print("处理完成")

# 示例调用
if __name__ == "__main__":
    image_path = "D:/Datasets/CowDetection/sam2_hq_seg_large/20240831_15m_0207.jpg"
    json_path = "D:/Datasets/CowDetection/sam2_hq_seg_large/20240831_15m_0207.json"  # 对应的JSON文件
    image_name = Path(image_path).stem
    output_dir = f"YOLOV11/Masks/{image_name}_fromjson"
    process_image_with_json(image_path, json_path, output_dir, aspect_ratio_threshold=2.9)