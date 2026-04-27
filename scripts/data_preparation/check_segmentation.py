"""
检查标注的labelme分割标注(json)质量
- 支持批量处理所有图像
- 支持单个文件检查
- 支持保存掩码图像
- 支持输出统计信息
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

def load_labelme_json(json_path: str) -> Dict:
    """Load labelme annotation file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def draw_polygon_mask(image: np.ndarray, polygon_points: List[List[float]], 
                     color: Tuple[int, int, int] = (0, 255, 0), 
                     alpha: float = 0.3) -> np.ndarray:
    """Draw polygon mask on image"""
    if len(polygon_points) < 3:
        return image
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    result = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return result

def draw_polygon_contour(image: np.ndarray, polygon_points: List[List[float]], 
                        color: Tuple[int, int, int] = (0, 255, 0), 
                        thickness: int = 2) -> np.ndarray:
    """Draw polygon contour on image"""
    if len(polygon_points) < 3:
        return image
    
    points = np.array(polygon_points, dtype=np.int32)
    cv2.polylines(image, [points], True, color, thickness)
    return image

def check_single_segmentation(image_path: str, json_path: str, 
                             output_dir: str = None) -> Dict:
    """Check single image segmentation annotation"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'error': f'Cannot read image: {image_path}'}
    
    # Read annotation
    try:
        labelme_data = load_labelme_json(json_path)
    except Exception as e:
        return {'success': False, 'error': f'Cannot read JSON: {e}'}
    
    # Create result image
    result_image = image.copy()
    
    # Get all segmentation annotations
    shapes = labelme_data.get('shapes', [])
    polygon_count = 0
    total_points = 0
    
    # Process all polygons
    for shape in shapes:
        if shape.get('shape_type') == 'polygon':
            polygon_points = shape.get('points', [])
            label = shape.get('label', 'unknown')
            
            if len(polygon_points) >= 3:
                polygon_count += 1
                total_points += len(polygon_points)
                
                # Draw mask
                result_image = draw_polygon_mask(
                    result_image, 
                    polygon_points, 
                    color=(0, 255, 0),
                    alpha=0.3
                )
                
                # Draw contour
                result_image = draw_polygon_contour(
                    result_image, 
                    polygon_points, 
                    color=(0, 255, 0),
                    thickness=2
                )
                
                # Add label text
                if polygon_points:
                    points_array = np.array(polygon_points)
                    center_x = int(np.mean(points_array[:, 0]))
                    center_y = int(np.mean(points_array[:, 1]))
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_image, 
                                (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                                (center_x + text_size[0]//2 + 5, center_y + 5),
                                (0, 0, 0), -1)
                    
                    cv2.putText(result_image, label, 
                              (center_x - text_size[0]//2, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save result
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{Path(image_path).stem}_mask.jpg"
        success = cv2.imwrite(str(output_file), result_image)
        
        if not success:
            return {'success': False, 'error': 'Failed to save image'}
    
    return {
        'success': True,
        'polygons': polygon_count,
        'points': total_points
    }

def batch_check_segmentation(data_dir: str, output_dir: str = None) -> None:
    """Batch check segmentation annotations"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory does not exist: {data_dir}")
        return
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = [f for f in data_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {data_dir}")
        return
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    print(f"Found {len(image_files)} images")
    print(f"Processing segmentation annotations...")
    
    # Statistics
    total_polygons = 0
    total_points = 0
    valid_files = 0
    processed_count = 0
    
    # Process with progress bar
    for image_file in tqdm(image_files, desc="Processing", unit="images"):
        # Find corresponding json file
        json_file = data_path / f"{image_file.stem}.json"
        
        if not json_file.exists():
            continue
        
        try:
            result = check_single_segmentation(
                str(image_file), 
                str(json_file),
                output_dir=output_dir
            )
            
            if result['success']:
                total_polygons += result['polygons']
                total_points += result['points']
                valid_files += 1
                processed_count += 1
            
        except Exception as e:
            continue
    
    # Print statistics
    print(f"\nProcessing completed!")
    print(f"Valid files: {valid_files}/{len(image_files)}")
    print(f"Processed files: {processed_count}")
    print(f"Total polygons: {total_polygons}")
    print(f"Total points: {total_points}")
    print(f"Average polygons per file: {total_polygons/valid_files:.1f}" if valid_files > 0 else "No valid files")
    print(f"Average points per polygon: {total_points/total_polygons:.1f}" if total_polygons > 0 else "No polygons")
    if output_dir:
        print(f"Mask images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Check labelme segmentation annotation quality")
    
    parser.add_argument("--data_dir", type=str, 
                       default="D:/Datasets/CowDetection/sam2_hq_segmentation",
                       help="Data directory path")
    parser.add_argument("--output_dir", type=str, 
                       default="D:/Datasets/CowDetection/segmentation_check",
                       help="Output directory path")
    parser.add_argument("--single_file", type=str,
                       help="Check single file (image path)")
    
    args = parser.parse_args()
    
    if args.single_file:
        # Check single file
        image_path = args.single_file
        json_path = str(Path(image_path).with_suffix('.json'))
        
        if not Path(json_path).exists():
            print(f"JSON file not found: {json_path}")
            return
        
        result = check_single_segmentation(
            image_path, 
            json_path,
            output_dir=args.output_dir
        )
        
        if result['success']:
            print(f"Processed: {result['polygons']} polygons, {result['points']} points")
        else:
            print(f"Error: {result['error']}")
    else:
        # Batch check
        batch_check_segmentation(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()