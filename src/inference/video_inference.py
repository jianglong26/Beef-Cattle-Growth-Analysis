import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import logging
from datetime import datetime
from typing import Tuple, List, Dict
import json

class VideoInference:
    def __init__(self, model_path: str, device: str = 'auto'):
        """初始化视频推理器
        
        Args:
            model_path: 模型路径
            device: 设备选择 ('auto', 'cpu', 'cuda', 'cuda:0')
        """
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device.startswith('cuda'):
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 加载模型
        self.model = YOLO(model_path)
        if self.device.startswith('cuda'):
            self.model.model = self.model.model.to(self.device)
        # 获取类别名称
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        print(f"Model classes: {self.class_names}")

        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'inference_times': [],
            'fps_history': []
        }
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("YOLOV11/output/video_inference")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"video_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def get_video_info(self, video_path: str) -> Dict:
        """获取视频信息
        
        Args:
            video_path: 视频路径
            
        Returns:
            Dict: 视频信息
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = None, 
                    scale_factor: float = None) -> Tuple[np.ndarray, float]:
        """调整帧尺寸
        
        Args:
            frame: 输入帧
            target_size: 目标尺寸 (width, height)
            scale_factor: 缩放因子
            
        Returns:
            Tuple[np.ndarray, float]: (调整后的帧, 实际缩放因子)
        """
        original_height, original_width = frame.shape[:2]
        
        if target_size:
            target_width, target_height = target_size
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            scale = min(scale_x, scale_y)  # 保持长宽比
        elif scale_factor:
            scale = scale_factor
        else:
            return frame, 1.0
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_frame, scale
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       scale_factor: float = 1.0) -> np.ndarray:
        """在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            detections: 检测结果
            scale_factor: 缩放因子（用于坐标还原）
            
        Returns:
            np.ndarray: 绘制后的帧
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            # 修改 draw_detections 方法中的标签部分
            for detection in detections:
                # 还原到原始坐标
                x1, y1, x2, y2 = [int(coord / scale_factor) for coord in detection['bbox']]
                confidence = detection['confidence']
                class_id = detection['class_id']
                
                # 确保坐标在有效范围内
                height, width = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 获取类别名称并绘制标签
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame
    
    def process_video(self, input_path: str, output_path: str = None, 
                 conf_threshold: float = 0.5, target_size: Tuple[int, int] = None,
                 scale_factor: float = None, save_detections: bool = True,
                 display_realtime: bool = False, skip_frames: int = 0) -> Dict:
        """处理视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
            target_size: 目标尺寸 (width, height)，None表示保持原尺寸
            scale_factor: 缩放因子，None表示不缩放
            save_detections: 是否保存检测结果到JSON
            display_realtime: 是否实时显示
            skip_frames: 跳帧处理（0表示处理所有帧）
            
        Returns:
            Dict: 处理结果统计
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        # 获取视频信息
        video_info = self.get_video_info(str(input_path))
        self.logger.info(f"Video Info: {video_info}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(input_path))
        
        # 设置输出视频路径 - 强制使用.mp4扩展名以确保兼容性
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_detected.mp4"
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / f"{input_path.stem}_detected.mp4"
            elif not output_path.suffix:
                output_path = output_path.with_suffix('.mp4')
            elif output_path.suffix.lower() not in ['.mp4', '.avi']:
                output_path = output_path.with_suffix('.mp4')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 改进的视频编码器设置 - 使用H.264编码，确保广泛兼容性
        out_fps = video_info['fps']
        out_width = video_info['width']
        out_height = video_info['height']
        
        # 如果指定了输出尺寸，调整输出视频尺寸
        if target_size:
            out_width, out_height = target_size
        elif scale_factor:
            out_width = int(video_info['width'] * scale_factor)
            out_height = int(video_info['height'] * scale_factor)
        
        # 确保宽度和高度是偶数（H.264要求）
        out_width = out_width + (out_width % 2)
        out_height = out_height + (out_height % 2)
        
        # 尝试多种编码器，优先使用最兼容的
        codecs_to_try = [
            ('mp4v', '.mp4'),  # MPEG-4，广泛支持
            ('XVID', '.avi'),  # Xvid编码，兼容性好
            ('MJPG', '.avi'),  # Motion JPEG，质量高
            ('H264', '.mp4'),  # H.264，如果系统支持
        ]
        
        out = None
        for fourcc_str, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                test_path = output_path.with_suffix(ext)
                out = cv2.VideoWriter(str(test_path), fourcc, out_fps, (out_width, out_height))
                
                # 测试是否能正常工作
                if out.isOpened():
                    output_path = test_path
                    self.logger.info(f"Using codec: {fourcc_str}, output: {output_path}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize codec {fourcc_str}: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None:
            raise RuntimeError("Failed to initialize video writer with any codec")
        
        # 处理统计
        frame_count = 0
        detection_results = []
        
        self.logger.info("Starting video processing...")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.stats['total_frames'] = frame_count
            
            # 跳帧处理
            if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                # 对于跳过的帧，直接写入原帧
                if target_size or scale_factor:
                    frame_resized, _ = self.resize_frame(frame, target_size, scale_factor)
                    # 确保帧尺寸匹配输出视频尺寸
                    if frame_resized.shape[:2] != (out_height, out_width):
                        frame_resized = cv2.resize(frame_resized, (out_width, out_height))
                    out.write(frame_resized)
                else:
                    # 确保帧尺寸匹配输出视频尺寸
                    if frame.shape[:2] != (out_height, out_width):
                        frame = cv2.resize(frame, (out_width, out_height))
                    out.write(frame)
                continue
            
            # 调整帧尺寸（用于推理）
            inference_frame, resize_scale = self.resize_frame(frame, target_size, scale_factor)
            
            # 推理
            inference_start = time.time()
            results = self.model(inference_frame, device=self.device, verbose=False, conf=conf_threshold)
            inference_time = time.time() - inference_start
            
            self.stats['inference_times'].append(inference_time)
            self.stats['processed_frames'] += 1
            
            # 解析检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'frame': frame_count,
                            'class_id': class_id,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence)
                        }
                        detections.append(detection)
                        
            self.stats['total_detections'] += len(detections)
            detection_results.extend(detections)
            
            # 绘制检测结果
            annotated_frame = self.draw_detections(inference_frame, detections, 1.0)
            
            # 添加信息文本
            fps_current = 1.0 / inference_time if inference_time > 0 else 0
            self.stats['fps_history'].append(fps_current)
            
            info_text = [
                f"Frame: {frame_count}/{video_info['frame_count']}",
                f"Detections: {len(detections)}",
                f"FPS: {fps_current:.1f}",
                f"Progress: {frame_count/video_info['frame_count']*100:.1f}%"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 确保帧尺寸匹配输出视频尺寸
            if annotated_frame.shape[:2] != (out_height, out_width):
                annotated_frame = cv2.resize(annotated_frame, (out_width, out_height))
            
            # 写入输出视频
            out.write(annotated_frame)
            
            # 实时显示
            if display_realtime:
                display_frame = cv2.resize(annotated_frame, (960, 540))  # 缩放显示
                cv2.imshow('Video Inference', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度显示
            if frame_count % 100 == 0:
                progress = frame_count / video_info['frame_count'] * 100
                avg_fps = np.mean(self.stats['fps_history'][-100:]) if self.stats['fps_history'] else 0
                self.logger.info(f"Progress: {progress:.1f}%, Avg FPS: {avg_fps:.1f}")
        
        # 清理资源
        cap.release()
        out.release()
        if display_realtime:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        # 保存检测结果
        if save_detections:
            detection_file = output_path.parent / f"{output_path.stem}_detections.json"
            with open(detection_file, 'w') as f:
                json.dump({
                    'video_info': video_info,
                    'processing_stats': self.get_processing_stats(total_time),
                    'detections': detection_results
                }, f, indent=2)
            self.logger.info(f"Detections saved to: {detection_file}")
        
        self.logger.info(f"Video processing completed! Output saved to: {output_path}")
        return self.get_processing_stats(total_time)
    
    def get_processing_stats(self, total_time: float) -> Dict:
        """获取处理统计信息
        
        Args:
            total_time: 总处理时间
            
        Returns:
            Dict: 统计信息
        """
        return {
            'total_frames': self.stats['total_frames'],
            'processed_frames': self.stats['processed_frames'],
            'total_detections': self.stats['total_detections'],
            'total_time': total_time,
            'avg_inference_time': np.mean(self.stats['inference_times']) if self.stats['inference_times'] else 0,
            'avg_fps': np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0,
            'processing_fps': self.stats['processed_frames'] / total_time if total_time > 0 else 0,
            'detections_per_frame': self.stats['total_detections'] / self.stats['processed_frames'] if self.stats['processed_frames'] > 0 else 0
        }

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Video Inference for Cattle Detection')
    parser.add_argument('--input', type=str, default=r"D:\Datasets\CowDetection\images_original\videos\20240811\15m\DJI_0134.MOV", help='Input video path')
    parser.add_argument('--output', type=str, default="YOLOV11/output/yolo11s_20250807_183015/test_results/0811_15m_DJI_0134_inference", help='Output video path (optional)')
    parser.add_argument('--model', type=str, default=r"YOLOV11\output\yolo11s_20250807_183015\weights\best.pt", 
                        help='Model path')
    parser.add_argument('--conf', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--target-width', type=int, help='Target width (optional)')
    parser.add_argument('--target-height', type=int, help='Target height (optional)')
    parser.add_argument('--scale', type=float, help='Scale factor (e.g., 0.5 for half size)')
    parser.add_argument('--skip-frames', type=int, default=0, help='Skip frames for faster processing')
    parser.add_argument('--no-save-detections', action='store_true', help='Do not save detection results')
    parser.add_argument('--display', action='store_true', default=True, help='Display real-time processing')
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        inferencer = VideoInference(args.model, args.device)
        
        # 设置目标尺寸
        target_size = None
        if args.target_width and args.target_height:
            target_size = (args.target_width, args.target_height)
        
        # 处理视频
        stats = inferencer.process_video(
            input_path=args.input,
            output_path=args.output,
            conf_threshold=args.conf,
            target_size=target_size,
            scale_factor=args.scale,
            save_detections=not args.no_save_detections,
            display_realtime=args.display,
            skip_frames=args.skip_frames
        )
        
        # 打印统计信息
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total frames: {stats['total_frames']}")
        print(f"Processed frames: {stats['processed_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Processing time: {stats['total_time']:.2f}s")
        print(f"Average inference FPS: {stats['avg_fps']:.2f}")
        print(f"Overall processing FPS: {stats['processing_fps']:.2f}")
        print(f"Detections per frame: {stats['detections_per_frame']:.2f}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())