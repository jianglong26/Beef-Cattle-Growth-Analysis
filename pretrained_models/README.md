# 预训练模型目录说明

## 📁 目录用途

此目录用于存放YOLOv11系列的预训练模型权重文件，统一管理项目中使用的所有预训练模型。

---

## 📦 当前模型

### 检测模型（Detection）

| 模型 | 大小 | 参数量 | 用途 | 文件 |
|------|------|--------|------|------|
| **YOLOv11n** | 5.4MB | ~2.6M | 轻量级，速度快 | `yolo11n.pt` |
| **YOLOv11s** | 19MB | ~9.4M | **当前使用**，平衡性能 | `yolo11s.pt` |
| **YOLOv11m** | - | ~20M | 中等模型 | `yolo11m.pt` |
| **YOLOv11l** | - | ~25M | 大型模型 | `yolo11l.pt` |
| **YOLOv11x** | - | ~57M | 超大模型 | `yolo11x.pt` |

### 分割模型（Segmentation）

| 模型 | 大小 | 用途 | 文件 |
|------|------|------|------|
| **YOLOv11s-seg** | 20MB | 实例分割 | `yolo11s-seg.pt` |
| **YOLOv11x-seg** | 120MB | 大型分割模型 | `yolo11x-seg.pt` |

---

## 🔧 配置使用

### 在 `config_s.py` 中配置：

```python
# 模型配置
MODEL_CONFIG = {
    "model_name": "yolo11s.pt",  # 模型名称
    "pretrained_path": PRETRAINED_MODELS_DIR / "yolo11s.pt",  # 完整路径
    ...
}
```

### 切换不同模型：

#### 使用YOLOv11n（轻量级，快速）：
```python
MODEL_CONFIG = {
    "model_name": "yolo11n.pt",
    "pretrained_path": PRETRAINED_MODELS_DIR / "yolo11n.pt",
    ...
}
```

#### 使用YOLOv11m（中等性能）：
```python
MODEL_CONFIG = {
    "model_name": "yolo11m.pt",
    "pretrained_path": PRETRAINED_MODELS_DIR / "yolo11m.pt",
    ...
}
```

---

## 📥 下载新模型

### 方式1：使用Ultralytics自动下载
```python
from ultralytics import YOLO

# 首次使用会自动下载到 ~/.cache/ultralytics/
model = YOLO('yolo11m.pt')  
```

### 方式2：手动下载
```bash
# 从GitHub下载
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt \
    -P /home/u11719919/project/agriculture_all/BeefCattleDetection/YOLOV11/pretrained_models/

# 或使用curl
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt \
    -o /home/u11719919/project/agriculture_all/BeefCattleDetection/YOLOV11/pretrained_models/yolo11m.pt
```

### 模型下载链接：

| 模型 | 下载链接 |
|------|---------|
| yolo11n.pt | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt |
| yolo11s.pt | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt |
| yolo11m.pt | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt |
| yolo11l.pt | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt |
| yolo11x.pt | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt |

---

## 🎯 训练时的模型加载逻辑

训练脚本 `train.py` 中的加载顺序：

```python
def load_model(self):
    # 1. 优先使用配置文件中指定的pretrained_path
    pretrained_path = self.config.MODEL_CONFIG.get("pretrained_path")
    if pretrained_path and Path(pretrained_path).exists():
        self.logger.info(f"Loading from: {pretrained_path}")
        self.model = YOLO(pretrained_path)
    
    # 2. 否则使用model_name（会从缓存或官方下载）
    else:
        model_name = self.config.MODEL_CONFIG["model_name"]
        self.logger.info(f"Loading official model: {model_name}")
        self.model = YOLO(model_name)
```

---

## 📊 模型选择建议

### 牛只检测项目推荐：

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| **实际部署（当前）** | **YOLOv11s** ✓ | 平衡速度和精度，适合边缘设备 |
| 实验对比 | YOLOv11n | 快速迭代测试 |
| 追求高精度 | YOLOv11m/l | 性能更好但速度慢 |
| 服务器部署 | YOLOv11x | 最高精度，适合高性能服务器 |

### 性能对比（COCO数据集）：

| 模型 | mAP@0.5:0.95 | 速度(ms) | 参数量 |
|------|--------------|----------|--------|
| YOLOv11n | 39.5% | 1.5 | 2.6M |
| YOLOv11s | 47.0% | 2.5 | 9.4M |
| YOLOv11m | 51.5% | 4.7 | 20.1M |
| YOLOv11l | 53.4% | 6.2 | 25.3M |
| YOLOv11x | 54.7% | 8.8 | 56.9M |

---

## 🛠️ 管理命令

### 查看当前模型
```bash
ls -lh pretrained_models/
```

### 删除不用的模型
```bash
rm pretrained_models/yolo11x-seg.pt
```

### 备份模型
```bash
cp -r pretrained_models/ pretrained_models_backup/
```

---

## ⚠️ 注意事项

1. **版本兼容**：确保模型版本与ultralytics包版本兼容
2. **磁盘空间**：大模型文件较大，注意磁盘空间
3. **路径配置**：修改配置后重新训练才生效
4. **不要上传到Git**：`.pt`文件已在`.gitignore`中

---

## 📝 更新日志

- **2026-01-09**: 创建预训练模型目录，统一管理模型文件
- 当前使用：YOLOv11s (yolo11s.pt)
- 项目路径：`YOLOV11/pretrained_models/`

---

**维护者**: BeefCattleDetection项目组  
**最后更新**: 2026-01-09
