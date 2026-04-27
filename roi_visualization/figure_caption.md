# Figure Caption: ROI Filtering Visualization

## English Version

**Figure X. Visual demonstration of the Region of Interest (ROI) filtering pipeline for cattle detection and instance selection.**

The red dashed box delineates the ROI region (center_70%, covering 70% × 70% of the image center with normalized coordinates [0.15, 0.15, 0.85, 0.85]). Green bounding boxes indicate detected standing cattle instances that pass the filtering criteria (confidence threshold ≥ 0.7 and centroid within ROI). Orange circles with white cross markers denote the geometric centroid of each detected bounding box, computed as the midpoint of the bounding box corners. The top-left corner displays detection statistics: 29 total standing cattle detected, with 18 instances (62.1%) retained within the ROI and 11 instances (37.9%) filtered out due to falling outside the ROI boundary. The top-right corner provides ROI region specifications in both pixel coordinates and normalized coordinates. This filtering approach mitigates edge effects caused by perspective distortion and occlusion, ensuring more reliable morphometric measurements for cattle growth monitoring.

---

## 中文版本

**图X. 牛只检测中感兴趣区域（ROI）过滤管道的可视化展示**

红色虚线框界定了ROI区域（center_70%，覆盖图像中心70% × 70%区域，归一化坐标为[0.15, 0.15, 0.85, 0.85]）。绿色检测框表示通过过滤条件的站立姿态牛只实例（置信度阈值≥0.7且质心位于ROI内）。橙色圆圈与白色十字标记表示每个检测框的几何质心，计算为边界框角点的中点。左上角显示检测统计信息：共检测到29头站立姿态牛只，其中18个实例（62.1%）保留在ROI内，11个实例（37.9%）因位于ROI边界外而被过滤。右上角提供ROI区域的像素坐标和归一化坐标规格。该过滤方法可减轻由透视畸变和遮挡引起的边缘效应，确保牛只生长监测中更可靠的形态测量。

---

## Short Version (简洁版)

**Figure X. ROI-based instance filtering for cattle detection.**

Red dashed box: ROI boundary (center_70%). Green boxes: detected standing cattle within ROI. Orange circles with white crosses: bounding box centroids. Statistics show 18 out of 29 detected instances (62.1%) retained after ROI filtering. Pixel and normalized coordinates are displayed in the top-right corner.

**图X. 基于ROI的牛只检测实例过滤**

红色虚线框：ROI边界（center_70%）。绿色框：ROI内检测到的站立牛只。橙色圆圈+白色十字：边界框质心。统计显示29个检测实例中有18个（62.1%）在ROI过滤后保留。右上角显示像素和归一化坐标。

---

## Key Points for Caption

1. **ROI Region**: Red dashed rectangle, center_70% configuration
2. **Detection Boxes**: Green bounding boxes for standing cattle
3. **Centroids**: Orange circles with white cross markers (geometric center of each box)
4. **Statistics**: 29 total, 18 in ROI (62.1%), 11 filtered out (37.9%)
5. **Coordinates**: Both pixel and normalized [0.15, 0.15, 0.85, 0.85]
6. **Purpose**: Reduce edge effects, improve measurement reliability
