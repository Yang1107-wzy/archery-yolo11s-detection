# 数据集卡：Target and Arrow Detection v6

数据来自 [Roboflow Universe 的 Target and Arrow Detection v6](https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection/dataset/6)，来源元数据记录许可为 MIT，原始署名文件保存在 `data/target-arrow-detection-v6/source_metadata/`。未重复发布的原始 ZIP SHA-256 为 `d81c8389e9bf1698b8bf86d76471024882c45cf090685f3704c0132357be53bd`。

本仓库包含 1,645 张图片与 1,645 个一一对应的 YOLO 标签：train 1,482、validation 98、test 65，图片为 640×640。类别顺序为 `0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, target`。自动检查未发现 EXIF 或 GPS，并验证了类别编号和归一化坐标范围。

来源元数据记录了拉伸缩放和离线增强。公开 split 之间可能存在视觉近似或增强版本，因此 test 只能作为同一公开协议下的比较，不能证明跨拍摄会话、场地、器材或被试泛化。标签是边界框，不包含经过标定的真实撞击点、相机几何或受控采集元数据。

