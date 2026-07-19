# Dataset Card: Target and Arrow Detection v6

## Origin and license

Source: [Target and Arrow Detection v6 on Roboflow Universe](https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection/dataset/6). The source metadata identifies the provider as Roboflow user content and the dataset license as MIT. Original attribution files are preserved in `data/target-arrow-detection-v6/source_metadata/`.

The omitted source ZIP has SHA-256 `d81c8389e9bf1698b8bf86d76471024882c45cf090685f3704c0132357be53bd`; it is not duplicated because the extracted data are already included.

## Composition

The release contains 1,645 images and 1,645 paired YOLO text labels: 1,482 train, 98 validation and 65 test. Images are 640 by 640 pixels. Class order is `0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, target`.

Automated release checks found no EXIF or GPS metadata. YOLO class IDs and normalized coordinates are checked for valid ranges.

## Preparation and known risks

Source metadata records stretch resizing to 640 by 640 and offline augmentation. Visually related or augmented variants may occur across public splits. Therefore, the public test split is useful for protocol-consistent comparison but does not prove capture-session-, venue-, equipment- or subject-independent generalization.

The labels are bounding boxes. They do not provide calibrated physical impact points, camera geometry, archer identity, or controlled acquisition metadata.

