# genpark-bounding-box-iou-coordinate-normalizer-skill

Bounding box normalizer and Non-Maximum Suppression (NMS) engine computing IoU overlaps and coordinate space mappings.

## Architecture

```mermaid
flowchart TD
    Boxes[Raw Model Detections] --> Normalizer[Coordinate Space Normalizer]
    Normalizer --> IoUEngine[Intersection over Union Calculator]
    IoUEngine --> NMS[Greedy NMS Filtering]
    NMS --> CleanBoxes[Deduplicated Object Targets]
```

## Features
- **Flexible Coordinate Formats**: Supports `[0, 1]`, `[0, 1000]`, and absolute pixels.
- **Fast IoU & NMS**: Pure Python implementation with zero C/torch dependencies.
