"""Bounding Box IoU & Coordinate Normalizer.
100% Python Standard Library.
"""

class BoundingBoxNormalizer:
    """Normalizes bounding boxes and computes Intersection over Union (IoU) and NMS."""
    @staticmethod
    def to_pixel(box, screen_w, screen_h, coord_type="normalized_0_1"):
        ymin, xmin, ymax, xmax = box
        if coord_type == "normalized_1000":
            return [
                int(ymin * screen_h / 1000.0),
                int(xmin * screen_w / 1000.0),
                int(ymax * screen_h / 1000.0),
                int(xmax * screen_w / 1000.0)
            ]
        elif coord_type == "normalized_0_1":
            return [
                int(ymin * screen_h),
                int(xmin * screen_w),
                int(ymax * screen_h),
                int(xmax * screen_w)
            ]
        return [int(v) for v in box]

    @staticmethod
    def compute_iou(box_a, box_b):
        ymin_a, xmin_a, ymax_a, xmax_a = box_a
        ymin_b, xmin_b, ymax_b, xmax_b = box_b

        inter_ymin = max(ymin_a, ymin_b)
        inter_xmin = max(xmin_a, xmin_b)
        inter_ymax = min(ymax_a, ymax_b)
        inter_xmax = min(xmax_a, xmax_b)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area_a = max(0, xmax_a - xmin_a) * max(0, ymax_a - ymin_a)
        area_b = max(0, xmax_b - xmin_b) * max(0, ymax_b - ymin_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return round(inter_area / union_area, 4)

    @staticmethod
    def non_max_suppression(boxes_with_scores, iou_threshold=0.5):
        sorted_boxes = sorted(boxes_with_scores, key=lambda x: x["score"], reverse=True)
        keep = []

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)
            remaining = []
            for item in sorted_boxes:
                iou = BoundingBoxNormalizer.compute_iou(best["box"], item["box"])
                if iou < iou_threshold:
                    remaining.append(item)
            sorted_boxes = remaining

        return keep
