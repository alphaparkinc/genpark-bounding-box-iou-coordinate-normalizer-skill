from client import BoundingBoxNormalizer

box1 = [100, 100, 200, 200]
box2 = [120, 120, 220, 220]
iou = BoundingBoxNormalizer.compute_iou(box1, box2)
print("Computed IoU:", iou)
