import torch
import numpy as np
import cv2


def yolo_seg_postprocess(preds, img_shape, conf_thres=0.25, iou_thres=0.45, device='cpu'):
    """
    YOLOv8-seg 后处理：NMS + 掩码解析
    :param preds: 模型输出 (N, 6+32)，带有 seg mask prototype（(32, H/4, W/4)）
    :param img_shape: 输入图像大小 (h, w)
    :param conf_thres: 置信度阈值
    :param iou_thres: NMS的IOU阈值
    :return: boxes, masks
    """
    from ultralytics.utils.ops import non_max_suppression, scale_masks

    pred, proto = preds[0], preds[1]  # [boxes+mask coeff], [mask proto]

    # Step 1: NMS（来自ultralytics库）
    det = non_max_suppression(pred, conf_thres, iou_thres, nc=None, classes=None, agnostic=False, max_det=1000)[0]
    if det is None or len(det) == 0:
        return [], []

    det = det.to(device)

    # Step 2: 掩码恢复（矢量化处理）
    masks = proto @ det[:, 6:].T  # shape: (mask_h, mask_w, num_dets)
    masks = masks.sigmoid().permute(2, 0, 1).contiguous()  # shape: (num_dets, mask_h, mask_w)

    # Step 3: 将掩码恢复到原图尺寸
    masks = scale_masks(img_shape, masks, det[:, :4])  # 还原到原图大小

    boxes = det[:, :4].cpu().numpy()
    masks = masks.cpu().numpy()

    return boxes, masks


from ultralytics import YOLO
import cv2

model = YOLO(r"E:\TRT\ultralytics\checkpoint\yolo11\segment\yolo11s-seg.onnx")
img = cv2.imread(r'D:\workspace\imageyolo\bear(1).jpg')
results = model(img)

boxes, masks = yolo_seg_postprocess(results, img.shape[:2])
print("ok")
