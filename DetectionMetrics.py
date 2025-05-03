import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt


class DetectionMetrics:
    def __init__(self, iou_threshold=0.5, class_names=['object']):
        """
        Initializes the class.

        - iou_threshold: how much two boxes should overlap to be counted as a match (e.g., 0.5 means 50% overlap).
        - class_names: names of the classes in your data. For your case, just ['bird'].
        """
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        self.metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
        self.preds_raw = []
        self.targets_raw = []

    def update(self, preds, targets):
        """
        Add predictions and ground truths for one batch.

        - preds: list of dicts with 'boxes', 'scores', 'labels'
        - targets: list of dicts with 'boxes', 'labels'
        """
        self.preds_raw += preds
        self.targets_raw += targets
        self.metric.update(preds, targets)

    def compute(self):
        """
        Calculate mAP and AP scores.

        Returns a dictionary with:
        - mAP@0.5
        - mAP@0.5:0.95
        - mAP@0.75
        - Per-class AP (if available)
        """
        results = self.metric.compute()

        map_50 = results['map_50']
        map_5095 = results['map']
        map_75 = results['map_75']
        per_class_ap = results['map_per_class']

        if per_class_ap is not None:
            per_class_ap = {
                self.class_names[i]: ap.item()
                for i, ap in enumerate(per_class_ap)
            }

        print(f"mAP@0.5: {map_50:.4f}")
        print(f"mAP@0.5:0.95: {map_5095:.4f}")
        print(f"mAP@0.75: {map_75:.4f}")
        if per_class_ap:
            print("Per-class AP:")
            for cls, val in per_class_ap.items():
                print(f"  {cls}: {val:.4f}")

        return {
            'mAP@0.5': map_50.item(),
            'mAP@0.5:0.95': map_5095.item(),
            'mAP@0.75': map_75.item(),
            'per_class_ap': per_class_ap
        }

    def compute_classification_metrics(self, confidence_threshold=0.5):
        """
        Estimate basic classification metrics from object detection results.

        Uses number of predicted boxes and ground truth boxes to compute:
        - Precision
        - Recall
        - F1-score

        This is a rough estimate, useful for understanding trade-offs.
        """
        total_true = 0
        total_pred = 0

        for pred, target in zip(self.preds_raw, self.targets_raw):
            # Only keep predictions with high confidence
            pred_boxes = pred['boxes'][pred['scores'] > confidence_threshold]
            gt_boxes = target['boxes']

            total_pred += len(pred_boxes)
            total_true += len(gt_boxes)

        # Estimate true positives as the minimum of correct and predicted boxes
        true_pos = min(total_true, total_pred)

        precision = true_pos / total_pred if total_pred > 0 else 0
        recall = true_pos / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def reset(self):
        """
        Reset everything to start a new evaluation.
        """
        self.metric.reset()
        self.preds_raw = []
        self.targets_raw = []
