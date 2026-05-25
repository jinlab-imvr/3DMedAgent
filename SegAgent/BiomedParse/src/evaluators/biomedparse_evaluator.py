from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F
from azureml.acft.image.components.olympus.evaluators.base import BaseOlympusEvaluator, CleanPredictions

import numpy as np
import os, json
import matplotlib.pyplot as plt


@dataclass
class SEEMPredictions:
    predictions: Dict
    labels: torch.Tensor
    gold_labels: torch.Tensor

def batch_edge_masks(masks: torch.Tensor) -> torch.Tensor:
    # masks: [B,1,H,W]
    x = masks.float()                            # [B,1,H,W]
    x = F.pad(x, (1,1,1,1), mode='replicate')    # [B,1,H+2,W+2]
    dil = F.max_pool2d(x, 3, stride=1)           # [B,1,H,W]
    ero = -F.max_pool2d(-x, 3, stride=1)         # [B,1,H,W]
    grad = dil - ero
    return (grad>0).to(torch.uint8)

class BiomedParseEvaluator(BaseOlympusEvaluator[SEEMPredictions]):

    def postprocess(self, model_outputs):
        low_res_pred = torch.sigmoid(model_outputs)
        image_seg = (low_res_pred > 0.5).int()
        return image_seg

    def predict(self, model: torch.nn.Module, batch: dict) -> SEEMPredictions:
        # given a model and a batch, return the model's predictions as a custom
        # ResultType
        outputs = model(batch, mode="eval")

        predictions = outputs["predictions"]

        gold_labels = batch.get("labels", None)

        batch_size, num_masks, height, width = gold_labels.shape
        gold_labels = gold_labels.view(batch_size * num_masks, 1, height, width)
        
        mask_preds = predictions["pred_gmasks"]
        if mask_preds.shape[-2:] != (height, width):
            mask_preds = F.interpolate(
                mask_preds,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        
        edge_masks = predictions["edge_masks"]
        if edge_masks.shape[-2:] != (height, width):
            edge_masks = F.interpolate(edge_masks, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        predictions["edge_masks"] = self.postprocess(edge_masks) * (predictions["object_existence"]>0
                                               ).int().unsqueeze(-1).unsqueeze(-1)

        gold_labels = gold_labels.to(mask_preds.device)
        
        predictions["edge_label"] = batch_edge_masks(gold_labels)

        mask_preds = self.postprocess(mask_preds)
        predictions["raw_masks"] = mask_preds
        
        predicted_labels = mask_preds * (predictions["object_existence"]>0
                                               ).int().unsqueeze(-1).unsqueeze(-1)

        predictions = SEEMPredictions(
            predictions=predictions, labels=predicted_labels, gold_labels=gold_labels
        )
        return predictions

    def evaluate(
        self,
        batch: dict,
        predictions: SEEMPredictions,
        loss_function: torch.nn.Module,
        metric_stage: str = "test",
    ) -> Dict[str, torch.Tensor]:
        # given a batch and pre-computed predictions, return a dictionary of metrics
        # that includes the loss in the "loss" key

        if "labels" not in batch:
            raise ValueError("Batch does not contain 'labels' key")

        metrics = {}
        loss = loss_function(predictions.predictions, predictions.gold_labels)
        metrics["test_loss"] = loss
        num_samples = predictions.gold_labels.size(0)
        metrics["sum_num_samples"] = num_samples

        gold_labels = 1 * (predictions.gold_labels > 0)  # [B*N,1,H,W]
        base_metrics = self._get_core_metrics(
            predictions.labels, gold_labels, metric_stage=metric_stage
        )
        metrics.update(base_metrics)
        
        # # raw mask predictions
        # raw_masks = predictions.predictions["raw_masks"]
        # raw_metrics = self._get_core_metrics(
        #     raw_masks, gold_labels, metric_stage=f'{metric_stage}_raw'
        # )
        # metrics.update(raw_metrics)
        
        # edge masks predictions
        edge_masks = predictions.predictions["edge_masks"]
        edge_labels = predictions.predictions["edge_label"]
        edge_metrics = self._get_core_metrics(
            edge_masks, edge_labels, metric_stage=f'{metric_stage}_edge'
        )
        metrics.update(edge_metrics)
        
        # extract metrics
        existence_target = (predictions.gold_labels.flatten(1).sum(dim=1) > 0).float()  # [B*N]
        existence_pred = (predictions.predictions["object_existence"]>0).int().view(num_samples)
        metrics["existence_accuracy"] = (existence_target == existence_pred).float().mean()
        metrics["positive"] = existence_target.mean()
        metrics["true_positive"] = (existence_target * existence_pred).mean()
        metrics["false_positive"] = ((1 - existence_target) * existence_pred).mean()
        metrics["false_negative"] = (existence_target * (1 - existence_pred)).mean()

        return metrics

    def clean_predictions(self, predictions: SEEMPredictions) -> CleanPredictions:
        # given precomputed predictions, return a list of per-item results to save.
        # These should be JSON-serialable dictionaries.
        batch_outputs = [
            {"predicted": label.tolist(), "gold": gold.tolist()}
            for label, gold in zip(predictions.labels, predictions.gold_labels)
        ]
        return batch_outputs


class MultiClassEvaluator(BaseOlympusEvaluator[SEEMPredictions]):
    """Evaluator for masks with multiple classes.
    Model inference is done for each prompt/prompt separately, and the results are
    merged into a single mask with multiple classes, with overlapping classes
    being resolved.
    """

    def postprocess(self, model_outputs, object_existence, num_prompts, class_ids=None, threshold=0.5):
        bs = num_prompts.shape[0]
        h, w = model_outputs.shape[-2:]
        multiclass_masks = torch.zeros((bs, h, w), dtype=torch.int64)
        start = 0
        for i in range(bs):
            n = num_prompts[i]
            masks = model_outputs[start : start + n]
            scores = object_existence[start : start + n]
            keep = nms_masks_batch_iou(masks > 0.5, scores, iou_threshold=0.5, score_threshold=threshold)
            
            if len(keep) == 0:
                start += n
                continue
            if len(keep) == 1:
                multiclass_masks[i] = (keep[0]+1) * (masks[keep[0]] > 0.5)
                start += n
                continue
            
            # zero out the masks that are not kept
            discard = [j for j in range(n) if j not in keep]
            masks[discard] = 0
            # stack a background mask with probability 0.5
            bg_mask = 0.5 * torch.ones_like(masks[0:1])
            keep_masks = torch.cat([bg_mask, masks], dim=0)
            class_mask = keep_masks.argmax(dim=0)
            
            ids = class_ids[i].split("&")
            id_map = {j+1: int(ids[j]) for j in range(len(ids)) if j+1 != int(ids[j])}
            if len(id_map) > 0:
                # fix id missmatch
                orig_mask = class_mask.clone()
                for j in id_map:
                    class_mask[orig_mask == j] = id_map[j]
             
            multiclass_masks[i] = class_mask
            start += n
            
        return multiclass_masks
        
        

    def predict(self, model: torch.nn.Module, batch: dict) -> SEEMPredictions:
        # given a model and a batch, return the model's predictions as a custom
        # ResultType
        outputs = model(batch, mode="eval")

        predictions = outputs["predictions"]
        
        class_ids = batch["class_ids"]

        gold_labels = batch.get("labels", None)

        batch_size, height, width = gold_labels.shape
        
        mask_preds = predictions["pred_gmasks"]
        if mask_preds.shape[-2:] != (height, width):
            mask_preds = F.interpolate(
                mask_preds,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )    # [num_preds, 1, h, w]
        mask_preds = mask_preds.squeeze(1)  # [num_preds, h, w]
        
        num_prompts = predictions["num_prompts"]
        object_existence = predictions["object_existence"].squeeze(1)
        
        predicted_labels = self.postprocess(mask_preds.detach().to(torch.float32).sigmoid(), 
                                            object_existence.detach().to(torch.float32).sigmoid(), 
                                            num_prompts, class_ids=class_ids, threshold=0.5)
        
        
        # extra predictions with different threshold
        predicted_labels_025 = self.postprocess(mask_preds.detach().to(torch.float32).sigmoid(),
                                            object_existence.detach().to(torch.float32).sigmoid(), 
                                            num_prompts, class_ids=class_ids, threshold=0.25)
        
        predicted_labels_0 = self.postprocess(mask_preds.detach().to(torch.float32).sigmoid(),
                                            object_existence.detach().to(torch.float32).sigmoid(), 
                                            num_prompts, class_ids=class_ids, threshold=0)
        
        predictions["predicted_labels_025"] = predicted_labels_025
        predictions["predicted_labels_0"] = predicted_labels_0

        predictions = SEEMPredictions(
            predictions=predictions, labels=predicted_labels, gold_labels=gold_labels
        )
        return predictions

    def evaluate(
        self,
        batch: dict,
        predictions: SEEMPredictions,
        loss_function: torch.nn.Module,
        metric_stage: str = "test",
    ) -> Dict[str, torch.Tensor]:
        # given a batch and pre-computed predictions, return a dictionary of metrics
        # that includes the loss in the "loss" key

        if "labels" not in batch:
            raise ValueError("Batch does not contain 'labels' key")

        metrics = {}
        num_samples = predictions.gold_labels.size(0)
        metrics["sum_num_samples"] = num_samples

        thresh_pred = {'05': predictions.labels.cpu().numpy(),
                       '025': predictions.predictions["predicted_labels_025"].cpu().numpy(),
                       '0': predictions.predictions["predicted_labels_0"].cpu().numpy()}
        
        extra_metrics = {}
        for k, v in thresh_pred.items():
            dice, extra = self.seg_metrics(v, predictions.gold_labels.cpu().numpy())
            metrics[f"dice_score_{k}"] = dice
            extra_metrics[k] = extra
        
        save_predictions = True
        if save_predictions and metric_stage == "test":
            # save predictions
            model_name = self.save_name
            for i in range(num_samples):
                mask_file = batch["mask_file"][i]
                mask_path = '/'.join(mask_file.split("/")[:-1])
                mask_name = mask_file.split("/")[-1]

                pred_path = mask_path.replace('val_mask', f'predictions/{model_name}')
                result_path = mask_path.replace('val_mask', f'metrics/{model_name}')
                if pred_path == mask_path:
                    raise ValueError("pred_path is the same as mask_path")
                if result_path == mask_path:
                    raise ValueError("result_path is the same as mask_path")
                os.makedirs(pred_path, exist_ok=True)
                os.makedirs(result_path, exist_ok=True)

                # save predictions
                mask = predictions.labels[i].cpu().numpy().astype(np.uint8)
                plt.imsave(f"{pred_path}/{mask_name}", mask, cmap='gray')

                # save metrics results as json
                output_matrics = {k: v for k, v in metrics.items() if isinstance(v, torch.Tensor)}
                output_matrics["mask_name"] = mask_name
                output_matrics['instance_label'] = batch["instance_label"][i].item()
                
                for k, v in extra_metrics.items():
                    for metric_name, metric_value in v.items():
                        output_matrics[f"{metric_name}_{k}"] = metric_value[i]
                
                with open(f"{result_path}/{mask_name}.json", 'w') as f:
                    json.dump(output_matrics, f)

        return metrics

    def clean_predictions(self, predictions: SEEMPredictions) -> CleanPredictions:
        # given precomputed predictions, return a list of per-item results to save.
        # These should be JSON-serialable dictionaries.
        batch_outputs = [
            {"predicted": label.tolist(), "gold": gold.tolist()}
            for label, gold in zip(predictions.labels, predictions.gold_labels)
        ]
        return batch_outputs
    
    def seg_metrics(self, pred, gt):
        # evaluation metrics
        dice_scores = []
        values = {'dice_scores': [], 'intersections': [], 'gt_sums': [], 'pred_sums': []}
        for i in range(len(pred)):
            pred_mask = pred[i]
            gt_mask = gt[i]
            # compute dice score
            dsc, I, T, P = compute_multi_class_dsc(gt_mask, pred_mask)
            dice_scores.append(dsc)
            values['dice_scores'].append(dsc)
            values['intersections'].append(I)
            values['gt_sums'].append(T)
            values['pred_sums'].append(P)
        return np.mean(dice_scores), values
        
        

def non_max_suppression(masks, scores, iou_threshold=0.5, score_threshold=0.5):
    # masks: a tensor of masks, each mask is a binary 2D PyTorch tensor (shape: [N, H, W])
    # scores: a tensor of scores for each mask (shape: [N])
    # iou_threshold: the IoU threshold for non-max suppression
    # score_threshold: the score threshold for selecting a mask
    # output: a list of indices of selected masks

    # Sort scores in descending order
    indices = torch.argsort(scores, descending=True)
    keep = []

    while len(indices) > 0:
        i = indices[0]
        if scores[i] < score_threshold:
            break
        if torch.sum(masks[i]) == 0:  # Skip empty masks
            indices = indices[1:]
            continue
        
        # Add the mask to the output
        keep.append(i.item())  # Convert tensor to scalar for appending

        if len(indices) == 1:
            break

        ious = []
        for j in indices[1:]:
            # Calculate IoU between mask i and mask j
            intersection = torch.sum(masks[i] & masks[j])  # Element-wise AND
            union = torch.sum(masks[i] | masks[j])  # Element-wise OR
            iou = intersection.float() / union.float()  # Compute IoU as a float
            ious.append(iou.item())  # Convert tensor to scalar

        # Update indices to remove masks with IoU above threshold
        indices = indices[1:][torch.tensor(ious) < iou_threshold]

    return keep

def nms_masks_batch_iou(masks: torch.Tensor,
                        scores: torch.Tensor,
                        iou_threshold: float = 0.5,
                        score_threshold: float = 0.5):
    """
    masks: (N, H, W) binary (0/1 or bool) tensor
    scores: (N,) tensor of confidence scores
    returns: List[int] of kept indices
    """
    
    # ensure bool for logical ops
    masks = masks.bool()
    # sort in descending score order
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        # stop if below score threshold
        if scores[i] < score_threshold:
            break
        # skip empty masks
        if masks[i].sum() == 0:
            order = order[1:]
            continue

        keep.append(i)
        if order.numel() == 1:
            break

        # batch compute IoUs of mask[i] vs all remaining
        cur_mask = masks[i]                     # (H, W)
        other_masks = masks[order[1:]]          # (M, H, W)
        # intersection / union per mask
        inter = (other_masks & cur_mask).view(other_masks.size(0), -1).sum(1).float()
        union = (other_masks | cur_mask).view(other_masks.size(0), -1).sum(1).float()
        ious = inter / union                    # (M,)

        # keep only those with IoU <= threshold
        remaining = torch.nonzero(ious <= iou_threshold, as_tuple=False).squeeze(1)
        order = order[1:][remaining]
    
    return keep


def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_gt = mask_gt.sum()
  volume_pred = mask_pred.sum()
  if volume_gt + volume_pred == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / (volume_gt+volume_pred), volume_intersect, volume_gt, volume_pred

def compute_multi_class_dsc(gt, seg):
    label_ids = np.unique(np.vstack([gt, seg]))[1:]
    dsc = []
    class_inter = {}
    class_gt = {}
    class_pred = {}
    for idx, i in enumerate(label_ids):
        gt_i = gt == i
        seg_i = seg == i
        dice, vol_inter, vol_gt, vol_pred = compute_dice_coefficient(gt_i, seg_i)
        if vol_gt > 0:
            dsc.append(dice)
        class_inter[int(i)] = float(vol_inter)
        class_gt[int(i)] = float(vol_gt)
        class_pred[int(i)] = float(vol_pred)
    
    if len(label_ids) == 0:
        dsc = 1.0 * (seg.max() == 0)
    else:
        dsc = np.nanmean(dsc)

    return dsc, class_inter, class_gt, class_pred