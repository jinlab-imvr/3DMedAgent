from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F
from azureml.acft.image.components.olympus.evaluators.base import BaseOlympusEvaluator, CleanPredictions

import numpy as np
import os, json
import matplotlib.pyplot as plt

from time import time 

from skimage.measure import label
from skimage.segmentation import expand_labels
# from skimage.morphology import remove_small_objects

from ..utils import process_output
from ..CVPR25_text_eval import evaluate_segmentation

@dataclass
class SEEMPredictions:
    predictions: Dict
    labels: torch.Tensor
    gold_labels: torch.Tensor


class BiomedParseEvaluator(BaseOlympusEvaluator[SEEMPredictions]):

    def postprocess(self, model_outputs, object_existence, threshold=0.5, do_nms=True):
        
        if do_nms and model_outputs.shape[0] > 1:
            # do non-max suppression for each slice
            return self.slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(), 
                                           iou_threshold=0.5, score_threshold=threshold)
        
        mask = (model_outputs.sigmoid()) * (
            object_existence.sigmoid() > threshold
            ).int().unsqueeze(-1).unsqueeze(-1)
        return mask
    
    
    def slice_nms(self, mask_preds, scores, iou_threshold=0.5, score_threshold=0.5):
        # do non-max suppression for each slice
        # mask_preds: (N, D, H, W), binary class probability masks
        # scores: (N, D), object existence scores
        # iou_threshold: IoU threshold for non-max suppression
        
        N, D, H, W = mask_preds.shape
        keep_masks = torch.zeros((N, D), dtype=torch.int64, device=mask_preds.device)
        for i in range(D):
            keep = nms_masks_batch_iou(mask_preds[:, i] > 0.5, scores[:,i], 
                                       iou_threshold=iou_threshold, 
                                       score_threshold=score_threshold)
            if len(keep) == 0:
                continue
            # make the kept masks 1 and the rest 0
            keep_masks[keep, i] = 1
            
        return mask_preds * keep_masks.unsqueeze(-1).unsqueeze(-1)
        
    def merge_multiclass_masks(self, masks, ids):
        # stack a background mask with probability 0.5
        bg_mask = 0.5 * torch.ones_like(masks[0:1])
        keep_masks = torch.cat([bg_mask, masks], dim=0)
        class_mask = keep_masks.argmax(dim=0)
        
        id_map = {j+1: int(ids[j]) for j in range(len(ids)) if j+1 != int(ids[j])}
        if len(id_map) > 0:
            # fix id mismatch
            orig_mask = class_mask.clone()
            for j in id_map:
                class_mask[orig_mask == j] = id_map[j]
                
        return class_mask
    
    def seg_metrics(self, pred, gt):
        # evaluation metrics
        dice_scores = []
        values = {'dice_scores': [], 'intersections': [], 'gt_sums': [], 'pred_sums': []}
        pred_mask = pred
        gt_mask = gt
        # compute dice score
        dsc, I, T, P = compute_multi_class_dsc(gt_mask, pred_mask)
        dice_scores.append(dsc)
        values['dice_scores'].append(dsc)
        values['intersections'].append(I)
        values['gt_sums'].append(T)
        values['pred_sums'].append(P)
        return np.mean(dice_scores), values


    def predict(self, model: torch.nn.Module, batch: dict) -> SEEMPredictions:
        # given a model and a batch, return the model's predictions as a custom ResultType
        
        outputs = model(batch, mode="eval")

        predictions = outputs["predictions"]

        gold_labels = batch.get("labels", None)
        
        class_ids = batch["class_ids"]

        # BS, D, H, W = gold_labels.shape
        
        padded_size = batch["padded_size"][0]
        H = padded_size
        W = padded_size
        
        mask_preds = predictions["pred_gmasks"]
        mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", 
                                   align_corners=False, antialias=True)
        masks = self.postprocess(mask_preds, 
                                object_existence=predictions["object_existence"], 
                                threshold=0.5)    # [N, D, H, W]
    
        predictions["mask"] = masks
        predictions["class_ids"] = [int(_) for _ in class_ids[0].split("&")]
        
        # edge masks and interior masks
        edge_masks = predictions["edge_masks"]
        if edge_masks is not None:
            edge_masks = F.interpolate(edge_masks, size=(512, 512), mode="bicubic", 
                                    align_corners=False, antialias=True)
            edge_masks = self.postprocess(edge_masks, 
                                        object_existence=predictions["object_existence"], 
                                        threshold=0.5)
            predictions["edge_masks"] = edge_masks
        
        instance_refine = False
        if batch["instance_label"][0] == 1 and instance_refine:
            t0 = time()
            masks = (masks > 0.5).int().squeeze(0).cpu().numpy()
            inst_pred = masks * ((edge_masks < 0.5).int().squeeze(0).cpu().numpy())    # remove edges
            inst_pred = label(inst_pred, connectivity=1)    # label connected components
            print(f"instance segmentation label time: {time() - t0:.2f} seconds")
            predicted_labels = expand_labels(inst_pred, distance=2) * masks    # expand labels to recover edges
            # convert to torch tensor
            predicted_labels = torch.from_numpy(predicted_labels)
            print(f"instance number: {len(np.unique(inst_pred))}, inference time: {time() - t0:.2f} seconds")
        else:
            predicted_labels = self.merge_multiclass_masks(masks, class_ids[0].split("&"))
        
        predicted_labels = process_output(predicted_labels, batch["pad_width"][0], padded_size, batch["axis"][0])
        
        # # extra predictions with NMS
        # nms_masks = self.slice_nms(mask_preds.sigmoid(), predictions["object_existence"].sigmoid())
        # nms_masks = self.merge_multiclass_masks(nms_masks, class_ids[0].split("&"))
        # predictions[f'predicted_labels_nms'] = process_output(nms_masks, batch["pad_width"][0], padded_size, batch["axis"][0])
        
        predictions = SEEMPredictions(
            predictions=predictions, labels=predicted_labels, gold_labels=gold_labels.to(mask_preds.device)
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
        metrics["inference_time"] = predictions.predictions["inference_time"]

        pred_versions = {'base': predictions.labels } #, 'nms': predictions.predictions['predicted_labels_nms']}
        
        extra_metrics = {}
        for k, v in pred_versions.items():
            if batch["instance_label"][0] == 0:
                dice, extra = self.seg_metrics(v, predictions.gold_labels[0].cpu().numpy())
            else:
                dice, extra = self.seg_metrics(v>0, predictions.gold_labels[0].cpu().numpy()>0)
            metrics[f"dice_score_{k}"] = dice
            extra_metrics[k] = extra
            
        eval_metrics = evaluate_segmentation(
            gt_npz=predictions.gold_labels[0].cpu().numpy(),
            seg_npz=predictions.labels,
            instance_label=batch["instance_label"][0].item(),
            class_ids=predictions.predictions["class_ids"],
        )
        metrics.update(eval_metrics)

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