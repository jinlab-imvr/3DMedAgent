import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
from skimage import segmentation
from skimage.measure import label
from scipy.optimize import linear_sum_assignment

from .SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def compute_multi_class_dsc(gt, seg, label_ids):
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    dsc = [None] * len(present_labels)
    for idx, i in enumerate(present_labels):
        gt_i = gt == i
        seg_i = seg == i
        dsc[idx] = compute_dice_coefficient(gt_i, seg_i)
    return np.nanmean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, label_ids, tolerance=2.0):
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    nsd = [None] * len(present_labels)
    for idx, i in enumerate(present_labels):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd[idx] = compute_surface_dice_at_tolerance(surface_distance, tolerance)
    return np.nanmean(nsd)

def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------
    iou: ND-array, float
        matrix of IOU pairs of size [masks_true.max()+1, masks_pred.max()+1]
        iou[i, j] is the IoU between ground truth instance i+1 and predicted instance j+1.
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    matched_pairs = [(t, p) for t, p, ok in zip(true_ind, pred_ind, match_ok) if ok]
    return tp, matched_pairs

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg>0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        tp, matched_pairs = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        matched_pairs = None
        
    return tp, fp, fn, matched_pairs



def evaluate_segmentation(gt_npz, seg_npz, instance_label, class_ids, spacing=None):
    # Load ground truth and segmentation masks
    # gt_npz = np.load(gt_path, allow_pickle=True)['gts']
    # seg_npz = np.load(seg_path, allow_pickle=True)['segs']

    # gt_npz = gt_npz.astype(np.uint8)
    # seg_npz = seg_npz.astype(np.uint8)

    class_ids_array = np.array(class_ids, dtype=np.int32)
    
    metric = {}

    if instance_label == 0:     # semantic masks
        # note: the semantic labels may not be sequential
        dsc = compute_multi_class_dsc(gt_npz, seg_npz, class_ids_array)
        # nsd = compute_multi_class_nsd(gt_npz, seg_npz, spacing, class_ids_array)
        f1_score = np.nan
        dsc_tp = np.nan
    elif instance_label == 1:  # instance masks
        dsc = compute_multi_class_dsc(gt_npz>0, seg_npz>0, class_ids_array)
        f1_score = np.nan
        dsc_tp = np.nan
    
        # Calculate F1 instead
        if len(np.unique(seg_npz)) == 2:
            print("converting segmentation to instance masks")
            # convert prediction masks from binary to instance
            tumor_inst = label(seg_npz, connectivity=1)

            # put the tumor instances back to gt_data_ori
            seg_npz[tumor_inst > 0] = (tumor_inst[tumor_inst > 0] + np.max(seg_npz))
        gt_npz = label(gt_npz, connectivity=1)
        gt_npz = segmentation.relabel_sequential(gt_npz)[0]
        seg_npz = segmentation.relabel_sequential(seg_npz)[0]

        tp, fp, fn, matched_pairs = eval_tp_fp_fn(gt_npz, seg_npz)        # default f1 overlap threshold is 0.5
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # compute DSC for TP cases
        if matched_pairs:
            dsc_list = []
            for gt_idx, pred_idx in matched_pairs:
                gt_mask = gt_npz == (gt_idx + 1)
                pred_mask = seg_npz == (pred_idx + 1)
                dsc_value = compute_dice_coefficient(gt_mask, pred_mask)
                dsc_list.append(dsc_value)
            dsc_tp = np.mean(dsc_list)
        else:
            dsc_tp = 0

        # Set DSC and NSD to None for instance masks
        dsc = None
        nsd = None

    metric['DSC'] = round(dsc, 4) if dsc is not None else np.nan
    # metric['NSD'] = round(nsd, 4) if nsd is not None else np.nan
    metric['F1'] = round(f1_score, 4) if f1_score is not None else np.nan
    metric['DSC_TP'] = round(dsc_tp, 4) if dsc_tp is not None else np.nan

    return metric
