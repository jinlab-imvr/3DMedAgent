import numpy as np
import torch
import torch.nn.functional as F


def get_axis(img):
    # get the axis to slice the 3D volume
    shape = img.shape
    # get shape difference between the axes
    diff_ratio = [2*abs(shape[1]-shape[2])/(shape[1]+shape[2]),
            2*abs(shape[0]-shape[2])/(shape[0]+shape[2]),
            2*abs(shape[0]-shape[1])/(shape[0]+shape[1])]
    
    if diff_ratio[0] < 0.5:
        valid_axis = 0
    else:
        min_axis = np.argmin(shape)
        valid_axis = min_axis
        
    return valid_axis
    
def get_padding(vol):
    shape = vol.shape[1:]
    if shape[0] > shape[1]:
        pad1 = (shape[0] - shape[1]) // 2
        pad2 = (shape[0] - shape[1]) - pad1
        pad_width = [[0, 0], [0, 0], [pad1, pad2]]
    else:
        pad1 = (shape[1] - shape[0]) // 2
        pad2 = (shape[1] - shape[0]) - pad1
        pad_width = [[0, 0], [pad1, pad2], [0, 0]]
    padded_size = max(shape)
    return pad_width, padded_size


def remove_padding(vol, pad_width):
    if pad_width is not None:
        l1 = int(pad_width[1][0])
        r1 = int(vol.shape[1] - pad_width[1][1])
        l2 = int(pad_width[2][0])
        r2 = int(vol.shape[2] - pad_width[2][1])
        vol = vol[:, l1:r1, l2:r2]
    return vol


def pad_and_resize(vol, size):
    pad_width, padded_size = get_padding(vol)
    if pad_width is not None:
        vol = np.pad(vol, pad_width, mode="constant", constant_values=0)
    vol = torch.from_numpy(vol).unsqueeze(0)
    resized_vol = F.interpolate(
        vol, size=(size, size), mode="bicubic", align_corners=False
    )
    return resized_vol.squeeze(0), pad_width, padded_size


def process_input(vol, size):
    # vol: 3D np.ndarray
    # size: int
    
    valid_axis = get_axis(vol)
    vol = np.moveaxis(vol, valid_axis, 0)
    
    # pad to square with equal padding on both sides
    vol, pad_width, padded_size = pad_and_resize(vol, size)
    
    return vol, pad_width, padded_size, valid_axis


def process_output(vol, pad_width, padded_size, valid_axis):
    # vol: torch.Tensor with batch size 1
    # pad_width: tuple
    # padded_size: int
    # valid_axis: int
    
    if vol.shape[-1] != padded_size or vol.shape[-2] != padded_size:
        vol = F.interpolate(
            vol.unsqueeze(0).float(), size=(padded_size, padded_size), mode="nearest", # align_corners=False
        )
        vol = vol.squeeze(0).int()
            
    vol = vol.cpu().numpy()
    vol = remove_padding(vol, pad_width)
    vol = np.moveaxis(vol, 0, valid_axis)
    
    return vol


def slice_nms(mask_preds, scores, iou_threshold=0.5, score_threshold=0.5):
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