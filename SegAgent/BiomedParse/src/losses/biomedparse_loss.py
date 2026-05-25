import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Assuming MedSamLoss and DiceLoss classes are imported or defined above

def batch_edge_masks(masks: torch.Tensor) -> torch.Tensor:
    # masks: [B,1,H,W]
    x = masks.float()                         # [B,1,H,W]
    x = F.pad(x, (1,1,1,1), mode='replicate')    # [B,1,H+2,W+2]
    dil = F.max_pool2d(x, 3, stride=1)           # [B,1,H,W]
    ero = -F.max_pool2d(-x, 3, stride=1)         # [B,1,H,W]
    grad = dil - ero
    return (1*(grad>0)).float()  # [B,1,H,W]
class HungarianMatcher(nn.Module):
    def __init__(self):
        super(HungarianMatcher, self).__init__()

    def forward(self, pred_masks, target_masks):
        batch_size, num_queries, height, width = pred_masks.shape
        pred_masks_flat = pred_masks.view(
            batch_size, num_queries, -1
        )  # Flatten height and width
        target_masks_flat = target_masks.view(
            batch_size, -1
        ).float()  # Flatten height and width and convert to float

        matched_pred_indices = []
        matched_target_indices = []

        for i in range(batch_size):
            pred_flat = pred_masks_flat[i]
            target_flat = target_masks_flat[i]

            # Calculate pairwise cost
            cost_matrix = torch.cdist(pred_flat, target_flat.unsqueeze(0), p=2).squeeze(
                0
            )

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            matched_pred_indices.append(torch.tensor(row_ind, device=pred_masks.device))
            matched_target_indices.append(
                torch.tensor(col_ind, device=target_masks.device)
            )

        return matched_pred_indices, matched_target_indices


class SEEMLoss(nn.Module):
    def __init__(self, matcher=None, loss=None):
        super(SEEMLoss, self).__init__()
        self.matcher = matcher if matcher else HungarianMatcher()
        self.loss = loss if loss else torch.nn.CrossEntropyLoss()

    def forward(self, predictions, labels):

        # temperature = predictions["logit_scale"]
        # # pred_masks = predictions["pred_masks"]
        target_masks = labels

        # matched_pred_indices, matched_target_indices = self.matcher(
        #     pred_masks, target_masks
        # )

        # batch_size = len(matched_pred_indices)

        batch_size, num_masks, height, width = predictions.shape
        mask_pred_results = []
        for idx in range(batch_size):
            pred_gmasks = predictions[idx]
            if height != target_masks.shape[-2] or width != target_masks.shape[-1]:
                pred_gmasks = torch.nn.functional.interpolate(
                    pred_gmasks[None,],
                    size=target_masks.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]
            # v_emb = predictions["pred_gtexts"][idx]
            # t_emb = predictions["class_emb"][idx]

            # # v1 similarity
            # t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            # v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

            # logits = torch.matmul(v_emb, t_emb.t())
            # out_prob = temperature.exp().clamp(max=100) * logits
            # matched_id = out_prob.max(0)[1]
            # matched_mask = pred_gmasks[matched_id, :, :][None, :, :]
            mask_pred_results.append(pred_gmasks)

            # losses = []
            # for i in range(batch_size):
            #     pred_indices = matched_pred_indices[i]
            #     # print(pred_indices)
            #     target_indices = matched_target_indices[i]

            #     selected_pred_masks = pred_masks[i, pred_indices]
            #     selected_target_masks = target_masks[i, target_indices]

            #     # cross entropy between logits and selected target masks
            #     pred_gmasks = predictions["pred_gmasks"][i]
            #     v_emb = predictions["pred_gtexts"][i]
            #     t_emb = predictions["class_emb"][i]

            #     t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            #     v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

            #     logits = torch.matmul(v_emb, t_emb.t())  # [101]
            #     logits = logits.unsqueeze(0)

            #     ce = torch.nn.CrossEntropyLoss()
            #     ce_loss = ce(logits, pred_indices)

            # Register a hook on the tensors only if they require gradients
            # if selected_pred_masks.requires_grad:
            #     selected_pred_masks.register_hook(
            #         lambda grad: print(f"Grad of selected_pred_masks: {grad}")
            #     )

            # Note: We do not register a hook on `selected_target_masks` because it typically does not require gradients

            # loss = self.loss(selected_pred_masks, selected_target_masks) + ce_loss
            # losses.append(loss)
        mask_preds = torch.stack(mask_pred_results, dim=0)
        total_loss = self.loss(mask_preds, target_masks)

        # # Register a hook on the total loss
        # if total_loss.requires_grad:
        #     total_loss.register_hook(lambda grad: print(f"Grad of total_loss: {grad}"))

        return total_loss

import time
class BiomedParseLossCLS(nn.Module):
    # SEEM loss with classification for object detection
    def __init__(self, matcher=None, loss=None, cls_coeff=1.0, pos_weight=3.0, edge_coeff=0.0):
        super(BiomedParseLossCLS, self).__init__()
        self.matcher = matcher if matcher else HungarianMatcher()
        self.loss_fn = loss if loss else torch.nn.CrossEntropyLoss(reduction="none")
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))
        self.cls_coeff = cls_coeff
        self.edge_coeff = edge_coeff

    def forward(self, predictions, labels):

        batch_size, num_masks, height, width = labels.shape
        labels = labels.view(batch_size * num_masks, 1, height, width)
        target_masks = (1*(labels > 0)).float()  # [B*N,1,H,W]
        pred_gmasks = predictions["pred_gmasks"]
        if pred_gmasks.shape[-2:] != (height, width):
            pred_gmasks = F.interpolate(
                pred_gmasks,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        existence_target = (target_masks.flatten(1).sum(dim=1) > 0).float()  # [B*N]
        seg_loss = self.loss_fn(pred_gmasks, target_masks)
        cls_loss = self.cls_loss_fn(
            predictions["object_existence"].view(batch_size * num_masks),
            existence_target,
        )
        
        if self.edge_coeff > 0:
            edge_masks = predictions["edge_masks"]
            if edge_masks.shape[-2:] != (height, width):
                edge_masks = F.interpolate(edge_masks, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
            target_edges = batch_edge_masks(labels)
            edge_loss = self.loss_fn(edge_masks.float(), target_edges.float())
            
        total_loss = self.cls_coeff * cls_loss + seg_loss * existence_target \
            + self.edge_coeff * edge_loss * existence_target
        return total_loss.mean()
        
        # for idx in range(batch_size * num_masks):
        #     if target_masks[idx].sum() == 0:
        #         # classificaiton loss with 0 
        #         total_loss += self.cls_loss(
        #             predictions["object_existence"][idx], 
        #             torch.tensor([0.0], device=predictions["object_existence"].device))
        #     else:
        #         total_loss += self.loss(pred_gmasks[idx], target_masks[idx])
        #         total_loss += self.cls_loss(
        #             predictions["object_existence"][idx], 
        #             torch.tensor([1.0], device=predictions["object_existence"].device))
        # print("SEEMLossCLS time: ", time.time()-t0)
        # return total_loss / num_masks / batch_size