# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import logging

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                cost_kpts: float = 1, cost_ctrs: float = 1, cost_deltas: float = 1, cost_kpts_class: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.cost_kpts = cost_kpts
        self.cost_ctrs = cost_ctrs
        self.cost_deltas = cost_deltas
        self.cost_kpts_class = cost_kpts_class
        assert cost_class != 0 or cost_kpts != 0 or cost_kpts_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        

        out_prob = outputs["pred_logits"].flatten(0,1).softmax(-1)  # [num_queries, num_classes]
        out_kpts = outputs["pred_kpts"].flatten(0, 1)  # [batch_size * num_queries, 53] 

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_kpts = torch.cat([v["keypoints"] for v in targets])

        num_persons = tgt_kpts.shape[0]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        out_kpts_ids = out_kpts[:, 4::3] #kpt visability (model does not predict center visability. No Vcenter in index 2)
        tgt_kpts_ids = tgt_kpts[:, 5::3].clone()#kpt visability (target keypoints have Vcenter in index 2)

        x_kpts = torch.cat((out_kpts[:, 0].unsqueeze(-1), out_kpts[:, 2::3]), dim=1) 
        y_kpts = torch.cat((out_kpts[:, 1].unsqueeze(-1), out_kpts[:, 3::3]), dim=1) 
        
        
        # MAA: Mask2Former Original
        # out_mask = outputs["pred_masks"][b] # [num_queries, H_pred, W_pred]
        # gt masks are already padded when preparing target
        # tgt_mask = targets[b]["masks"].to(out_mask)

        out_mask = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, H_pred, W_pred]
        tgt_mask = torch.cat([v["masks"] for v in targets]).to(out_mask.device)  # [sum(num_masks), H_pred, W_pred]

        out_mask = out_mask[:, None].float() #MAA: turned into float
        tgt_mask = tgt_mask[:, None].float() #MAA: turned into float

        # all masks share the same set of points for efficient matching!
        point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
        # get gt labels
        tgt_mask = point_sample(
            tgt_mask,
            point_coords.repeat(tgt_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        out_mask = point_sample(
            out_mask,
            point_coords.repeat(out_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        # Preparing keypoints for computing matching loss on absolute positions
        #Prediction Keypoints
        x_kpts_abs, y_kpts_abs = x_kpts.clone(), y_kpts.clone()
        x_kpts_abs[:, 1:] = (x_kpts_abs[:, 1:] - 0.5) * 2
        y_kpts_abs[:, 1:] = (y_kpts_abs[:, 1:] - 0.5) * 2
        x_kpts_abs[:, 1:] += x_kpts_abs[:, 0].clone().unsqueeze_(1)
        y_kpts_abs[:, 1:] += y_kpts_abs[:, 0].clone().unsqueeze_(1)
        #Target (Ground Truth)
        tgt_kpts_abs = tgt_kpts.clone()
        tgt_kpts_abs[:, 3::3] = (tgt_kpts_abs[:, 3::3] - 0.5) * 2
        tgt_kpts_abs[:, 4::3] = (tgt_kpts_abs[:, 4::3] - 0.5) * 2
        tgt_kpts_abs[:, 3::3] += tgt_kpts_abs[:, 0].clone().unsqueeze_(1)
        tgt_kpts_abs[:, 4::3] += tgt_kpts_abs[:, 1].clone().unsqueeze_(1)

        with autocast(enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
        
        # Compute the keypoint loss

        if all(tgt_kpts[:,2].clone().flatten()): # all centers are visible (e.g. center of mass) #MAA: All instances has Vcenter=1
            visible_mask = (tgt_kpts[:, 2::3].clone() == 1) #MAA: keypoints with Vcenter=1
            #MAA: centers and deltas positions
            out_kpts = torch.stack((x_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                    y_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                    #dim=2).reshape(-1, 36, num_persons)
                                    dim=2).reshape(bs*num_queries, 8, num_persons)
            tgt_kpts = torch.stack((tgt_kpts[:, 0::3].clone() * visible_mask,
                                    tgt_kpts[:, 1::3].clone() * visible_mask), dim=2).view(-1, 8)
            #MAA: abs positions
            out_kpts_abs = torch.stack((x_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                        y_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                        dim=2).reshape(bs*num_queries, 8, num_persons)
            tgt_kpts_abs = torch.stack((tgt_kpts_abs[:, 0::3].clone() * visible_mask,
                                        tgt_kpts_abs[:, 1::3].clone() * visible_mask), dim=2).view(-1, 8)
            
                               
            cost_kpts = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L1 cost on absolute positions
            cost_ctrs = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L2 cost on centers
            cost_deltas = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L1 cost on deltas
            for i in range(num_persons):
                cost_deltas[:, i, None] = torch.cdist(out_kpts[:,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=1)
                #cost_deltas[:, i, None] = torch.cdist(out_kpts[:,:,i], tgt_kpts[i,:].unsqueeze(0), p=1)
                cost_kpts[:, i, None] = torch.cdist(out_kpts_abs[:,2:,i], tgt_kpts_abs[i,2:].unsqueeze(0), p=1)
                cost_ctrs[:, i, None] = torch.cdist(out_kpts[:,:2,i], tgt_kpts[i,:2].unsqueeze(0), p=2)

            # Compute the L2 cost between keypoints classes
            cost_kpts_class = torch.cdist(out_kpts_ids, tgt_kpts_ids, p=2)
            
        else:
            #have mask on predicted keypoints (and keypoints classes) when target keypoints are invisible
            visible_mask = (tgt_kpts[:, 2::3].clone() == 1) #MAA: all instances with target keypoints with V=1 (for all keypoits, not limited to Vcenters)
            visible_mask[torch.where(visible_mask[:, 0] == 0)] = False
            out_kpts = torch.stack((x_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                    y_kpts.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                    #dim=2).reshape(-1, 36, num_persons)
                                    dim=2).reshape(bs*num_queries, 8, num_persons)
            tgt_kpts = torch.stack((tgt_kpts[:, 0::3].clone() * visible_mask,
                                    tgt_kpts[:, 1::3].clone() * visible_mask), dim=2).view(-1, 8)
            out_kpts_abs = torch.stack((x_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T,
                                        y_kpts_abs.unsqueeze(2).expand(-1,-1,num_persons) * visible_mask.T),
                                        dim=2).reshape(bs*num_queries, 8, num_persons)
            tgt_kpts_abs = torch.stack((tgt_kpts_abs[:, 0::3].clone() * visible_mask,
                                        tgt_kpts_abs[:, 1::3].clone() * visible_mask), dim=2).view(-1, 8)

            src_kpts_ids = out_kpts_ids.clone()
            target_kpts_ids = tgt_kpts_ids.clone()
            # FS: Debug prints before the error line
            logging.debug(f"visible_mask shape: {visible_mask.shape}")
            logging.debug(f"src_kpts_ids shape: {src_kpts_ids.shape}")
            src_kpts_ids[torch.where(visible_mask[:, 0] == 0)] = 0

            # FS: Debug prints before the error line
            #logging.debug(f"visible_mask: {visible_mask}")
            logging.debug(f"target_kpts_ids shape: {tgt_kpts_ids.shape}")
            #logging.debug(f"target_kpts_ids: {tgt_kpts_ids}")
            logging.debug(f"Indices to access: {torch.where(visible_mask[:, 0] == 0)}")

            #FS: Error Line 148
            target_kpts_ids[torch.where(visible_mask[:, 0] == 0)] = 0 #CUDA ERROR out of bounds
            cost_kpts = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L1 cost on absolute positions
            cost_ctrs = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L2 cost on centers
            cost_deltas = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L1 cost on deltas
            cost_kpts_class = torch.empty(x_kpts.shape[0], num_persons, device=out_prob.device) # L2 cost

            for i in range(num_persons):
                # L1 cost of keypoint offsets
                cost_deltas[:, i, None] = torch.cdist(out_kpts[:,2:,i], tgt_kpts[i,2:].unsqueeze(0), p=1)
                #cost_deltas[:, i, None] = torch.cdist(out_kpts[:,:,i], tgt_kpts[i,:].unsqueeze(0), p=1)
                # L1 cost of keypoints
                cost_kpts[:, i, None] = torch.cdist(out_kpts_abs[:,2:,i], tgt_kpts_abs[i,2:].unsqueeze(0), p=1)
                # L2 cost of center keypoints
                cost_ctrs[:, i, None] = torch.cdist(out_kpts[:,:2,i], tgt_kpts[i,:2].unsqueeze(0), p=2)
                # L2 cost of keypoint classes
                cost_kpts_class[:, i, None] = torch.cdist(src_kpts_ids, target_kpts_ids[i,:].unsqueeze(0), p=2)

        # Final cost matrix
        C = (self.cost_kpts * cost_kpts 
             + self.cost_ctrs * cost_ctrs 
             + self.cost_deltas * cost_deltas 
             + self.cost_class * cost_class 
             + self.cost_kpts_class * cost_kpts_class
             + self.cost_mask * cost_mask
             + self.cost_dice * cost_dice
        )
        C = C.reshape(bs, num_queries, -1).cpu()

        sizes = [len(v["keypoints"]) for v in targets] #MAA number of instances
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
