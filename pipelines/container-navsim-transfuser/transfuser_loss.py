# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# This file contains code derived from NAVSIM
# (https://github.com/autonomousvision/navsim)
# Copyright: University of Tübingen, Tübingen AI Center, and contributors
# Original License: Apache License 2.0
#
# Changes from the original:
#   - Replaced navsim imports with local references

from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from transfuser_config import TransfuserConfig
from transfuser_model import BoundingBox2DIndex


def transfuser_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TransfuserConfig,
) -> torch.Tensor:
    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )
    loss = (
        config.trajectory_weight * trajectory_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
    )
    return loss


def _agent_loss(targets, predictions, config):
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for c in cost]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat(
        [t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0
    ).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(
        pred_valid_idx, gt_valid_idx, reduction="none"
    )
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid, pred_logits):
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()
    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    return ce_cost.permute(0, 2, 1)


@torch.no_grad()
def _get_l1_cost(gt_states, pred_states, gt_valid):
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)
    return l1_cost.permute(0, 2, 1)


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
