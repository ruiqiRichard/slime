from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

def get_opd_turn_advantages(
    log_diffs: List[torch.Tensor],
    loss_masks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Compute turn-level advantages for OPD.

    Args:
        log_diffs (List[torch.Tensor]): List of log probability differences per sample.
        loss_masks (List[torch.Tensor]): List of loss masks per sample.

    Returns:
        List[torch.Tensor]: List of turn-level advantages per sample.
    """
    turn_advantages = []
    for log_diff, mask in zip(log_diffs, loss_masks):
        T = log_diff.size(0)
        turn_adv = torch.zeros_like(log_diff)
        i = 0
        while i < T:
            if mask[i] == 1:
                j = i
                while j < T and mask[j] == 1:
                    j += 1
                turn_sum = log_diff[i:j].mean()
                turn_adv[i:j] = turn_sum
                i = j
            else:
                i += 1
        # turn_adv /= (mask.sum() + 1e-8)
        turn_advantages.append(turn_adv)
        
    return turn_advantages