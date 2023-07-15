import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def reg_l1_loss(pred, target, mask):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


class DETloss(nn.Module):
    def __init__(self, alpha=1):
        super(DETloss, self).__init__()
        self.alpha = alpha

    def forward(self, hm, wh, offset, batch):
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        c_loss = focal_loss(hm, batch_hms)
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

        loss = c_loss + wh_loss + off_loss

        return loss, c_loss, wh_loss + off_loss
