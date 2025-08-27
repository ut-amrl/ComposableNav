from omegaconf import OmegaConf
import torch.nn as nn 
import numpy as np
import torch


class MSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss(*args, **kwargs)
    
    def forward(self, pred, target):
        return self.mse_loss(pred, target)


class WeightedMSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='none', weights=None):
        super().__init__()
        self.mse_loss = nn.MSELoss(size_average, reduce, reduction)
        if weights is not None:
            weights = OmegaConf.to_container(weights)
            assert isinstance(weights, list), "weights should be either None or a list of numbers"
            weights = torch.FloatTensor(weights)
            self.register_buffer('weights', weights)
        else:
            self.weights = None

    def forward(self, pred, target):
        if self.weights is not None:
            assert self.weights.shape[0] == pred.shape[-1]
            weights = self.weights.expand_as(pred)
            pred = pred * torch.sqrt(weights)
            target = target * torch.sqrt(weights)
        return self.mse_loss(pred, target)
    

class AngularWeightedMSELoss(WeightedMSELoss):
    def __init__(self, angle_dim, size_average=None, reduce=None, reduction='none', weights=None):
        super().__init__(size_average, reduce, reduction, weights)
        self.angle_dim = angle_dim
        self.reduction = reduction
    
    def forward(self, pred, target):
        assert self.angle_dim < pred.shape[-1]

        # Extract the angle dimension
        pred_angle = pred[..., self.angle_dim].unsqueeze(-1) * torch.pi
        target_angle = target[..., self.angle_dim].unsqueeze(-1) * torch.pi

        # Compute sin and cos of the angles
        pred_trigo = torch.cat([torch.sin(pred_angle), torch.cos(pred_angle)], dim=-1) / np.sqrt(2)
        target_trigo = torch.cat([torch.sin(target_angle), torch.cos(target_angle)], dim=-1) / np.sqrt(2)

        # Replace the angle dimension with sin and cos
        pred_modified = torch.cat([
            pred[..., :self.angle_dim], pred_trigo, pred[..., self.angle_dim+1:]
        ], dim=-1)
        target_modified = torch.cat([
            target[..., :self.angle_dim], target_trigo, target[..., self.angle_dim+1:]
        ], dim=-1)

        if self.weights is not None:
            assert self.weights.shape[0] == pred.shape[-1]

            weights_modified = torch.cat([
                self.weights[:self.angle_dim],
                self.weights[self.angle_dim].repeat(2),  # Repeat the angle weight for sin and cos
                self.weights[self.angle_dim+1:]
            ])
            weights_modified = weights_modified.expand_as(pred_modified)
            pred_modified = pred_modified * torch.sqrt(weights_modified)
            target_modified = target_modified * torch.sqrt(weights_modified)

        mse_loss = self.mse_loss(pred_modified, target_modified)
        if self.reduction == 'none':
            angle_loss = mse_loss[..., self.angle_dim:self.angle_dim+2].sum(dim=-1)
            mse_loss = torch.cat([
                mse_loss[..., :self.angle_dim],
                angle_loss.unsqueeze(-1),
                mse_loss[..., self.angle_dim+2:]
            ], dim=-1)
                    
        return mse_loss

