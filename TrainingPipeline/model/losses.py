import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.helpers as helpers


class VICRegLoss(nn.Module):
    """VICReg loss 
    """

    def __init__(self, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def _sim_loss(self, x, y):
        return F.mse_loss(x, y)

    def _var_loss(self, x):
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(1 - std_x))
        return std_loss

    def _cov_loss(self, x):
        n, d = x.size()
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (n - 1)
        off_diag_cov = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        cov_loss = (off_diag_cov ** 2).sum() / d
        return cov_loss

    def forward(self, x, y):
        sim_loss = self._sim_loss(x, y)
        var_loss = self._var_loss(x) + self._var_loss(y)
        cov_loss = self._cov_loss(x) + self._cov_loss(y)

        loss = self.sim_coeff * sim_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss
        return loss, sim_loss, var_loss, cov_loss
    