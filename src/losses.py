# PyTorch
import torch

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        return {'loss': nll, 'nll': nll}

class DataEmphasizedELBo(torch.nn.Module):
    def __init__(self, kappa, sigma_param, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion
        self.kappa = kappa
        self.sigma_param = sigma_param

    def forward(self, labels, logits, params, N=1):
        nll = self.criterion(logits, labels)
        loc_diff_norm = (params**2).sum()
        lambda_star = (loc_diff_norm/len(params)) + torch.nn.functional.softplus(self.sigma_param)**2
        term1 = (torch.nn.functional.softplus(self.sigma_param)**2/lambda_star) * len(params)
        term2 = (1/lambda_star) * loc_diff_norm
        term3 = (len(params) * torch.log(lambda_star)) - (len(params) * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        kl = (1/2) * (term1 + term2 - len(params) + term3)
        return {'kl': kl, 'lambda_star': lambda_star, 'loss': self.kappa * N * nll + kl, 'nll': nll}