import numpy as np
# PyTorch
import torch
import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.d = 0 # Number of backbone parameters
        self.criterion = criterion

    def forward(self, logits, labels, bb_params, clf_params):
        nll = self.criterion(logits, labels)
        losses = {'loss': nll, 'nll': nll, 'bb_log_prior': torch.tensor(0.0), 'clf_log_prior': torch.tensor(0.0)}
        return losses
    
class L2NormLoss(nn.Module):
    def __init__(self, weight_decay, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.weight_decay = weight_decay
        self.d = 0 # Number of backbone parameters
        self.criterion = criterion

    def forward(self, logits, labels, bb_params, clf_params):
        nll = self.criterion(logits, labels)
        clf_log_prior = self.weight_decay * (clf_params**2).sum()/2
        clf_log_prior = torch.clamp(clf_log_prior, min=-1e20, max=1e20)
        loss = nll + clf_log_prior
        losses = {'loss': loss, 'nll': nll, 'bb_log_prior': torch.tensor(0.0), 'clf_log_prior': clf_log_prior}
        return losses
    
class MAPAdaptationLoss(nn.Module):
    def __init__(self, loc, weight_decay, criterion=nn.CrossEntropyLoss(), m=6):
        super().__init__()
        self.loc = loc
        self.locs = torch.stack([torch.load(f'/cluster/tufts/hugheslab/eharve06/torchvision-checkpoints/{index}.pt') for index in range(m)])
        self.weight_decay = weight_decay
        self.d = 0 # Number of backbone parameters
        self.criterion = criterion

    def forward(self, logits, labels, bb_params, clf_params):
        nll = self.criterion(logits, labels)
        
        #clf_log_prior = self.weight_decay * ((clf_params-self.loc.to(clf_params.device))**2).sum()/2
        #clf_log_prior = torch.clamp(clf_log_prior, min=-1e20, max=1e20)
        #loss = nll + clf_log_prior
        #losses = {'loss': loss, 'nll': nll, 'bb_log_prior': torch.tensor(0.0), 'clf_log_prior': clf_log_prior}

        num_locs, d = self.locs.shape
        clf_log_prior = self.weight_decay * ((clf_params[d:]-torch.zeros_like(clf_params[d:]).to(clf_params.device))**2).sum()/2
        bb_log_prior = torch.logsumexp(torch.log(torch.softmax(torch.rand(num_locs), dim=0).to(clf_params.device)) -(self.weight_decay * ((clf_params[:d]-self.locs.to(clf_params.device))**2).sum(dim=1)/2), dim=0)
        clf_log_prior = torch.clamp(clf_log_prior, min=-1e20, max=1e20)
        bb_log_prior = torch.clamp(bb_log_prior, min=-1e20, max=1e20)
        loss = nll - bb_log_prior + clf_log_prior
        losses = {'loss': loss, 'nll': nll, 'bb_log_prior': bb_log_prior, 'clf_log_prior': clf_log_prior}
        
        return losses
        
class MAPTransferLearning(nn.Module):
    # Note: There are more effient ways to train StdPrior, LearnedPriorIso, and LearnedPriorLR. Our implementation focuses on making the template for our probabilistic model easy to read.
    def __init__(
        self, 
        bb_prior, # Backbone prior dictionary
        bb_weight_decay, # Backbone weight decay
        clf_prior, # Classifier prior dictionary
        clf_weight_decay, # Classifier weight decay
        device, 
        n, # Training set size
        criterion=nn.CrossEntropyLoss()
    ):        
        assert all(item in bb_prior for item in ['cov_diag', 'cov_factor', 'loc']), 'Backbone prior dictionary must include \'cov_diag\', \'cov_factor\', and \'loc\''
        assert all(item in clf_prior for item in ['cov_diag', 'cov_factor', 'loc']), 'Classifier prior dictionary must include \'cov_diag\', \'cov_factor\', and \'loc\''
        super().__init__()
                                        
        # Backbone multivariate Normal
        if bb_weight_decay != 0:
            self.bb_mvn = LowRankMultivariateNormal(
                loc=(bb_prior['loc']).to(device), 
                cov_factor=(np.sqrt(bb_weight_decay)*bb_prior['cov_factor'].t()).to(device), 
                cov_diag=(bb_weight_decay*bb_prior['cov_diag']).to(device) if 'prior_eps' not in bb_prior else (bb_weight_decay*bb_prior['cov_diag']+bb_prior['prior_eps']).to(device)
            )
        else:
            self.bb_mvn = None
        # Classifier multivariate Normal
        if clf_weight_decay != 0:
            self.clf_mvn = LowRankMultivariateNormal(
                loc=(clf_prior['loc']).to(device), 
                cov_factor=(np.sqrt(clf_weight_decay)*clf_prior['cov_factor'].t()).to(device), 
                cov_diag=(clf_weight_decay*clf_prior['cov_diag']).to(device)
            )
        else:
            self.clf_mvn = None
            
        self.d = len(bb_prior['loc']) # Number of backbone parameters
        self.n = n
        self.criterion = criterion
        
    def forward(self, logits, labels, bb_params, clf_params):
        nll = self.criterion(logits, labels)
        bb_log_prior = torch.zeros_like(nll).to(nll.device) if self.bb_mvn is None else self.bb_mvn.log_prob(bb_params).sum()/self.n
        clf_log_prior = torch.zeros_like(nll).to(nll.device) if self.clf_mvn is None else self.clf_mvn.log_prob(clf_params).sum()/self.n
        bb_log_prior = torch.clamp(bb_log_prior, min=-1e20, max=1e20)
        clf_log_prior = torch.clamp(clf_log_prior, min=-1e20, max=1e20)
        loss = nll - clf_log_prior - bb_log_prior
        losses = {'loss': loss, 'nll': nll, 'bb_log_prior': bb_log_prior, 'clf_log_prior': clf_log_prior}
        return losses