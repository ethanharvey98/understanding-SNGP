import os
import copy
# PyTorch
import torch

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def flatten_params(model, deepcopy=True):
    if deepcopy: model = copy.deepcopy(model)
    return torch.cat([param.detach().view(-1) for param in model.parameters()])

class _BoundedSpectralNorm(torch.nn.utils.parametrizations._SpectralNorm):
    """
    _BoundedSpectralNorm extends the _SpectralNorm class from PyTorch and adds a bound.
    
    Reference:
        For the original _SpectralNorm class implementation, see:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrizations.py
    """
    def __init__(self, weight, spec_norm_iteration=1, spec_norm_bound=1.0, dim=0, eps=1e-12):
        super(_BoundedSpectralNorm, self).__init__(weight, spec_norm_iteration, dim=0, eps=1e-12)
        self.spec_norm_bound = spec_norm_bound

    def forward(self, weight):
        if weight.ndim == 1:
            sigma = torch.linalg.vector_norm(weight)
            #return F.normalize(weight, dim=0, eps=self.eps)
            return self.spec_norm_bound * F.normalize(weight, dim=0, eps=self.eps) if self.spec_norm_bound < sigma else weight
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            #return weight / sigma
            return self.spec_norm_bound * weight / sigma if self.spec_norm_bound < sigma else weight
        
def apply_bounded_spectral_norm(model, name='weight', spec_norm_iteration=1, spec_norm_bound=1.0):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, _BoundedSpectralNorm(getattr(module, name, None), spec_norm_iteration=spec_norm_iteration, spec_norm_bound=spec_norm_bound)
            )
    return module
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}

    for images, labels in dataloader:
        
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        probabilities = torch.softmax(logits, dim=1)
        acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
        metrics['acc1'] += batch_size/dataset_size*acc1.item()
        metrics['acc5'] += batch_size/dataset_size*acc5.item()
        metrics['loss'] += batch_size/dataset_size*loss.item()
        
        if lr_scheduler:
            lr_scheduler.step()
            
    return metrics

def evaluate(model, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}
            
    with torch.no_grad():
        for images, labels in dataloader:
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, labels)
            
            batch_size = len(images)
            probabilities = torch.softmax(logits, dim=1)
            acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
            metrics['acc1'] += batch_size/dataset_size*acc1.item()
            metrics['acc5'] += batch_size/dataset_size*acc5.item()
            metrics['loss'] += batch_size/dataset_size*loss.item()
    
    return metrics
