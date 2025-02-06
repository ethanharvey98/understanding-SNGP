# PyTorch
import torch

class RandomFeatureGaussianProcess(torch.nn.Module):
    def __init__(self, in_features, out_features, lengthscale=20, outputscale=1.0, rank=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.rank = rank
        self.register_buffer('feature_weight', torch.randn(self.rank, self.in_features))
        self.register_buffer('feature_bias', 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)
                                        
    def featurize(self, h):
        features = torch.nn.functional.linear(h, (1/self.lengthscale) * self.feature_weight, self.feature_bias)
        return (2/self.rank)**0.5 * torch.cos(features)
        
    def forward(self, h):
        batch_size, hidden_dim = h.shape
        features = self.outputscale * self.featurize(h)
        logits = self.linear(features)
        return logits

class VariationalLinear(torch.nn.Module):
    def __init__(self, layer, sigma_param, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.sigma_param = sigma_param
        self.use_posterior = use_posterior
        
    def forward(self, x):
        if self.training or self.use_posterior:
            return torch.nn.functional.linear(
                x,
                self.variational_weight,
                self.variational_bias,
            )
            
        return self.layer(x)
    
    @property
    def variational_weight(self):
        return self.layer.weight + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.weight).to(self.layer.weight.device)
            
    @property
    def variational_bias(self):
        return self.layer.bias + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.bias).to(self.layer.bias.device) if self.layer.bias is not None else None
