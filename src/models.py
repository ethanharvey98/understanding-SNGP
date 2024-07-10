import math
# PyTorch
import torch

#model.forward = models.custom_forward.__get__(model)
def custom_forward(self, x, return_posterior=False):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x, return_posterior=return_posterior)

    return x
        
class RandomFeatureGaussianProcess(torch.nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        lengthscale=math.sqrt(20), 
        rank=1024, 
        record_cov_inv=False, 
        tau=1e-6,
    ):
        super(RandomFeatureGaussianProcess, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lengthscale = lengthscale
        self.rank = rank
        self.record_cov_inv = record_cov_inv
        self.tau = tau
        
        self.register_buffer('feature_weight', (1/self.lengthscale**2) * torch.randn(self.rank, self.in_features))
        self.register_buffer('feature_bias', 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)
        
        self.register_buffer('running_cov_inv', torch.zeros(self.rank, self.rank))
        self.register_buffer('running_dataset_size', torch.tensor(0))
        self.register_buffer('cov', torch.zeros(self.rank, self.rank))
                    
    def featurize(self, h):
        features = torch.nn.functional.linear(h, -self.feature_weight, self.feature_bias)
        return (2 / self.rank) ** 0.5 * torch.cos(features)
        
    def forward(self, h, return_posterior=False):
        
        features = self.featurize(h)
        logits = self.linear(features)

        if self.record_cov_inv:
            
            batch_cov_inv = 2 * features.T @ features
            self.running_cov_inv += batch_cov_inv
            self.running_dataset_size += len(features)
            
        # TODO: if return_posterior == True: return torch.distributions.MultivariateNormal()
            
        return logits
    
    def invert_cov_inv(self):
        self.cov = torch.linalg.pinv(self.tau * torch.eye(self.rank).to(self.running_cov_inv.device) + self.running_cov_inv / self.running_dataset_size)
                
    def register_buffers(self):
        self.register_buffer('running_cov_inv', torch.zeros(self.rank, self.rank))
        self.register_buffer('running_dataset_size', torch.tensor(0))
        self.register_buffer('cov', torch.zeros(self.rank, self.rank))

    def reset_cov_inv(self):
        self.running_cov_inv = torch.zeros(self.rank, self.rank)
        self.running_dataset_size = torch.tensor(0)
        self.cov = torch.zeros(self.rank, self.rank)
        