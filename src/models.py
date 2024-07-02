# PyTorch
import torch
        
class RandomFeatureGaussianProcess(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=1024):
        super(RandomFeatureGaussianProcess, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.register_buffer('W', 0.05 * torch.randn(self.rank, self.in_features))
        self.register_buffer('b', 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)
        
    def forward(self, x):
        x = torch.nn.functional.linear(x, -self.W, self.b)
        x = (2 / self.rank) ** 0.5 * torch.cos(x)
        x = self.linear(x)
        return x
    