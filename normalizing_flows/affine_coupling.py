from ast import List
import torch
from typing import Optional


class Scale(torch.nn.Module):
    def __init__(self, input_dim: int, scale_init: float = 0.9):
        super().__init__()
        self.scale = self.scale = torch.nn.Parameter(torch.full((input_dim,), scale_init))
        hidden_size = input_dim * 2
        self.ln = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_dim)
        )

    def forward(self, x):
        x = torch.tanh(x)
        x = self.ln(x)
        return x*self.scale

class Translation(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        hidden_size = size * 2
        self.ln = torch.nn.Sequential(
            torch.nn.Linear(size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, size)
        )

    def forward(self, x):
        return self.ln(x)


class AffineCoupling(torch.nn.Module):
    def __init__(self, size: int, s: Optional[Scale] = None, t: Optional[Translation] = None, d: int = 1, mask: Optional[torch.Tensor] = None):
        super().__init__()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.zeros(size)
            self.mask[:d] = 1.
        if s is not None:
            self.s = s
        else:
            self.s = Scale(size)
        if t is not None:
            self.t = t
        else:
            self.t = Translation(size)
        self.d = d
        self.size = size
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(x.size()) == 2
        assert x.size(1) >= self.d
        x_masked = self.mask * x
        s = self.s(x_masked)
        t = self.t(x_masked)
        exp_s = torch.exp(s)
        y = x_masked + (1 - self.mask) * (x * exp_s + t)
        log_det_J = ((1 - self.mask) * s).sum(dim=1)
        return y, log_det_J

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        assert len(y.size()) == 2
        x = torch.empty(y.size())
        x = self.mask * y
        self.exp_ns = torch.exp(-self.s(self.mask*y)) # necessary to compute jacobian
        x += (1-self.mask) * ((y - self.t(self.mask*y)) * self.exp_ns)
        return x

    def test_identity(self, tolerance = 1e-6):
        size = (3, self.size)
        x = torch.rand(size)
        y,_ = self.forward(x)
        recovered = self.inverse(y)
        print(torch.min(recovered-x), torch.max(recovered-x))
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)

import torch.nn as nn

class BatchNormFlow(torch.nn.Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.log_gamma = torch.nn.Parameter(torch.zeros(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.training = True

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0, unbiased=False)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        y = torch.exp(self.log_gamma) * x_hat + self.beta

        log_det_J = self.log_gamma - 0.5 * torch.log(batch_var + self.eps)
        return y, log_det_J.sum(dim=-1)

    def inverse(self, y):
        x_hat = (y - self.beta) / torch.exp(self.log_gamma)
        x = x_hat * torch.sqrt(self.running_var + self.eps) + self.running_mean
        return x
    
    def test_identity(self, tolerance=1e-6):
        size = (3, self.log_gamma.size(0))
        x = torch.rand(size)
        y, _ = self.forward(x)
        self.training = False
        y, _ = self.forward(x)
        recovered = self.inverse(y)
        print(torch.min(recovered-x), torch.max(recovered-x))
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)

class NormalizingFlow(torch.nn.Module):
    def __init__(self, input_dim, num_layers, masks:list[Optional[torch.Tensor]]=[]):
        if len(masks) == 0:
            masks = [None] * num_layers
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if masks[i] is not None:
                mask = masks[i]
            else:
                mask = torch.zeros(input_dim)
                mask[:input_dim//2] = 1.
            self.layers.append(AffineCoupling(size=input_dim, mask=mask))
            self.layers.append(BatchNormFlow(input_dim))

    def forward_train(self, x):
        log_det_J = torch.zeros(x.size(0))
        for layer in self.layers:
            if isinstance(layer, BatchNormFlow):
                x, log_det = layer(x)
                log_det_J += log_det
            else:
                x, log_det = layer.forward(x)
                log_det_J += log_det
        return x, log_det_J

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z
    
    def test_identity(self, tolerance=1e-6):
        size = (3, self.layers[0].size)
        x = torch.rand(size)
        _, _ = self.forward_train(x)
        self.eval()
        z, _ = self.forward_train(x)
        recovered = self.inverse(z)
        print(torch.min(recovered-x), torch.max(recovered-x))
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)
    
