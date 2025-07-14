import torch
from typing import Optional


class Scale(torch.nn.Module):
    def __init__(self, input_dim: int, scale_init: float = 0.4):
        self.scale = self.scale = nn.Parameter(torch.full((input_dim,), scale_init))

    def forward(self, x):
        x = torch.tanh(x)
        return x*self.scale

class Translation(torch.nn.Module):
    def __init__(self, size: int):
        self.ln = torch.nn.Linear(int, int, bias = True)

    def forward(self, x):
        return self.ln(x)


class AffineCoupling(torch.nn.Module):
    def __init__(self, size: int, s: Optional[Scale] = None, t: Optional[Translation] = None, d: int = 2, mask: Optional[torch.Tensor] = None):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2
        assert x.size(1) >= self.d
        y = torch.empty(x.size())
        y = self.mask * x
        self.exp_s =torch.exp(self.s(x[:,:d])) # necessary to compute jacobian
        y += (1-self.mask) * (x * (self.exp_s + self.t(x)))
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        assert len(y.size()) == 2
        assert y.size(1) >= self.d
        x = torch.empty(y.size())
        x = self.mask * y
        self.exp_ns = torch.exp(-self.s(y)) # necessary to compute jacobian
        x[:,d:] = (1-self.mask) * (y - self.t(y)) * self.exp_ns
        return y

    def test_identity(self):
        size = (4, 3*d)
        x = torch.rand(size)
        y = self.forward(x)
        tolerance = 1e-6
        assert torch.allclose(x, self.inverse(y), atol=tolerance, rtol=tolerance)


    

    
