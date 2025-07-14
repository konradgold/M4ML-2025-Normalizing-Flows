import torch
from typing import Optional


class Scale(torch.nn.Module):
    def __init__(self, input_dim: int, scale_init: float = 0.4):
        super().__init__()
        self.scale = self.scale = torch.nn.Parameter(torch.full((input_dim,), scale_init))

    def forward(self, x):
        x = torch.tanh(x)
        return x*self.scale

class Translation(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ln = torch.nn.Linear(size, size, bias = True)

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
        self.size = size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2
        assert x.size(1) >= self.d
        y = torch.empty(x.size())
        y = self.mask * x
        self.exp_s = torch.exp(self.s(self.mask*x)) # necessary to compute jacobian
        y += (1-self.mask) * (x * self.exp_s + self.t(self.mask*x))
        return y

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
        y = self.forward(x)
        recovered = self.inverse(y)
        print(torch.min(recovered-x), torch.max(recovered-x))
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)

    

    
