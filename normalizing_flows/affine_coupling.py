import torch
from typing import Optional

class AffineCoupling(torch.nn.Module):
    def init(self, s: torch.nn.Module, t: torch.nn.Module, d: int = 2, mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.s = s
        self.t = t
        self.d = d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2
        assert x.size(1) >= self.d
        y = torch.empty(x.size())
        y[:,:d] = x[:,d]
        self.exp_s = torch.exp(self.s(x[:,:d])) # necessary to compute jacobian
        y[:, d:] = x[:, d:] * (self.exp_s + self.t(x[:,:d]))
        return y

    

    
