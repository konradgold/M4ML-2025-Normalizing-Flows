import torch
from typing import Optional


class Scale(torch.nn.Module):
    """
    A simple linear layer that outputs a vector of the same size as the input.
    """
    def __init__(self, input_dim: int, out_dim: int, hidden_size: Optional[int] = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_dim * 2
        self.ln = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        return self.ln(x)


class Translation(torch.nn.Module):
    """
    A simple linear layer that outputs a vector of the same size as the input.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_size: Optional[int] = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = in_dim * 2
        self.ln = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        return self.ln(x)
    

class Permute(torch.nn.Module):
    """
    This layer randomly permutes the input features. It is used to randomize the masking in the affine coupling layers.
    """
    def __init__(self, num_features):
        super().__init__()
        self.perm = torch.randperm(num_features)
        self.inv_perm = torch.argsort(self.perm)

    def forward(self, x):
        return x[:, self.perm], 0.0  # no log det

    def inverse(self, x):
        return x[:, self.inv_perm]


class AffineCoupling(torch.nn.Module):
    def __init__(self, size: int, d: int = 1, mask: Optional[torch.Tensor] = None, hidden_size: Optional[int] = None):
        super().__init__()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.zeros(size)
            self.mask[:d] = 1.
        self.mask = self.mask.bool()
    
        # The output size must be decreased, to that learning is independant of the masking.
        self.s = Scale(int(self.mask.sum().item()), size - int(self.mask.sum().item()), hidden_size=hidden_size)
        self.t = Translation(int(self.mask.sum().item()), size - int(self.mask.sum().item()), hidden_size=hidden_size)
        self.d = d
        self.size = size
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(x.size()) == 2
        assert x.size(1) >= self.d
        y = torch.empty_like(x)
        p_masked = x[:, self.mask]
        n_masked = x[:, ~self.mask] # negated mask
        s = self.s(p_masked)
        t = self.t(p_masked)
        exp_s = torch.exp(s)
        y[:, self.mask] = p_masked
        y[:, ~self.mask] = n_masked * exp_s + t
        log_det_J = s.sum(dim=1)
        return y, log_det_J

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        assert len(y.size()) == 2
        x = torch.empty(y.size())
        p_y = y[:, self.mask]
        n_y = y[:, ~self.mask]
        self.exp_ns = torch.exp(-self.s(p_y)) 
        x[:, self.mask] = p_y
        x[:, ~self.mask] = (n_y - self.t(p_y)) * self.exp_ns
        return x

    def test_identity(self, tolerance = 1e-6):
        size = (3, self.size)
        x = torch.rand(size)
        y,_ = self.forward(x)
        recovered = self.inverse(y)
        print(f"Min: {torch.min(recovered-x)}, Max: {torch.max(recovered-x)}")
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)
        return True


class NormalizingFlow(torch.nn.Module):
    def __init__(self, input_dim, num_layers, mask: Optional[torch.Tensor] = None, hidden_size: Optional[int] = None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if mask is None:
            # mask first dimensions by default
            mask = torch.zeros(input_dim)
            mask[:input_dim//2] = 1.
        for i in range(num_layers):
            self.layers.append(AffineCoupling(size=input_dim, mask=mask, hidden_size=hidden_size))
            self.layers.append(Permute(input_dim))

    def forward_train(self, x):
        log_det_J = torch.zeros(x.size(0))
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_J += log_det
        return x, log_det_J

    def inverse(self, z):
        for layer in reversed(self.layers):
            # layers are applied in reverse order during inference, inversion already implemented
            z = layer.inverse(z) # type: ignore
        return z
    
    def test_identity(self, tolerance=1e-6):
        size = (3, self.layers[0].size)
        x = torch.rand(size) # type: ignore
        self.eval()
        z, _ = self.forward_train(x)
        recovered = self.inverse(z)
        print(f"Min: {torch.min(recovered-x)}, Max: {torch.max(recovered-x)}")
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)
        return True
    
