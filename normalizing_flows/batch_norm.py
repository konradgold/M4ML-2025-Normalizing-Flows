import torch

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

    def invert(self, y):
        x_hat = (y - self.beta) / torch.exp(self.log_gamma)
        x = x_hat * torch.sqrt(self.running_var + self.eps) + self.running_mean
        return x
    
    def test_identity(self, tolerance=1e-6):
        size = (3, self.log_gamma.size(0))
        x = torch.rand(size)
        y, _ = self.forward(x)
        self.training = False
        y, _ = self.forward(x)
        recovered = self.invert(y)
        print(torch.min(recovered-x), torch.max(recovered-x))
        assert torch.allclose(recovered, x, atol=tolerance, rtol=tolerance)