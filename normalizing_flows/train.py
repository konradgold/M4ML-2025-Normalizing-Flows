from math import log
import torch
import torch.utils.data as data
from normalizing_flows.affine_coupling import NormalizingFlow
from normalizing_flows.config import Config
from sklearn.datasets import make_moons



def compute_loss(data, model):
    z, log_det_J = model.forward_train(data)
    log_pz = -0.5 * ((z ** 2) + torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)
    return -(log_pz + log_det_J).sum()

config = Config('config.yaml')

X, _ = make_moons(n_samples=20000, noise=0.1)
x_tensor = torch.tensor(X, dtype=torch.float32)
dataset = data.TensorDataset(x_tensor)
dataloader = data.DataLoader(dataset, batch_size=256, shuffle=True)

model = NormalizingFlow(input_dim=config.input_dim, num_layers=config.num_layers)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(config.num_epochs):
    for batch in dataloader:
        loss = compute_loss(batch[0], model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Sampling and visualization (optional)
with torch.no_grad():
    model.eval()
    z_samples = torch.randn(1000, config.input_dim)
    x_samples = model.inverse(z_samples)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.title("Generated Samples")
    plt.scatter(x_samples[:, 0], x_samples[:, 1], alpha=0.5)
    plt.show()
