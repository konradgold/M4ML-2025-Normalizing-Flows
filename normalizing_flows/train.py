from math import log
import torch
import torch.utils.data as data
from affine_coupling import NormalizingFlow
from config import Config
from sklearn.datasets import make_moons
from scipy.spatial.distance import directed_hausdorff



def compute_loss(data, model):
    z, log_det_J = model.forward_train(data)
    log_pz = -0.5 * ((z ** 2) + torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)
    return -(log_pz + log_det_J).sum()


def train_loop(config, model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_history = []
    hausdorff_hist = []
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            loss = compute_loss(batch[0], model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Store loss for plotting
            loss_history.append(loss.item())
        # Clear previous console output before printing new log
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}") # type: ignore
        if epoch % config.log_interval == 0:
            # Compute Hausdorff distance between generated samples and original data
            # Convert tensors to numpy arrays if needed
            z_samples = torch.randn(1000, config.input_dim)
            x_samples = model.inverse(z_samples).detach().numpy()
            X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
            X_np = X

            x_samples = x_samples/ x_samples.std(axis=0) * X_np.std(axis=0) - x_samples.mean(axis=0)  + X_np.mean(axis=0)

            hausdorff_ab = directed_hausdorff(x_samples, X_np)[0]
            hausdorff_ba = directed_hausdorff(X_np, x_samples)[0]
            hausdorff_dist = max(hausdorff_ab, hausdorff_ba)
            hausdorff_hist.append(hausdorff_dist)
            print(f"Hausdorff distance: {hausdorff_dist:.4f}")
    return model, loss_history, hausdorff_hist
