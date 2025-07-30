import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.datasets import make_moons
from affine_coupling import NormalizingFlow
from config import Config
from train import train_loop
import random


# --- Set seed helper ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_stages(stages, title_prefix="", filename=None):
    """Helper to plot each stage of the transformation """
    n_stages = len(stages)
    cols = int(np.ceil(n_stages / 2))  # split into 2 rows
    rows = 2

  
    plt.figure(figsize=(4 * cols, 3 * rows))  
    for i, (data_t, title) in enumerate(stages):
        pts = data_t.detach().cpu().numpy()
        plt.subplot(rows, cols, i + 1)
        plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.4)
        plt.title(f"{title_prefix}{title}", fontsize=12, fontweight="bold")  
        plt.axis("equal")

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=200, bbox_inches="tight") 
        print(f"Saved plot: {filename}")
    plt.close()  




def visualize_forward(model, x_tensor, config):
    """Data x → latent z"""
    with torch.no_grad():
        stages = [(x_tensor, "Original Data")]
        current = x_tensor
        coupling_count = 0

        for layer in model.layers:
            out = layer.forward(current)
            y = out[0] if isinstance(out, tuple) else out
            current = y
            if layer.__class__.__name__ == "AffineCoupling":
                coupling_count += 1
                if coupling_count < config.num_layers:
                    stages.append((current, f"After {coupling_count} Coupling Layer(s)"))
                else:
                    stages.append((current, "Latent Space (z)"))

        plot_stages(stages, filename="plots/forward.png")


def visualize_inverse(model, config):
    """Latent z → data x"""
    with torch.no_grad():
        z_samples = torch.randn(config.samples, config.input_dim)  # seeded RNG
        stages = [(z_samples, "Latent Space (z)")]
        current = z_samples
        coupling_count = 0

        for layer in reversed(model.layers):
            y = layer.inverse(current)
            current = y
            if layer.__class__.__name__ == "AffineCoupling":
                coupling_count += 1
                if coupling_count < config.num_layers:
                    stages.append((current, f"After {coupling_count} Inverse Coupling Layer(s)"))
                else:
                    stages.append((current, "Reconstructed Data (x)"))

        plot_stages(stages, filename="plots/inverse.png")


if __name__ == "__main__":

    config = Config('config.yaml')
    set_seed(config.seed)

    # Load and prepare data
    X, _ = make_moons(n_samples=config.samples, noise=0.1)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = data.TensorDataset(x_tensor)
    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize and train the model
    model = NormalizingFlow(input_dim=config.input_dim,
                            num_layers=config.num_layers,
                            hidden_size=config.hidden_size)
    model.train()
    model, loss_history, hausdorff_hist = train_loop(config, model, dataloader)
    model.eval()

    # Visualizations
    visualize_forward(model, x_tensor, config)
    visualize_inverse(model, config)

    # Training Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_loss.png", dpi=300, bbox_inches="tight")
    print("Saved plot: plots/training_loss.png")
    plt.show()

    # Hausdorff Distance Curve
    plt.figure(figsize=(6, 4))
    plt.plot(hausdorff_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Hausdorff Distance")
    plt.title("Training Hausdorff Distance")
    plt.savefig("plots/hausdorff.png", dpi=300, bbox_inches="tight")
    print("Saved plot: plots/hausdorff.png")
    plt.show()
