import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    loss_path = "results/loss_history.npy"

    if not os.path.exists(loss_path):
        raise FileNotFoundError(f"Cannot find {loss_path}")

    loss_history = np.load(loss_path)

    output_dir = "DD2424 Assignment 4 Report"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "smooth_loss_100000.png")

    updates = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(updates, loss_history)
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.title("Smooth Loss During Training")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved smooth loss plot to: {output_path}")
    print(f"Number of updates plotted: {len(loss_history)}")
    print(f"Initial smooth loss: {loss_history[0]:.4f}")
    print(f"Final smooth loss: {loss_history[-1]:.4f}")
    print(f"Minimum smooth loss: {loss_history.min():.4f}")


if __name__ == "__main__":
    main()
