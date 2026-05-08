import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    sequential_loss_path = "results/loss_history.npy"
    random_loss_path = "results_bonus/random_loss_history.npy"

    if not os.path.exists(sequential_loss_path):
        raise FileNotFoundError(f"Cannot find {sequential_loss_path}")

    if not os.path.exists(random_loss_path):
        raise FileNotFoundError(f"Cannot find {random_loss_path}")

    sequential_loss = np.load(sequential_loss_path)
    random_loss = np.load(random_loss_path)

    output_dir = "DD2424 Assignment 4 Bonus Report"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "sequential_vs_random_loss.png")

    sequential_updates = np.arange(1, len(sequential_loss) + 1)
    random_updates = np.arange(1, len(random_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(sequential_updates, sequential_loss, label="Sequential order")
    plt.plot(random_updates, random_loss, label="Random sequence order")
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.title("Sequential vs Random Sequence Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved comparison plot to: {output_path}")
    print()
    print("Sequential training:")
    print(f"  final smooth loss: {sequential_loss[-1]:.4f}")
    print(f"  minimum smooth loss: {sequential_loss.min():.4f}")
    print()
    print("Random-sequence training:")
    print(f"  final smooth loss: {random_loss[-1]:.4f}")
    print(f"  minimum smooth loss: {random_loss.min():.4f}")


if __name__ == "__main__":
    main()
