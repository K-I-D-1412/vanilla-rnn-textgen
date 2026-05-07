import numpy as np
import torch

from rnn import (
    read_book_data,
    build_char_mappings,
    chars_to_one_hot,
    initialize_rnn_parameters,
    forward_pass,
    backward_pass,
)


def torch_forward(RNN_np, X_np, Y_np, h0_np):
    # Compute the forward pass using PyTorch autograd.
    # This should match our NumPy forward pass.
    U = torch.tensor(RNN_np["U"], dtype=torch.float64, requires_grad=True)
    W = torch.tensor(RNN_np["W"], dtype=torch.float64, requires_grad=True)
    V = torch.tensor(RNN_np["V"], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(RNN_np["b"], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(RNN_np["c"], dtype=torch.float64, requires_grad=True)

    X = torch.tensor(X_np, dtype=torch.float64)
    Y = torch.tensor(Y_np, dtype=torch.float64)
    h = torch.tensor(h0_np, dtype=torch.float64)

    seq_length = X.shape[1]
    loss = 0.0

    for t in range(seq_length):
        x_t = X[:, t:t + 1]
        y_t = Y[:, t:t + 1]

        a_t = W @ h + U @ x_t + b
        h = torch.tanh(a_t)
        o_t = V @ h + c

        p_t = torch.softmax(o_t, dim=0)

        loss = loss - torch.log(torch.sum(y_t * p_t) + 1e-12)

    loss = loss / seq_length

    params = {
        "U": U,
        "W": W,
        "V": V,
        "b": b,
        "c": c,
    }

    return loss, params


def relative_error(grad_np, grad_torch):
    # Compute relative error between two gradients.
    numerator = np.max(np.abs(grad_np - grad_torch))
    denominator = max(
        1e-12,
        np.max(np.abs(grad_np)) + np.max(np.abs(grad_torch)),
    )

    return numerator / denominator


def main():
    book_path = "data/goblet_book.txt"

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, _ = build_char_mappings(book_data)

    K = len(unique_chars)
    m = 10
    seq_length = 25

    RNN = initialize_rnn_parameters(K, m=m, seed=42)

    X_chars = book_data[:seq_length]
    Y_chars = book_data[1:seq_length + 1]

    X = chars_to_one_hot(X_chars, char_to_ind, K).astype(np.float64)
    Y = chars_to_one_hot(Y_chars, char_to_ind, K).astype(np.float64)
    h0 = np.zeros((m, 1), dtype=np.float64)

    loss_np, cache, _ = forward_pass(RNN, X, Y, h0)
    grads_np = backward_pass(RNN, cache)

    loss_torch, params_torch = torch_forward(RNN, X, Y, h0)
    loss_torch.backward()

    print("Gradient checking with PyTorch autograd")
    print("--------------------------------------")
    print("NumPy loss: ", loss_np)
    print("Torch loss: ", loss_torch.item())
    print("Loss abs difference:", abs(loss_np - loss_torch.item()))
    print()

    for key in ["U", "W", "V", "b", "c"]:
        grad_np = grads_np[key]
        grad_torch = params_torch[key].grad.detach().numpy()

        max_abs_diff = np.max(np.abs(grad_np - grad_torch))
        rel_err = relative_error(grad_np, grad_torch)

        print(f"{key}:")
        print(f"  shape: {grad_np.shape}")
        print(f"  max abs diff: {max_abs_diff:.12e}")
        print(f"  relative error: {rel_err:.12e}")


if __name__ == "__main__":
    main()