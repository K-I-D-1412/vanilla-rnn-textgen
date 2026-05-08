import time
import numpy as np

from rnn_bonus import (
    read_book_data,
    build_char_mappings,
    chars_to_one_hot,
    initialize_rnn_parameters,
    forward_pass,
    backward_pass,
    softmax,
)


def chars_to_indices(chars, char_to_ind):
    # Convert a string of characters into integer indices.
    return np.array([char_to_ind[ch] for ch in chars], dtype=np.int64)


def fast_forward_pass(RNN, x_indices, y_indices, h0):
    """
    Faster forward pass using character indices instead of one-hot matrices.

    Main speed idea:
    Instead of computing U @ x_t where x_t is one-hot,
    directly use U[:, index].
    """
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]
    b = RNN["b"]
    c = RNN["c"]

    seq_length = len(x_indices)

    a_list = []
    h_list = [h0]
    o_list = []
    p_list = []

    loss = 0.0

    for t in range(seq_length):
        x_idx = x_indices[t]
        y_idx = y_indices[t]

        a_t = W @ h_list[-1] + U[:, x_idx:x_idx + 1] + b
        h_t = np.tanh(a_t)
        o_t = V @ h_t + c
        p_t = softmax(o_t)

        loss += -np.log(p_t[y_idx, 0] + 1e-12)

        a_list.append(a_t)
        h_list.append(h_t)
        o_list.append(o_t)
        p_list.append(p_t)

    loss = loss / seq_length

    cache = {
        "x_indices": x_indices,
        "y_indices": y_indices,
        "a": a_list,
        "h": h_list,
        "o": o_list,
        "p": p_list,
        "h0": h0,
    }

    h_last = h_list[-1]

    return loss, cache, h_last


def fast_backward_pass(RNN, cache):
    """
    Faster backward pass using character indices.

    Main speed ideas:
    1. Avoid multiplying by sparse one-hot x_t.
    2. Update only the active column of grad_U.
    3. Use np.outer for vector outer products.
    """
    W = RNN["W"]
    V = RNN["V"]

    x_indices = cache["x_indices"]
    y_indices = cache["y_indices"]
    h = cache["h"]
    p = cache["p"]

    seq_length = len(x_indices)

    grads = {
        "U": np.zeros_like(RNN["U"]),
        "W": np.zeros_like(RNN["W"]),
        "V": np.zeros_like(RNN["V"]),
        "b": np.zeros_like(RNN["b"]),
        "c": np.zeros_like(RNN["c"]),
    }

    dh_next = np.zeros_like(h[0])

    for t in reversed(range(seq_length)):
        x_idx = x_indices[t]
        y_idx = y_indices[t]

        h_t = h[t + 1]
        h_prev = h[t]

        do = p[t].copy()
        do[y_idx, 0] -= 1.0
        do = do / seq_length

        grads["V"] += np.outer(do[:, 0], h_t[:, 0])
        grads["c"] += do

        dh = V.T @ do + dh_next
        da = dh * (1 - h_t ** 2)

        grads["W"] += np.outer(da[:, 0], h_prev[:, 0])

        # Since x_t is one-hot, only one column of U receives gradient.
        grads["U"][:, x_idx:x_idx + 1] += da

        grads["b"] += da

        dh_next = W.T @ da

    for key in grads:
        grads[key] = np.clip(grads[key], -5, 5)

    return grads


def max_relative_error(a, b):
    numerator = np.max(np.abs(a - b))
    denominator = max(1e-12, np.max(np.abs(a)) + np.max(np.abs(b)))
    return numerator / denominator


def check_correctness(RNN, X, Y, x_indices, y_indices, h0):
    # Check that the fast implementation matches the original implementation.
    loss_original, cache_original, _ = forward_pass(RNN, X, Y, h0)
    grads_original = backward_pass(RNN, cache_original)

    loss_fast, cache_fast, _ = fast_forward_pass(RNN, x_indices, y_indices, h0)
    grads_fast = fast_backward_pass(RNN, cache_fast)

    print("Correctness check")
    print("-----------------")
    print(f"Original loss: {loss_original:.12f}")
    print(f"Fast loss:     {loss_fast:.12f}")
    print(f"Loss abs diff: {abs(loss_original - loss_fast):.12e}")
    print()

    for key in ["U", "W", "V", "b", "c"]:
        rel_err = max_relative_error(grads_original[key], grads_fast[key])
        max_abs_diff = np.max(np.abs(grads_original[key] - grads_fast[key]))
        print(f"{key}: max abs diff = {max_abs_diff:.12e}, relative error = {rel_err:.12e}")

    print()


def benchmark(RNN, X, Y, x_indices, y_indices, h0, num_repeats=2000):
    # Benchmark original forward+backward against fast forward+backward.
    
    # Warm-up
    for _ in range(10):
        loss, cache, _ = forward_pass(RNN, X, Y, h0)
        _ = backward_pass(RNN, cache)

        loss, cache, _ = fast_forward_pass(RNN, x_indices, y_indices, h0)
        _ = fast_backward_pass(RNN, cache)

    start = time.perf_counter()
    for _ in range(num_repeats):
        loss, cache, _ = forward_pass(RNN, X, Y, h0)
        _ = backward_pass(RNN, cache)
    original_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(num_repeats):
        loss, cache, _ = fast_forward_pass(RNN, x_indices, y_indices, h0)
        _ = fast_backward_pass(RNN, cache)
    fast_time = time.perf_counter() - start

    speedup = original_time / fast_time

    print("Speed benchmark")
    print("---------------")
    print(f"Number of repeats: {num_repeats}")
    print(f"Original total time: {original_time:.6f} seconds")
    print(f"Fast total time:     {fast_time:.6f} seconds")
    print(f"Speedup:             {speedup:.2f}x")


def main():
    book_path = "data/goblet_book.txt"

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, _ = build_char_mappings(book_data)

    K = len(unique_chars)
    hidden_size = 100
    seq_length = 25

    RNN = initialize_rnn_parameters(K, m=hidden_size, seed=42)

    X_chars = book_data[:seq_length]
    Y_chars = book_data[1:seq_length + 1]

    X = chars_to_one_hot(X_chars, char_to_ind, K)
    Y = chars_to_one_hot(Y_chars, char_to_ind, K)

    x_indices = chars_to_indices(X_chars, char_to_ind)
    y_indices = chars_to_indices(Y_chars, char_to_ind)

    h0 = np.zeros((hidden_size, 1))

    check_correctness(RNN, X, Y, x_indices, y_indices, h0)
    benchmark(RNN, X, Y, x_indices, y_indices, h0, num_repeats=2000)


if __name__ == "__main__":
    main()
