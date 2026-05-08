import os
import numpy as np

from rnn_bonus import (
    read_book_data,
    build_char_mappings,
    chars_to_one_hot,
    one_hot_to_chars,
    softmax,
    load_rnn_parameters,
)


def softmax_temperature(o, temperature):
    """
    Compute softmax with temperature.

    Lower temperature makes the distribution sharper.
    Higher temperature makes the distribution flatter.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    return softmax(o / temperature)


def sample_from_probabilities(p, rng):
    # Sample one index from a probability vector.
    cp = np.cumsum(p, axis=0)
    a = rng.uniform()
    sampled_index = np.argmax(cp - a > 0)

    return sampled_index


def nucleus_filter(p, theta):
    # Apply nucleus sampling filter to a probability vector.
    if theta <= 0 or theta > 1:
        raise ValueError("theta must be in the interval (0, 1]")

    p_flat = p[:, 0]

    sorted_indices = np.argsort(p_flat)[::-1]
    sorted_probs = p_flat[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    cutoff_position = np.searchsorted(cumulative_probs, theta)
    selected_indices = sorted_indices[:cutoff_position + 1]

    p_filtered = np.zeros_like(p_flat)
    p_filtered[selected_indices] = p_flat[selected_indices]
    p_filtered = p_filtered / np.sum(p_filtered)

    return p_filtered.reshape(-1, 1)


def synthesize_temperature(RNN, h0, x0, n, temperature, rng):
    # Synthesize text using temperature sampling.
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]
    b = RNN["b"]
    c = RNN["c"]

    K = x0.shape[0]
    h = h0.copy()
    x = x0.copy()

    Y = np.zeros((K, n))

    for t in range(n):
        a = W @ h + U @ x + b
        h = np.tanh(a)
        o = V @ h + c

        p = softmax_temperature(o, temperature)
        sampled_index = sample_from_probabilities(p, rng)

        x = np.zeros((K, 1))
        x[sampled_index, 0] = 1
        Y[sampled_index, t] = 1

    return Y


def synthesize_nucleus(RNN, h0, x0, n, theta, rng):
    # Synthesize text using nucleus sampling.
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]
    b = RNN["b"]
    c = RNN["c"]

    K = x0.shape[0]
    h = h0.copy()
    x = x0.copy()

    Y = np.zeros((K, n))

    for t in range(n):
        a = W @ h + U @ x + b
        h = np.tanh(a)
        o = V @ h + c

        p = softmax(o)
        p = nucleus_filter(p, theta)
        sampled_index = sample_from_probabilities(p, rng)

        x = np.zeros((K, 1))
        x[sampled_index, 0] = 1
        Y[sampled_index, t] = 1

    return Y


def synthesize_temperature_nucleus(RNN, h0, x0, n, temperature, theta, rng):
    """
    Synthesize text using combined temperature and nucleus sampling.
    I think it will be interesting to see if it will produce better output.

    The order used here is:
    1. Apply temperature to the logits.
    2. Convert logits to probabilities using softmax.
    3. Apply nucleus filtering to remove low-probability characters.
    4. Sample one character from the filtered probability distribution.
    """
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]
    b = RNN["b"]
    c = RNN["c"]

    K = x0.shape[0]
    h = h0.copy()
    x = x0.copy()

    Y = np.zeros((K, n))

    for t in range(n):
        a = W @ h + U @ x + b
        h = np.tanh(a)
        o = V @ h + c

        # First apply temperature, then apply nucleus filtering.
        p = softmax_temperature(o, temperature)
        p = nucleus_filter(p, theta)

        sampled_index = sample_from_probabilities(p, rng)

        x = np.zeros((K, 1))
        x[sampled_index, 0] = 1
        Y[sampled_index, t] = 1

    return Y


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    os.makedirs("results_bonus", exist_ok=True)

    book_path = "data/goblet_book.txt"
    model_path = "results/best_rnn.npz"

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, ind_to_char = build_char_mappings(book_data)
    K = len(unique_chars)

    RNN = load_rnn_parameters(model_path)

    hidden_size = RNN["W"].shape[0]
    h0 = np.zeros((hidden_size, 1))

    # Use the first character of the book as the initial input.
    x0 = chars_to_one_hot(book_data[0], char_to_ind, K)

    n = 1000

    temperature_values = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
    }

    nucleus_values = {
        "low": 0.5,
        "medium": 0.8,
        "high": 0.95,
    }

    combined_values = {
        "conservative": (0.7, 0.8),
        "balanced": (1.0, 0.9),
        "creative": (1.3, 0.95),
    }

    print("Bonus sampling experiments")
    print("--------------------------")
    print("Generating text from best_rnn.npz")
    print("Text length:", n)
    print()

    for name, temperature in temperature_values.items():
        rng = np.random.default_rng(42)

        Y = synthesize_temperature(
            RNN,
            h0,
            x0,
            n=n,
            temperature=temperature,
            rng=rng,
        )

        text = one_hot_to_chars(Y, ind_to_char)
        output_path = f"results_bonus/temperature_{name}_T{temperature}.txt"
        save_text(output_path, text)

        print(f"Temperature sampling ({name}, T={temperature})")
        print(text[:500])
        print()

    for name, theta in nucleus_values.items():
        rng = np.random.default_rng(42)

        Y = synthesize_nucleus(
            RNN,
            h0,
            x0,
            n=n,
            theta=theta,
            rng=rng,
        )

        text = one_hot_to_chars(Y, ind_to_char)
        output_path = f"results_bonus/nucleus_{name}_theta{theta}.txt"
        save_text(output_path, text)

        print(f"Nucleus sampling ({name}, theta={theta})")
        print(text[:500])
        print()

    for name, (temperature, theta) in combined_values.items():
        rng = np.random.default_rng(42)

        Y = synthesize_temperature_nucleus(
            RNN,
            h0,
            x0,
            n=n,
            temperature=temperature,
            theta=theta,
            rng=rng,
        )

        text = one_hot_to_chars(Y, ind_to_char)
        output_path = (
            f"results_bonus/combined_{name}_T{temperature}_theta{theta}.txt"
        )
        save_text(output_path, text)

        print(
            f"Combined sampling ({name}, "
            f"T={temperature}, theta={theta})"
        )
        print(text[:500])
        print()


if __name__ == "__main__":
    main()