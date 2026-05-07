import numpy as np


def read_book_data(book_path):
    # Read the full training text from a text file.
    # Return a long string.
    with open(book_path, "r", encoding="utf-8") as f:
        book_data = f.read()

    return book_data


def build_char_mappings(book_data):
    # Build character-to-index and index-to-character mappings.
    unique_chars = sorted(list(set(book_data)))

    char_to_ind = {}
    ind_to_char = {}

    for i, ch in enumerate(unique_chars):
        char_to_ind[ch] = i
        ind_to_char[i] = ch

    return unique_chars, char_to_ind, ind_to_char


def chars_to_one_hot(chars, char_to_ind, K):
    # Convert a string of characters into a one-hot matrix.
    seq_length = len(chars)
    X = np.zeros((K, seq_length))

    for t, ch in enumerate(chars):
        index = char_to_ind[ch]
        X[index, t] = 1

    return X


def one_hot_to_chars(X, ind_to_char):
    # Convert a one-hot matrix back to a string.
    chars = []

    for t in range(X.shape[1]):
        index = np.argmax(X[:, t])
        chars.append(ind_to_char[index])

    return "".join(chars)


def initialize_rnn_parameters(K, m=100, seed=42):
    # Initialize the parameters of a vanilla RNN.
    # Return a dictionary containing RNN parameters U, W, V, b, c.
    rng = np.random.default_rng(seed)

    RNN = {}

    RNN["b"] = np.zeros((m, 1))
    RNN["c"] = np.zeros((K, 1))

    RNN["U"] = (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(m, K))
    RNN["W"] = (1 / np.sqrt(2 * m)) * rng.standard_normal(size=(m, m))
    RNN["V"] = (1 / np.sqrt(m)) * rng.standard_normal(size=(K, m))

    return RNN


def softmax(o):
    exp_o = np.exp(o - np.max(o))
    p = exp_o / np.sum(exp_o)

    return p


def forward_pass(RNN, X, Y, h0):
    # Run the forward pass of the vanilla RNN.
    
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]
    b = RNN["b"]
    c = RNN["c"]

    seq_length = X.shape[1]

    a_list = []
    h_list = [h0]
    o_list = []
    p_list = []

    loss = 0.0

    for t in range(seq_length):
        x_t = X[:, t:t + 1]
        y_t = Y[:, t:t + 1]

        a_t = W @ h_list[-1] + U @ x_t + b
        h_t = np.tanh(a_t)
        o_t = V @ h_t + c
        p_t = softmax(o_t)

        loss += -np.log(np.sum(y_t * p_t) + 1e-12)

        a_list.append(a_t)
        h_list.append(h_t)
        o_list.append(o_t)
        p_list.append(p_t)

    loss = loss / seq_length

    cache = {
        "X": X,
        "Y": Y,
        "a": a_list,
        "h": h_list,
        "o": o_list,
        "p": p_list,
        "h0": h0,
    }

    h_last = h_list[-1]

    return loss, cache, h_last


def synthesize(RNN, h0, x0, n, rng=None):
    # Synthesize a sequence of characters from the RNN.
    if rng is None:
        rng = np.random.default_rng()

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

        cp = np.cumsum(p, axis=0)
        a_random = rng.uniform()
        sampled_index = np.argmax(cp - a_random > 0)

        x = np.zeros((K, 1))
        x[sampled_index, 0] = 1

        Y[sampled_index, t] = 1

    return Y


def backward_pass(RNN, cache):
    # Run the backward pass of the vanilla RNN using BPTT.
    U = RNN["U"]
    W = RNN["W"]
    V = RNN["V"]

    X = cache["X"]
    Y = cache["Y"]
    h = cache["h"]
    p = cache["p"]

    seq_length = X.shape[1]

    grads = {}
    grads["U"] = np.zeros_like(RNN["U"])
    grads["W"] = np.zeros_like(RNN["W"])
    grads["V"] = np.zeros_like(RNN["V"])
    grads["b"] = np.zeros_like(RNN["b"])
    grads["c"] = np.zeros_like(RNN["c"])

    # Gradient flowing from the future hidden state.
    dh_next = np.zeros_like(h[0])

    for t in reversed(range(seq_length)):
        x_t = X[:, t:t + 1]
        y_t = Y[:, t:t + 1]
        h_t = h[t + 1]
        h_prev = h[t]

        # Since loss is averaged over seq_length in forward_pass,
        # the output gradient should also be divided by seq_length.
        do = (p[t] - y_t) / seq_length

        grads["V"] += do @ h_t.T
        grads["c"] += do

        dh = V.T @ do + dh_next

        # h_t = tanh(a_t), so derivative is 1 - h_t^2.
        da = dh * (1 - h_t ** 2)

        grads["W"] += da @ h_prev.T
        grads["U"] += da @ x_t.T
        grads["b"] += da

        dh_next = W.T @ da

    # Gradient clipping to avoid exploding gradients.
    for key in grads:
        grads[key] = np.clip(grads[key], -5, 5)

    return grads


if __name__ == "__main__":
    book_path = "data/goblet_book.txt"

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, ind_to_char = build_char_mappings(book_data)

    K = len(unique_chars)

    print("Total number of characters in the book:", len(book_data))
    print("Number of unique characters K:", K)
    print("First 100 characters:")
    print(book_data[:100])

    test_chars = book_data[:25]
    X = chars_to_one_hot(test_chars, char_to_ind, K)
    recovered_chars = one_hot_to_chars(X, ind_to_char)

    print("Test sequence:")
    print(test_chars)
    print("One-hot shape:", X.shape)
    print("Recovered sequence:")
    print(recovered_chars)

    assert test_chars == recovered_chars
    print("Data preprocessing test passed.")

    RNN = initialize_rnn_parameters(K, m=100, seed=42)

    print("\nRNN parameter shapes:")
    for key in ["U", "W", "V", "b", "c"]:
        print(key, RNN[key].shape)

    assert RNN["U"].shape == (100, K)
    assert RNN["W"].shape == (100, 100)
    assert RNN["V"].shape == (K, 100)
    assert RNN["b"].shape == (100, 1)
    assert RNN["c"].shape == (K, 1)

    print("RNN parameter initialization test passed.")

    X_chars = book_data[:25]
    Y_chars = book_data[1:26]

    X = chars_to_one_hot(X_chars, char_to_ind, K)
    Y = chars_to_one_hot(Y_chars, char_to_ind, K)

    h0 = np.zeros((100, 1))

    loss, cache, h_last = forward_pass(RNN, X, Y, h0)

    print("\nForward pass test:")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Initial hidden state shape:", h0.shape)
    print("Final hidden state shape:", h_last.shape)
    print("Loss:", loss)

    assert h_last.shape == (100, 1)
    assert len(cache["a"]) == 25
    assert len(cache["h"]) == 26
    assert len(cache["o"]) == 25
    assert len(cache["p"]) == 25

    print("Forward pass test passed.")

    rng = np.random.default_rng(42)
    x0 = X[:, 0:1]
    Y_synth = synthesize(RNN, h0, x0, n=200, rng=rng)
    generated_text = one_hot_to_chars(Y_synth, ind_to_char)

    print("\nSynthesized text from randomly initialized RNN:")
    print(generated_text)

    assert Y_synth.shape == (K, 200)
    print("Text synthesis test passed.")

    grads = backward_pass(RNN, cache)

    print("\nBackward pass test:")
    for key in ["U", "W", "V", "b", "c"]:
        print(key, grads[key].shape, "max abs grad:", np.max(np.abs(grads[key])))

    assert grads["U"].shape == RNN["U"].shape
    assert grads["W"].shape == RNN["W"].shape
    assert grads["V"].shape == RNN["V"].shape
    assert grads["b"].shape == RNN["b"].shape
    assert grads["c"].shape == RNN["c"].shape

    print("Backward pass shape test passed.")