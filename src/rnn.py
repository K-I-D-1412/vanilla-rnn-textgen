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