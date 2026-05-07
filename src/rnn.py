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