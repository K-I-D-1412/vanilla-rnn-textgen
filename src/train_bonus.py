import os
import copy
import numpy as np

from rnn_bonus import (
    read_book_data,
    build_char_mappings,
    chars_to_one_hot,
    one_hot_to_chars,
    initialize_rnn_parameters,
    forward_pass,
    backward_pass,
    synthesize,
    initialize_adam_state,
    adam_update,
)


def train_random_sequence_bonus(
    book_path="data/goblet_book.txt",
    hidden_size=100,
    seq_length=25,
    eta=0.001,
    num_updates=100000,
    print_every=100,
    synth_every=10000,
    synth_length=200,
    final_synth_length=1000,
    seed=42,
):
    """
    Bonus experiment: train the RNN by randomly sampling training sequences.

    Unlike the main assignment training loop, which scans through the book
    sequentially, this bonus version randomly samples a new text position at
    every update step. Since consecutive sampled sequences are not necessarily
    adjacent in the book, the hidden state is reset to zero for every update.
    """
    os.makedirs("results_bonus", exist_ok=True)

    rng = np.random.default_rng(seed)

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, ind_to_char = build_char_mappings(book_data)
    K = len(unique_chars)

    RNN = initialize_rnn_parameters(K, m=hidden_size, seed=seed)
    adam_state = initialize_adam_state(RNN)

    smooth_loss = -np.log(1.0 / K)
    best_smooth_loss = float("inf")
    best_RNN = None

    loss_history = []

    print("Bonus training: random sequence order")
    print("-------------------------------------")
    print("Number of characters in book:", len(book_data))
    print("Number of unique characters K:", K)
    print("Hidden size:", hidden_size)
    print("Sequence length:", seq_length)
    print("Learning rate:", eta)
    print("Number of updates:", num_updates)
    print()

    # Sample before training.
    h0 = np.zeros((hidden_size, 1))
    x0 = chars_to_one_hot(book_data[0], char_to_ind, K)

    Y_synth_0 = synthesize(
        RNN,
        h0,
        x0,
        n=synth_length,
        rng=rng,
    )
    generated_text_0 = one_hot_to_chars(Y_synth_0, ind_to_char)

    with open("results_bonus/random_sample_update_000000.txt", "w", encoding="utf-8") as f:
        f.write(generated_text_0)

    print("Sample before random-sequence training:")
    print(generated_text_0)
    print()

    max_start_index = len(book_data) - seq_length - 1

    for update_step in range(1, num_updates + 1):
        # Randomly sample a sequence start position.
        e = rng.integers(0, max_start_index)

        X_chars = book_data[e:e + seq_length]
        Y_chars = book_data[e + 1:e + seq_length + 1]

        X = chars_to_one_hot(X_chars, char_to_ind, K)
        Y = chars_to_one_hot(Y_chars, char_to_ind, K)

        # Since sequences are sampled randomly, reset hidden state each update.
        hprev = np.zeros((hidden_size, 1))

        loss, cache, _ = forward_pass(RNN, X, Y, hprev)
        grads = backward_pass(RNN, cache)
        RNN, adam_state = adam_update(RNN, grads, adam_state, eta=eta)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        loss_history.append(smooth_loss)

        if smooth_loss < best_smooth_loss:
            best_smooth_loss = smooth_loss
            best_RNN = copy.deepcopy(RNN)

        if update_step % print_every == 0:
            print(
                f"update={update_step:6d}, "
                f"sampled_e={e:8d}, "
                f"loss={loss:.4f}, "
                f"smooth_loss={smooth_loss:.4f}"
            )

        if update_step % synth_every == 0:
            h_synth = np.zeros((hidden_size, 1))
            x_synth = X[:, 0:1]

            Y_synth = synthesize(
                RNN,
                h_synth,
                x_synth,
                n=synth_length,
                rng=rng,
            )
            generated_text = one_hot_to_chars(Y_synth, ind_to_char)

            sample_path = f"results_bonus/random_sample_update_{update_step:06d}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            print()
            print(f"Random-sequence sample at update {update_step}:")
            print(generated_text)
            print()

    np.save("results_bonus/random_loss_history.npy", np.array(loss_history))

    if best_RNN is not None:
        np.savez(
            "results_bonus/random_best_rnn.npz",
            U=best_RNN["U"],
            W=best_RNN["W"],
            V=best_RNN["V"],
            b=best_RNN["b"],
            c=best_RNN["c"],
        )

        h0_best = np.zeros((hidden_size, 1))
        x0_best = chars_to_one_hot(book_data[0], char_to_ind, K)

        Y_best = synthesize(
            best_RNN,
            h0_best,
            x0_best,
            n=final_synth_length,
            rng=rng,
        )
        best_text = one_hot_to_chars(Y_best, ind_to_char)

        with open("results_bonus/random_best_model_sample_1000.txt", "w", encoding="utf-8") as f:
            f.write(best_text)

    print("Random-sequence bonus training finished.")
    print("Best random-sequence smooth loss:", best_smooth_loss)


if __name__ == "__main__":
    train_random_sequence_bonus()