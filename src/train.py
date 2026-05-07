import os
import copy
import numpy as np

from rnn import (
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


def train(
    book_path="data/goblet_book.txt",
    hidden_size=100,
    seq_length=25,
    eta=0.001,
    num_updates=10000,
    print_every=100,
    synth_every=1000,
    synth_length=200,
    seed=42,
):
    os.makedirs("results", exist_ok=True)

    rng = np.random.default_rng(seed)

    book_data = read_book_data(book_path)
    unique_chars, char_to_ind, ind_to_char = build_char_mappings(book_data)
    K = len(unique_chars)

    RNN = initialize_rnn_parameters(K, m=hidden_size, seed=seed)
    adam_state = initialize_adam_state(RNN)

    hprev = np.zeros((hidden_size, 1))
    e = 0

    smooth_loss = -np.log(1.0 / K)
    best_smooth_loss = float("inf")
    best_RNN = None

    loss_history = []

    print("Training vanilla RNN")
    print("--------------------")
    print("Number of characters in book:", len(book_data))
    print("Number of unique characters K:", K)
    print("Hidden size:", hidden_size)
    print("Sequence length:", seq_length)
    print("Learning rate:", eta)
    print("Number of updates:", num_updates)
    print()

    for update_step in range(1, num_updates + 1):
        if e + seq_length + 1 >= len(book_data):
            e = 0
            hprev = np.zeros((hidden_size, 1))

        X_chars = book_data[e:e + seq_length]
        Y_chars = book_data[e + 1:e + seq_length + 1]

        X = chars_to_one_hot(X_chars, char_to_ind, K)
        Y = chars_to_one_hot(Y_chars, char_to_ind, K)

        loss, cache, hprev = forward_pass(RNN, X, Y, hprev)
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
                f"e={e:8d}, "
                f"loss={loss:.4f}, "
                f"smooth_loss={smooth_loss:.4f}"
            )

        if update_step % synth_every == 0:
            x0 = X[:, 0:1]
            Y_synth = synthesize(
                RNN,
                hprev,
                x0,
                n=synth_length,
                rng=rng,
            )
            generated_text = one_hot_to_chars(Y_synth, ind_to_char)

            sample_path = f"results/sample_update_{update_step:06d}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            print()
            print(f"Sample at update {update_step}:")
            print(generated_text)
            print()

        e += seq_length

    np.save("results/loss_history.npy", np.array(loss_history))

    if best_RNN is not None:
        np.savez(
            "results/best_rnn.npz",
            U=best_RNN["U"],
            W=best_RNN["W"],
            V=best_RNN["V"],
            b=best_RNN["b"],
            c=best_RNN["c"],
        )

    print("Training finished.")
    print("Best smooth loss:", best_smooth_loss)


if __name__ == "__main__":
    train()