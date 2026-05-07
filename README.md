# DD2424 Assignment 4: Character-level RNN

This repository contains my implementation for DD2424 Assignment 4.

The goal of the assignment is to train a vanilla recurrent neural network to synthesize English text character by character using the text from *Harry Potter and the Goblet of Fire*.

## Project structure

```text
DD2424_Assignment4/
├── data/          # Training text data
├── src/           # Source code
├── results/       # Training curves, logs, and generated samples
├── report/        # Final report
├── README.md
└── .gitignore
```

Main components

* Data preprocessing and character-to-index mappings
* Vanilla RNN forward pass
* Back-propagation through time
* Adam optimizer
* Text synthesis
* Gradient checking
* Bonus experiments

