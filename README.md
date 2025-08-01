# Transformer Implementation using PyTorch (Attention is All You Need)

This repository provides a from-scratch implementation of the Transformer model, as described in the `"Attention is All You Need"` paper, using `PyTorch`. It serves as a learning resource for understanding the complexities of the Transformer architecture and its components.


## Transformer Architecture

<img width="556" height="731" alt="image" src="https://github.com/user-attachments/assets/0d434c8d-d2ab-494d-8895-cf504a5a4417" />

## Repository Structure

 In this repo Transformer is constructed in a layered approach, starting from the fundamental building blocks and assembling them into the complete architecture.

```bash

📦 transformer-implementation-with-pytorch
│
├── model/                            # Final transformer model
│   ├── __init__.py                   # Makes 'model' a Python package
│   └── transformer.py                # Finished Transformer architecture
│
├── model_components/                # Core components of the Transformer model
│   ├── blocks/                       # Encoder and Decoder blocks
│   │   ├── __init__.py               # Makes 'blocks' a Python package
│   │   ├── encoder.py                # Encoder block implementation
│   │   └── decoder.py                # Decoder block implementation
│   │
│   ├── embedding/                    # Embedding layers
│   │   ├── __init__.py               # Makes 'embedding' a Python package
│   │   └── input_embedding.py        # Input token + positional embedding
│   │
│   └── layers/                       # Core sub-layers of transformer blocks
│       ├── __init__.py               # Makes 'layers' a Python package
│       ├── feed_forward_nn.py       # Position-wise feed-forward network
│       └── multi_head_attention.py  # Multi-head self-attention mechanism
│
├── model_test.py                    # Script for testing the model
├── requirements.txt                 # Python dependencies for the project
├── README.md                        # Project overview and documentation
├── .gitignore                       # Files/folders to ignore in version control
└── __init__.py                      # Makes the root directory a Python package

``` 






