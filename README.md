# Paper Implementations

This repository provides a from-scratch implementation of the Transformer model (as described in the `"Attention is All You Need"` paper) and Vision Transformer model (known from a paper called: `AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE`), using `PyTorch`. It serves as a learning resource for understanding the complexities of the Transformer architecture in a variety of domains, including natural language processing and computer vision, by providing clean, well-documented, and easy-to-follow code implementations.


## Transformer Architecture

<img width="556" height="731" alt="image" src="https://github.com/user-attachments/assets/0d434c8d-d2ab-494d-8895-cf504a5a4417" />

## Vision Transformer Architecture

<img width="908" height="460" alt="image" src="https://github.com/user-attachments/assets/fb251aae-a183-4961-9133-3d0ee60b15a6" />


## Repository Files Structure 

 In this repo Transformer is constructed in a layered approach, starting from the fundamental building blocks and assembling them into the complete architecture.

```bash

📦 transformer-implementation-with-pytorch
│
├── transformer/                      # Contains Transformer model and components
│   ├── __init__.py                  # Makes 'transformer' a Python package
│   │
│   ├── model/                       # Final transformer model
│   │   ├── __init__.py              # Makes 'model' a Python package
│   │   └── transformer.py           # Finished Transformer architecture
│   │
│   └── model_components/            # Core components of the Transformer model
│       ├── blocks/                  # Encoder and Decoder blocks
│       │   ├── __init__.py          # Makes 'blocks' a Python package
│       │   ├── encoder.py           # Encoder block implementation
│       │   └── decoder.py           # Decoder block implementation
│       │
│       ├── embedding/               # Embedding layers
│       │   ├── __init__.py          # Makes 'embedding' a Python package
│       │   └── input_embedding.py   # Input token + positional embedding
│       │
│       └── layers/                  # Core sub-layers of transformer blocks
│           ├── __init__.py          # Makes 'layers' a Python package
│           ├── feed_forward_nn.py   # Position-wise feed-forward network
│           └── multi_head_attention.py  # Multi-head self-attention mechanism
│
├── vision_transformer/              # Vision Transformer implementation
│   ├── __init__.py                  # Makes 'vision_transformer' a Python package
│   ├── implementation.py            # Vision Transformer code
│   └── train_mnist.ipynb            # Training notebook for MNIST
│
├── model_test.py                    # Script for testing the model
├── requirements.txt                 # Python dependencies for the project
├── README.md                       # Project overview and documentation
├── .gitignore                      # Files/folders to ignore in version control
└── __init__.py                     # Makes the root directory a Python package


``` 






