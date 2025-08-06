# 📚 Paper Implementations

This repository provides `from-scratch implementations` of core models and architectures introduced in seminal machine learning papers, using `PyTorch`. It serves as both an educational and reference resource for understanding foundational and cutting-edge approaches across a `variety of domains`.

Below is a list of the `research papers` currently implemented in this repository, with proper attribution.

| Paper Title                                                                                                    | Domain          | Citation                 |
| -------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------ |
| [Attention is All You Need](https://arxiv.org/abs/1706.03762)                                                  | NLP             | Vaswani et al., 2017     |
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | Computer Vision | Dosovitskiy et al., 2020 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)              | NLP             | Su et al., 2021          |


## Transformer Architecture (Attention is All You Need)

<img width="556" height="731" alt="image" src="https://github.com/user-attachments/assets/0d434c8d-d2ab-494d-8895-cf504a5a4417" />

## Vision Transformer Architecture (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)

<img width="908" height="460" alt="image" src="https://github.com/user-attachments/assets/fb251aae-a183-4961-9133-3d0ee60b15a6" />

## RoPE (RoFormer: Enhanced Transformer with Rotary Position Embedding)

<img width="1026" height="583" alt="image" src="https://github.com/user-attachments/assets/87ffe32f-e1d4-4fa4-834a-2d260b1d0939" />



## 🗂️ Repository Files Structure 

The implementations in this repository are built using a layered and modular approach, starting from fundamental components and gradually assembling them into `complete, functional architectures`. This design promotes reusability, understanding, and easy experimentation with different model variants.

```bash

transformer-implementation-with-pytorch
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
├── roformer/                         # RoFormer implementation
│   ├── __init__.py                  
│   ├── rope.py                       # Rotary Positional Encoding
│   └── transformer_with_rope.py      # Transformer model with RoPE
├── model_test.py                    # Script for testing the model
├── requirements.txt                 # Python dependencies for the project
├── README.md                       # Project overview and documentation
├── .gitignore                      # Files/folders to ignore in version control
└── __init__.py                     # Makes the root directory a Python package


``` 






