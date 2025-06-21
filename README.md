# ScratchGPT Implementation by an Undergraduate Student

This repository contains my implementation of the **ScratchGPT** project. 
## About the Project

The goal of this project was to reproduce the GPT-2 (124M parameter) model, learning the architecture and training process by implementing the components myself. This involved building the transformer blocks, attention mechanisms, tokenization, data loading, training loop, and evaluation routines.

By following a detailed repository and accompanying video lectures, I gained hands-on experience in:

- Transformer architecture basics
- Training large language models
- Using PyTorch for efficient model implementation
- Distributed training with DDP for scalability
- Tokenization and batching of large text datasets
- Evaluation on benchmark datasets like HellaSwag

## Features

- From-scratch implementation of GPT-2 architecture
- Training on large-scale datasets with gradient accumulation
- Evaluation on multiple-choice benchmarks
- Distributed Data Parallel (DDP) support for multi-GPU training
- Sample text generation to test model learning

## How to Run

1. **Install dependencies**:

   ```bash
   pip install torch transformers tiktoken tqdm datasets numpy
   ```

2. **Prepare data**:

   Download and tokenize datasets (e.g., fineweb-edu, HellaSwag) as shown in the scripts.

3. **Training**:

   For single-GPU training, simply run:

   ```bash
   python train_scratchgpt.py --model_type gpt2 --device cuda
   ```

   For multi-GPU distributed training using 8 GPUs:

   ```bash
   torchrun --standalone --nproc_per_node=8 train_scratchgpt.py --model_type gpt2 --device cuda
   ```

4. **Evaluation**:

   Evaluate the trained model on the HellaSwag benchmark by running:

   ```bash
   python evaluate_hellaswag.py --model_type gpt2 --device cuda
   ```

5. **Generation**:

   Generate sample text using the trained model to see it "dream" text resembling its training corpus.

## Learning Outcomes

Implementing this project deepened my understanding of:

- Transformer-based language modeling fundamentals
- Efficient PyTorch programming patterns
- Handling large datasets and training at scale
- Challenges and considerations in language model training

This hands-on experience is invaluable for my growth as an AI researcher and practitioner.

---

Feel free to reach out if you want to discuss the implementation details or collaborate!
