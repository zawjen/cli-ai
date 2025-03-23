# Introduction

A CLI for Zawjen AI based Q&A. Retrieval-Augmented Generation (RAG) System.

## Setup

Install the following dependencies:

```sh
pip install sentence-transformers faiss-cpu numpy transformers argparse torch
```

1. `sentence-transformers`:  Generates embeddings for document chunks.  
2. `faiss-cpu`:  Efficient similarity search for retrieval.  
3. `numpy`:  Handles numerical operations.  
4. `transformers`:  Loads and runs the Flan-T5 model.  
5. `argparse`:  Parses command-line arguments (`--qid`).  
6. `torch`:  Required by transformers for deep learning model execution.  

## Run

Use following command to run application:

```sh
python main.py --qid Q1
```

Or

```sh
py main.py --qid Q1
```

## QA Model

Model `free` alternatives depending on **hardware, accuracy needs, and speed requirements**. Since we have **16GB RAM and an Intel Core i7**, you can run **mid-sized models efficiently**, though GPU-based models may still be slow.  

`Flan-T5 (Large or XL)` is good for `Query Answer` use case as we have.

| **Model**                   | **Size**  | **Pros**                            | **Cons**                                | **Ideal For**                   |
| --------------------------- | --------- | ----------------------------------- | --------------------------------------- | ------------------------------- |
| **Flan-T5 (Base/Large/XL)** | 250M - 3B | Lightweight, free, good for Q&A     | Less powerful than GPT-4                | General NLP, Query Answering |
| **Mistral-7B**              | 7B        | High accuracy, better than LLaMA-7B | Needs 8GB+ VRAM for good speed          | Chatbots, Reasoning             |
| **LLaMA 2-7B/13B**          | 7B / 13B  | Strong general performance          | Slower on CPU                           | Research, Chatbots              |
| **Falcon-7B/40B**           | 7B / 40B  | Open-source, good for text tasks    | Slow on CPU, 40B too large for 16GB RAM | Content Generation              |
| **Gemma-7B**                | 7B        | Optimized for instruction-following | Requires GPU for fast results           | AI Assistants                   |
| **GPT-J-6B**                | 6B        | Reasonably fast, free               | Lower accuracy vs. newer models         | Creative Writing                |
| **GPT-NeoX-20B**            | 20B       | Powerful for reasoning              | Too large for 16GB RAM alone            | Advanced Research               |
| **Phi-2**                   | 2.7B      | Optimized, small, powerful          | Not as tested as Mistral                | Small-scale AI Apps             |

### Best Choice for QA, Intel i7 + 16GB RAM Setup

**Fastest (CPU-Friendly) Choice:**  
- **Flan-T5 (Large)** - Works well on CPU, good accuracy.  

**More Powerful But Heavier:**  
- **Mistral-7B** - Better than LLaMA-7B, but slower on CPU.  
- **LLaMA 2-7B** - Great, but needs more resources.  

**Best for Instruction-Following (Chatbots):**  
- **Gemma-7B** - Google's latest model.  
- **Phi-2** - Small but powerful.  

## Embeddings Model

The model **`sentence-transformers/all-MiniLM-L6-v2`** is used for generating **embeddings** in project. If you want alternatives, you can choose based on **speed, accuracy, and size**.  

## Best Alternatives for `all-MiniLM-L6-v2`
| **Model**                              | **Size** | **Speed** | **Accuracy**                    | **Best For**          |
| -------------------------------------- | -------- | --------- | ------------------------------- | --------------------- |
| `all-MiniLM-L6-v2` *(Current Model)*   | 22M      | Fast      | Good                            | General-purpose       |
| **Smaller (Faster but Less Accurate)** |
| `all-MiniLM-L12-v2`                    | 33M      | Fast      | Slightly better accuracy        | Speed-focused apps    |
| `paraphrase-MiniLM-L6-v2`              | 22M      | Fast      | Optimized for similar sentences | Paraphrasing tasks    |
| `all-distilroberta-v1`                 | 82M      | Medium    | More accurate                   | General-purpose       |
| **Larger (More Accurate but Slower)**  |
| `multi-qa-mpnet-base-dot-v1`           | 110M     | Slower    | Higher accuracy                 | QA & retrieval        |
| `all-mpnet-base-v2`                    | 110M     | Slower    | More accurate than MiniLM       | Search engines        |
| `bge-large-en-v1.5`                    | 300M     | Slow      | Best accuracy                   | Search, QA            |
| `e5-large-v2`                          | 350M     | Slow      | Very high accuracy              | Large-scale retrieval |

### Best Choice for Your System (16GB RAM, Intel i7)
**If You Want Speed & Low Memory Usage**  
- **`all-MiniLM-L12-v2`** (Slightly better than current MiniLM model)  

**If You Want Higher Accuracy Without Too Much Slowdown**  
- **`all-mpnet-base-v2`** (More accurate than MiniLM, but slower)  

**If You Want the Best Accuracy & Can Handle More RAM Usage**  
- **`bge-large-en-v1.5`** or **`e5-large-v2`** (Best for retrieval tasks)  
