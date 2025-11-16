# Tiny RAG-LLM: Retrieval-Augmented Language Model

A complete implementation of a tiny Retrieval-Augmented Generation (RAG) language model built from scratch using PyTorch. This project demonstrates how to build an AI system that combines web search, document retrieval, and text generation to answer questions intelligently.

![Project Architecture](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Tiny+RAG-LLM+Architecture)

## 🚀 Features

- **Custom Transformer Architecture**: Built from scratch with multi-head attention, positional encoding, and feed-forward networks
- **Word-Level Tokenization**: Semantic tokenization for better language understanding
- **Web Retrieval**: DuckDuckGo search integration for real-time information gathering
- **Vector Embeddings**: ChromaDB-powered semantic search for document retrieval
- **RAG Pipeline**: Combines retrieval and generation for accurate, context-aware answers
- **Interactive Mode**: Command-line interface for asking questions
- **Training Pipeline**: Complete training loop with optimization and evaluation

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Technical Details](#technical-details)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Web Retrieval  │───▶│  Document       │
│                 │    │  (DuckDuckGo)   │    │  Processing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Embedding      │◀───│  Vector Store   │◀────────────┘
│  Model          │    │  (ChromaDB)     │
│  (SBERT)        │    │                 │
└─────────────────┘    └─────────────────┘
         │                        │
         └────────────────────────┼─────────────────────────┐
                                  ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Context        │───▶│  Prompt         │───▶│  Language       │
│  Augmentation   │    │  Engineering    │    │  Model          │
│                 │    │                 │    │  (Transformer)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Answer         │◀───│  Post-          │◀────────────┘
│  Generation     │    │  Processing     │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

### Model Architecture

```
Transformer Block
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Multi-Head      │    │ Feed-Forward    │                 │
│  │ Self-Attention  │───▶│ Network         │                 │
│  │ (8 heads)       │    │ (d_ff=512)      │                 │
│  └─────────────────┘    └─────────────────┘                 │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│           ┌─────────────────┐                               │
│           │ Layer Norm      │                               │
│           │ + Residual      │                               │
│           └─────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query Processing**: User question is processed and sent to retrieval system
2. **Web Search**: DuckDuckGo searches for relevant information
3. **Document Scraping**: Top results are scraped and cleaned
4. **Embedding**: Documents are converted to vector embeddings
5. **Retrieval**: Semantic search finds most relevant context
6. **Augmentation**: Retrieved context is added to the prompt
7. **Generation**: Transformer model generates answer using context
8. **Post-processing**: Answer is formatted with sources

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Internet connection (for web retrieval)

### Setup

1. **Clone or download the project**
   ```bash
   cd your-project-directory
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- **torch**: Deep learning framework
- **numpy**: Numerical computing
- **tqdm**: Progress bars
- **requests**: HTTP requests for web scraping
- **beautifulsoup4**: HTML parsing
- **duckduckgo-search**: Web search API
- **sentence-transformers**: Text embeddings
- **chromadb**: Vector database

## 🚀 Quick Start

1. **Run the complete system**
   ```bash
   python tiny_rag_llm.py
   ```

2. **The system will:**
   - Load or create training data
   - Build vocabulary and tokenizer
   - Train the language model (5-10 minutes)
   - Initialize web retrieval and embeddings
   - Run example queries
   - Enter interactive mode

3. **Ask questions interactively**
   ```
   ❓ Your question: What is machine learning?
   📋 ANSWER:
   Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...
   ```

## 💡 Usage

### Command Line Interface

```bash
# Run with default settings
python tiny_rag_llm.py

# The system will train automatically and enter interactive mode
```

### Programmatic Usage

```python
from tiny_rag_llm import MiniGPT, SimpleTokenizer, WebRetriever, EmbeddingStore

# Initialize components
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(training_text)

model = MiniGPT(vocab_size=tokenizer.vocab_size)
retriever = WebRetriever()
embedding_store = EmbeddingStore()

# Ask a question
from tiny_rag_llm import generate_with_rag
answer = generate_with_rag(model, tokenizer, "What is photosynthesis?", retriever, embedding_store)
print(answer)
```

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_MODEL` | 256 | Embedding dimension |
| `NUM_LAYERS` | 6 | Number of transformer layers |
| `NUM_HEADS` | 8 | Attention heads per layer |
| `SEQ_LEN` | 256 | Maximum sequence length |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 15 | Training epochs |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.5 | Sampling temperature (0.0-1.0) |
| `top_k` | 40 | Top-K filtering |
| `top_p` | 0.95 | Nucleus sampling |
| `max_new_tokens` | 150 | Maximum generated tokens |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | 3 | Search results to retrieve |
| `n_results` | 2 | Documents to use for context |

## 🎯 Training

### Training Data

The model uses text data for training. By default:

- Looks for `data.txt` in the project directory
- If not found, creates sample training data
- Includes diverse topics: ML, physics, biology, etc.

### Training Process

1. **Tokenization**: Build vocabulary from training text
2. **Dataset Creation**: Convert text to sequences
3. **Model Training**: 15 epochs with optimization
4. **Validation**: Sample generation during training

### Custom Training Data

Create a `data.txt` file with your training text:

```text
Machine learning is a method of data analysis that automates analytical model building.
Deep learning uses neural networks with multiple layers...
```

## 🔧 Technical Details

### Model Architecture

- **Embedding Layer**: Word embeddings with positional encoding
- **Transformer Blocks**: 6 layers with multi-head attention
- **Feed-Forward Networks**: Position-wise with ReLU activation
- **Output Layer**: Linear projection to vocabulary
- **Regularization**: Dropout, layer norm, label smoothing

### Tokenization

- **Word-level**: Preserves semantic meaning
- **Special tokens**: `<PAD>`, `<UNK>`, `<EOS>`, `<BOS>`
- **Punctuation handling**: Maintains sentence structure

### Retrieval System

- **Search Engine**: DuckDuckGo (no API key required)
- **Scraping**: BeautifulSoup for HTML parsing
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB with cosine similarity

### Generation Strategy

- **Top-K Sampling**: Sample from top 40 tokens
- **Nucleus Sampling**: Sample from top 95% probability mass
- **Temperature Control**: 0.5 for coherent output
- **Length Control**: Minimum/maximum token limits

## 📝 Examples

### Example Queries

```
❓ What is photosynthesis?
📋 ANSWER:
Photosynthesis is the process by which plants convert light energy into chemical energy. This occurs in the chloroplasts of plant cells, where light energy is used to drive chemical reactions that produce glucose and oxygen...

📚 Sources:
[1] Photosynthesis - Wikipedia
    https://en.wikipedia.org/wiki/Photosynthesis
```

```
❓ Explain machine learning
📋 ANSWER:
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves...

📚 Sources:
[1] Machine Learning - Wikipedia
    https://en.wikipedia.org/wiki/Machine_learning
```

### Interactive Mode

```bash
python tiny_rag_llm.py

# System trains, then:
❓ Your question: How do transformers work?
📋 ANSWER:
Transformers work by using the attention mechanism to process all tokens in parallel. This allows them to capture long-range dependencies in the data more effectively than previous architectures...

❓ Your question: quit
👋 Thank you for using Tiny RAG-LLM!
```

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements.txt
```

**Slow training**
- Reduce `EPOCHS` to 5-10
- Use GPU if available (modify code to use `cuda`)

**Poor answers**
- Check training data quality
- Increase `EPOCHS` for better learning
- Adjust `temperature` (lower = more coherent)

**Web retrieval fails**
- Check internet connection
- DuckDuckGo may block requests (add delays)
- Use different search results

### Performance Tuning

- **CPU**: ~5-10 minutes training
- **Memory**: ~200-300MB RAM
- **Storage**: ~50MB for embeddings
- **Inference**: ~1-2 seconds per query

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Areas for Improvement

- Add GPU support
- Implement model saving/loading
- Add more sophisticated retrieval
- Improve error handling
- Add unit tests
- Create web interface

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- Based on the Transformer architecture (Vaswani et al.)
- Uses open-source libraries: PyTorch, SentenceTransformers, ChromaDB
- Inspired by modern RAG implementations

---

**Built with ❤️ using PyTorch and modern AI techniques**

For questions or issues, please check the troubleshooting section or create an issue in the repository.