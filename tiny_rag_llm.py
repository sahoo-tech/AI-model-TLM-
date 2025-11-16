import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')




class SimpleTokenizer:
 
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        self.pad_idx = None
        self.unk_idx = None
        self.eos_idx = None
        self.bos_idx = None

    def build_vocab(self, text):
        """Build vocabulary from training text (word-level)."""
        # Tokenize into words
        import re
        words = re.findall(r'\b\w+\b|[.!?,;]', text.lower())
        unique_words = sorted(set(words))

        # Add special tokens first for consistent indices
        special_tokens = [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        words_list = special_tokens + unique_words

        # Build mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(words_list)}
        self.idx_to_word = {idx: word for idx, word in enumerate(words_list)}
        self.vocab_size = len(words_list)

        # Store special token indices for quick access
        self.pad_idx = self.word_to_idx[self.pad_token]
        self.unk_idx = self.word_to_idx[self.unk_token]
        self.eos_idx = self.word_to_idx[self.eos_token]
        self.bos_idx = self.word_to_idx[self.bos_token]

        print(f"Vocabulary size: {self.vocab_size} words")

    def encode(self, text, max_len=None):
        """Convert text to token indices (word-level)."""
        import re
        words = re.findall(r'\b\w+\b|[.!?,;]', text.lower())
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]

        if max_len:
            if len(indices) < max_len:
                indices = indices + [self.pad_idx] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

        return indices

    def decode(self, indices, remove_special=True):
        """Convert token indices back to text (word-level)."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist() if indices.dim() > 0 else [indices.item()]

        words = []
        for idx in indices:
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if len(idx) > 0 else self.pad_idx
            word = self.idx_to_word.get(idx, self.unk_token)
            words.append(word)

        # Join words with spaces, but keep punctuation attached
        text = []
        for word in words:
            if word in ['.', ',', '!', '?', ';', ':']:
                if text:
                    text[-1] += word
            else:
                text.append(word)

        result = ' '.join(text)

        if remove_special:
            # Remove special tokens
            result = result.replace(self.pad_token, '')
            result = result.replace(self.eos_token, '')
            result = result.replace(self.unk_token, '')
            result = result.replace(self.bos_token, '')

        return result.strip()




class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class MiniGPT(nn.Module):
    """Improved Mini GPT-style transformer language model."""
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, 
                 d_ff=512, max_seq_len=256, dropout=0.15):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # Output normalization

        self._init_parameters()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ Model initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.layer_norm(x)  # Final layer normalization
        logits = self.output(x)
        return logits

    def generate(self, prompt_indices, max_new_tokens=50, temperature=0.7, 
                 tokenizer=None, min_tokens=5, top_k=40, top_p=0.95):
        """
        Generate text using improved sampling with top-k and top-p filtering.
        UPGRADED: Better quality generation with coherence control
        """
        self.eval()

        with torch.no_grad():
            current_indices = prompt_indices.clone()
            generated_tokens = []

            for step in range(max_new_tokens):
                # Truncate if too long
                if current_indices.size(1) > self.max_seq_len:
                    current_indices = current_indices[:, -self.max_seq_len:]

                # Get predictions
                logits = self.forward(current_indices)
                next_token_logits = logits[:, -1, :] / max(temperature, 0.1)

                # Numerical stability
                next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]

                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Calculate probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[0, indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # Handle NaN/Inf
                if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

                generated_tokens.append(next_token.item())
                current_indices = torch.cat([current_indices, next_token], dim=1)

                # Stop on EOS after minimum tokens
                if tokenizer and step >= min_tokens:
                    if next_token.item() == tokenizer.eos_idx:
                        break

            # Return generated text
            full_output = current_indices[0].tolist()

            if tokenizer:
                prompt_len = prompt_indices.size(1)
                generated_part = full_output[prompt_len:]

                if len(generated_part) > 0:
                    generated_text = tokenizer.decode(generated_part)
                else:
                    generated_text = "I don't have enough information to answer."

                return generated_text.strip() if generated_text.strip() else "I don't have enough information to answer."
            else:
                return full_output



class TextDataset(Dataset):
    """Simple dataset for character-level language modeling."""
    def __init__(self, text, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = tokenizer.encode(text)

        # FIXED: Ensure we have enough data
        if len(self.data) < seq_len + 1:
            raise ValueError(f"Training text too short. Need at least {seq_len + 1} tokens, got {len(self.data)}")

    def __len__(self):
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def create_causal_mask(seq_len):
    """Create causal mask for autoregressive attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask


def train_model(model, train_loader, tokenizer, epochs=15, lr=0.0005, weight_decay=1e-5):
   
    print("\n" + "="*70)
    print("🚀 TRAINING TRANSFORMER (IMPROVED)")
    print("="*70)

    device = torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}")

        try:
            for batch_idx, (x, y) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)

                # Create causal mask
                mask = create_causal_mask(x.size(1)).to(device)

                # Forward pass
                logits = model(x, mask)

                # Compute loss
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n⚠ NaN loss at batch {batch_idx}. Skipping...")
                    continue

                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            scheduler.step()
            
            print(f"   Average Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Sample text generation every 3 epochs
            if (epoch + 1) % 3 == 0:
                try:
                    sample_prompts = ["what is", "machine learning", "the"]
                    for prompt in sample_prompts[:1]:
                        prompt_indices = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                        sample_text = model.generate(prompt_indices, max_new_tokens=40, 
                                                    temperature=0.6, tokenizer=tokenizer, 
                                                    min_tokens=3, top_k=30, top_p=0.9)
                        print(f"   Sample: '{prompt}' → {sample_text[:60]}...")
                except Exception as e:
                    print(f"Error in sample generation: {e}")

    print("\n✓ Training complete!")
    return model




class WebRetriever:
    """Web retrieval module with better error handling."""
    def __init__(self, max_results=3):
        self.max_results = max_results

    def search(self, query):
        """Search DuckDuckGo and return top results."""
        print(f"\n🔍 Searching DuckDuckGo for: '{query}'")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            print(f"Found {len(results)} results")
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def scrape_page(self, url, timeout=5):
        """Scrape a webpage and extract readable text."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # FIXED: Ensure we return something meaningful
            text = text[:2000] if text else f"Unable to extract text from {url}"
            return text

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return f"Scraping failed: {str(e)[:100]}"

    def retrieve(self, query):
        """Complete retrieval pipeline."""
        search_results = self.search(query)

        if not search_results:
            return []

        documents = []
        print("\n📄 Scraping pages...")

        for result in search_results:
            url = result.get('href', result.get('link', ''))
            title = result.get('title', 'Untitled')

            if not url:
                continue

            print(f"  - {title[:50]}...")
            text = self.scrape_page(url)

            if text:
                documents.append({
                    'text': text,
                    'url': url,
                    'title': title
                })

        print(f"\n✓ Retrieved {len(documents)} documents")
        return documents




class EmbeddingStore:

    def __init__(self, collection_name="rag_docs"):
        print("\n🧠 Initializing embedding model (all-MiniLM-L6-v2)...")

        try:
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.client = chromadb.Client()

            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except:
                self.collection = self.client.get_collection(name=collection_name)

            print("✓ Embedding store ready")
        except Exception as e:
            print(f"Error initializing embedding store: {e}")
            raise

    def add_documents(self, documents):
        """Embed and store documents in ChromaDB."""
        if not documents:
            return

        print(f"\n💾 Storing {len(documents)} documents...")

        try:
            texts = [doc['text'] for doc in documents]
            embeddings = self.embed_model.encode(texts, show_progress_bar=True)

            ids = [f"doc_{i}" for i in range(len(documents))]
            metadatas = [{'url': doc['url'], 'title': doc['title']} 
                        for doc in documents]

            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )

            print("✓ Documents stored")
        except Exception as e:
            print(f"Error storing documents: {e}")

    def retrieve_relevant(self, query, n_results=2):
        """Retrieve most relevant documents for query."""
        print(f"\n🎯 Retrieving relevant context for: '{query}'")

        try:
            query_embedding = self.embed_model.encode([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )

            retrieved = []
            for i in range(len(results['documents'][0])):
                retrieved.append({
                    'text': results['documents'][0][i],
                    'url': results['metadatas'][0][i]['url'],
                    'title': results['metadatas'][0][i]['title'],
                    'score': 1 - results['distances'][0][i]
                })

            print(f"✓ Retrieved {len(retrieved)} relevant snippets")
            return retrieved
        except Exception as e:
            print(f"Error retrieving: {e}")
            return []




def augment_prompt_with_context(query, retrieved_docs):
    """Build augmented prompt with retrieved context."""
    if not retrieved_docs:
        # Fallback context if nothing retrieved
        prompt = f"Answer the following question based on your knowledge:\n\nQuestion: {query}\nAnswer:"
        return prompt

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['title']}\n{doc['text'][:300]}...\nSource: {doc['url']}")

    context = "\n\n".join(context_parts)

    augmented_prompt = f"""Based on the following information, answer the question.
If the answer is not in the provided context, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

    return augmented_prompt


def generate_with_rag(model, tokenizer, query, retriever, embedding_store, 
                     temperature=0.5, max_new_tokens=150):
   
    print("\n" + "="*70)
    print("RAG PIPELINE: RETRIEVAL-AUGMENTED GENERATION")
    print("="*70)

    try:
        documents = retriever.retrieve(query)

        if not documents:
            print("\n⚠ No documents retrieved. Using model knowledge only.")
            prompt = f"Answer the question: {query}\nAnswer: "
            sources = []
        else:
            embedding_store.add_documents(documents)
            retrieved = embedding_store.retrieve_relevant(query, n_results=2)
            prompt = augment_prompt_with_context(query, retrieved)
            sources = retrieved

        print(f"\n📝 Prompt (first 150 chars):\n{prompt[:150]}...\n")

        # Encode and generate with improved parameters
        print("🤖 Generating answer...")
        prompt_indices = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        generated_text = model.generate(
            prompt_indices, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,  # Reduced from 0.7 to 0.5 for more coherence
            tokenizer=tokenizer,
            min_tokens=5,
            top_k=40,
            top_p=0.9
        )

        # Format output
        result = f"{generated_text}\n"

        if sources:
            result += "\n📚 Sources:\n"
            for i, source in enumerate(sources, 1):
                result += f"[{i}] {source['title']}\n    {source['url']}\n"

        return result

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Error generating answer: {str(e)}"




def create_sample_training_data():
    """Create comprehensive training data for better model learning."""
    sample_text = """
What is machine learning? Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Natural language processing is a field of artificial intelligence that focuses on the interactions between computers and human language. It enables computers to understand, interpret, and generate human language in a meaningful way. NLP is used in many applications like machine translation, sentiment analysis, and chatbots.

Deep learning uses neural networks with multiple layers to process data. These layers are organized hierarchically, allowing the network to learn increasingly abstract representations of the input data. Deep learning has revolutionized many fields including computer vision and natural language processing.

Python is a popular programming language for artificial intelligence and data science applications. It has many libraries like TensorFlow, PyTorch, and scikit-learn that make building machine learning models easier. Python is known for its simplicity and readability.

Transformers are a revolutionary neural network architecture introduced in 2017. They use the attention mechanism to process sequences of data in parallel, making them much faster than previous approaches. Transformers have become the foundation for many state-of-the-art language models.

The attention mechanism allows models to focus on the most relevant parts of the input. This is achieved through a weighted sum of the input values, where the weights are learned during training. Attention mechanisms are crucial for the success of transformer models.

Language models predict the next word in a sequence based on the previous words. They are trained on large amounts of text data to learn the patterns and structure of language. Language models can be used for text generation, machine translation, and other NLP tasks.

Physics is the study of matter, energy, and the fundamental forces of nature. It seeks to understand how the universe works at all scales, from subatomic particles to galaxy clusters. Physics is divided into several branches including mechanics, thermodynamics, electromagnetism, and quantum mechanics.

Photosynthesis is the process by which plants convert light energy into chemical energy. This occurs in the chloroplasts of plant cells, where light energy is used to drive chemical reactions that produce glucose and oxygen. Photosynthesis is essential for life on Earth.

Cells are the basic units of life. They are the smallest living units that can carry out all life functions independently. There are two main types of cells: prokaryotic cells like bacteria and archaea, and eukaryotic cells like those in animals and plants.

Evolution is the process by which populations of organisms change over time. It explains the diversity of life on Earth and the similarities between different species. Evolution occurs through natural selection, where organisms with advantageous traits are more likely to survive and reproduce.

Gravity is a fundamental force in the universe that attracts objects to each other. It is responsible for holding planets in orbit around stars and for the structure of galaxies. Einstein described gravity as a curvature of spacetime caused by mass and energy.

Atoms are the basic building blocks of matter. They consist of a nucleus containing protons and neutrons, surrounded by a cloud of electrons. The properties of atoms determine the properties of the elements and compounds they form.

Molecules are formed when atoms bond together through chemical bonds. These bonds are formed by the sharing or transfer of electrons between atoms. The properties of molecules depend on the atoms they contain and how they are arranged.

Energy is the capacity to do work or cause change. It can take many forms including kinetic energy, potential energy, thermal energy, and electrical energy. Energy is conserved in closed systems, but it can be transformed from one form to another.

What are neural networks? Neural networks are computational models inspired by the biological neural networks in animal brains. They consist of interconnected nodes that process information. Neural networks are the foundation of deep learning and have many applications in AI.

How do transformers work? Transformers work by using the attention mechanism to process all tokens in parallel. This allows them to capture long-range dependencies in the data more effectively than previous architectures like recurrent neural networks. The transformer architecture has become the standard for NLP tasks.

What is supervised learning? Supervised learning is a machine learning approach where the model learns from labeled data. The model is given input data along with the correct output labels, and it learns to map inputs to outputs. Supervised learning is used for tasks like classification and regression.

What is unsupervised learning? Unsupervised learning is a machine learning approach where the model learns from unlabeled data. The model tries to find patterns and structure in the data without being given explicit labels. Unsupervised learning is used for tasks like clustering and dimensionality reduction.

What is reinforcement learning? Reinforcement learning is a machine learning approach where an agent learns by interacting with an environment. The agent takes actions and receives rewards or penalties based on those actions. The goal is to learn a policy that maximizes cumulative rewards over time.
""" * 3  # Repeat for more training data
    return sample_text


def main():
    """Main execution with comprehensive error handling."""
    print("\n" + "="*70)
    print("TINY RAG-LLM: Retrieval-Augmented Language Model")
    print("Build a mini GPT + Web RAG system from scratch")
    print("="*70)

    try:
        # IMPROVED CONFIGURATION
        SEQ_LEN = 256          # Longer sequences for better context
        BATCH_SIZE = 16        # Smaller batches for stability
        EPOCHS = 15            # More epochs for better learning (increased from 3)
        D_MODEL = 256          # Larger model capacity (increased from 128)
        NUM_LAYERS = 6         # More transformer layers (increased from 4)
        NUM_HEADS = 8          # More attention heads (increased from 4)

        # Step 1: Load training data
        print("\n📚 Loading training data...")
        try:
            with open('data.txt', 'r', encoding='utf-8') as f:
                training_text = f.read()
            print(f"Loaded {len(training_text)} characters from data.txt")
        except FileNotFoundError:
            print("data.txt not found, using sample data")
            training_text = create_sample_training_data()
            with open('data.txt', 'w', encoding='utf-8') as f:
                f.write(training_text)
            print(f"Created sample data.txt with {len(training_text)} characters")

        if len(training_text) < 500:
            print("WARNING: Training text is very short. Model may not learn well.")

        # Step 2: Build tokenizer
        print("\n🔤 Building tokenizer...")
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(training_text)

        # Step 3: Create dataset
        try:
            dataset = TextDataset(training_text, tokenizer, seq_len=SEQ_LEN)
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            print(f"Dataset size: {len(dataset)} sequences")
        except ValueError as e:
            print(f"Error creating dataset: {e}")
            return

        # Step 4: Create model
        print("\n🏗️ Building Mini-GPT model...")
        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_ff=D_MODEL*2,
            max_seq_len=SEQ_LEN
        )

        # Step 5: Train model
        model = train_model(model, train_loader, tokenizer, epochs=EPOCHS)

        # Step 6: Initialize retrieval
        print("\n🌐 Initializing web retrieval...")
        retriever = WebRetriever(max_results=3)
        embedding_store = EmbeddingStore()

        # Step 7: Run example queries
        print("\n" + "="*70)
        print("RAG SYSTEM READY!")
        print("="*70)

        example_queries = [
            "What is photosynthesis?",
            "Explain machine learning"
        ]

        for query in example_queries:
            print("\n" + "="*70)
            print(f"USER QUERY: {query}")
            print("="*70)

            try:
                answer = generate_with_rag(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    retriever=retriever,
                    embedding_store=embedding_store,
                    temperature=0.5,  # Improved: more coherent answers
                    max_new_tokens=150
                )

                print("\n📋 ANSWER:")
                print("-" * 70)
                print(answer)
                print("-" * 70)
            except Exception as e:
                print(f"Error processing query: {e}")

        # Step 8: Interactive mode
        print("\n" + "="*70)
        print("✨ INTERACTIVE MODE (Type 'quit' to exit)")
        print("="*70)
        print("Ask questions and get AI-powered answers:\n")

        while True:
            try:
                user_query = input("\n❓ Your question: ").strip()

                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Tiny RAG-LLM!")
                    break

                if not user_query:
                    print("Please enter a question.")
                    continue

                answer = generate_with_rag(
                    model=model,
                    tokenizer=tokenizer,
                    query=user_query,
                    retriever=retriever,
                    embedding_store=embedding_store,
                    temperature=0.5,  # Improved: more coherent answers
                    max_new_tokens=150
                )

                print("\n📋 ANSWER:")
                print("-" * 70)
                print(answer)
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()