import os
import re
import torch
import numpy as np
import random as rn
import pickle
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, random_split

def set_seed(seed=42):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def clean_and_tokenize(text): 
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9(){}\[\].,!^:;\-_/]", " ", text)
    text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def tokens_to_ids(tokens, word2idx): 
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

def load_data(data_dir="WOS11967"):
    # Adjust path if necessary based on where the script is run
    # Check if data_dir exists directly
    if not os.path.exists(data_dir):
        # Try looking in the current directory if data_dir was relative
        if os.path.exists(os.path.join(os.getcwd(), data_dir)):
             data_dir = os.path.join(os.getcwd(), data_dir)
        # Try looking one level up (common if running from a subdir)
        elif os.path.exists(os.path.join("..", data_dir)):
             data_dir = os.path.join("..", data_dir)
        # Try looking in COMP551_A4/WOS11967
        elif os.path.exists(os.path.join("COMP551_A4", data_dir)):
             data_dir = os.path.join("COMP551_A4", data_dir)
             
    # Load texts and label files. Keep the simple file format used
    # in the assignment: one sample / label per line.
    with open(os.path.join(data_dir, "X.txt"), "r", encoding="utf-8") as f:
        X = [line.strip() for line in f.readlines()]
    with open(os.path.join(data_dir, "YL1.txt"), "r") as f:
        y1 = [int(line.strip()) for line in f.readlines()]
    with open(os.path.join(data_dir, "YL2.txt"), "r") as f:
        y2 = [int(line.strip()) for line in f.readlines()]
    return X, y1, y2

def get_lstm_loaders(data_dir="WOS11967", embeddings_path="saved_models/embeddings.pkl", batch_size=32, max_len=300, embedding_dim=100, seed=42):
    set_seed(seed)
    
    print("Loading data...")
    X, y1, y2 = load_data(data_dir)

    # Check if embeddings exist
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        word2idx = data["word2idx"]
        embedding_matrix = data["embedding_matrix"]
        vocab_size = len(word2idx)
        
        print("Preprocessing data (tokenization only)...")
        tokenized_X = [clean_and_tokenize(line) for line in X]
        
    else:
        # Fallback for local run if file doesn't exist (and gensim is available)
        print(f"Embeddings file {embeddings_path} not found. Attempting to train Word2Vec...")
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("Gensim is not installed and no embeddings file found. Please run build_w2v.py locally first.")

        print("Preprocessing data...")
        tokenized_X = [clean_and_tokenize(line) for line in X]
        word_frequency = Counter(word for line in tokenized_X for word in line)

        # Build vocabulary; reserve indices for PAD and UNK
        idx2word = ['<PAD>', '<UNK>'] + list(word_frequency.keys())
        word2idx = {word: idx for idx, word in enumerate(idx2word)}
        vocab_size = len(word2idx)

        # Train a small Word2Vec model and build the embedding matrix
        print("Training Word2Vec...")
        w2v_model = Word2Vec(sentences=tokenized_X, vector_size=embedding_dim, min_count=1, sg=1)

        embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim)) * 0.01
        for word, idx in word2idx.items():
            if word in w2v_model.wv:
                embedding_matrix[idx] = w2v_model.wv[word]

    # Convert token lists to id sequences and pad/truncate
    sequences = [tokens_to_ids(line, word2idx) for line in tokenized_X]
    full_sequences = [seq[:max_len] + [0]*(max_len-len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
   

    # Convert arrays to torch tensors for use in models
    embedding_matrix_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
    X_tensor = torch.tensor(full_sequences, dtype=torch.long)
    y1_tensor = torch.tensor(y1, dtype=torch.long)
    y2_tensor = torch.tensor(y2, dtype=torch.long)

    # Package into a TensorDataset so DataLoader yields tuples
    dataset = TensorDataset(X_tensor, y1_tensor, y2_tensor)

    # Create deterministic splits: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders — do not shuffle val/test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders created: Train ({len(train_loader)}), Val ({len(val_loader)}), Test ({len(test_loader)})")

    return train_loader, val_loader, test_loader, embedding_matrix_tensor, vocab_size

if __name__ == "__main__":
    # Test the loader
    try:
        train, val, test, emb, vocab = get_lstm_loaders()
        print("Successfully created loaders.")
        print(f"Embedding matrix shape: {emb.shape}, vocab_size: {vocab}")
    except Exception as e:
        print(f"Error creating loaders: {e}")
