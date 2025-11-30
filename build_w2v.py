import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
import sys

# Ensure we can import from data_loader in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, clean_and_tokenize

def build_and_save_embeddings(data_dir="WOS11967", save_path="saved_models/embeddings.pkl", embedding_dim=100):
    print(f"Loading data from {data_dir}...")
    X, _, _ = load_data(data_dir)
    
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
            
    print(f"Saving embeddings and vocab to {save_path}...")
    data_to_save = {
        "word2idx": word2idx,
        "embedding_matrix": embedding_matrix
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print("Done.")

if __name__ == "__main__":
    # Check if WOS11967 exists in current dir or parent
    data_dir = "WOS11967"
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)
        
    build_and_save_embeddings(data_dir=data_dir)
