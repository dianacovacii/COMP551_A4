from data_loader import load_data
import numpy as np

def count_classes():
    print("Loading data...")
    try:
        X, y1, y2 = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    unique_domains = set(y1)
    unique_subdomains = set(y2)

    num_domains = len(unique_domains)
    num_subdomains = len(unique_subdomains)

    print(f"Number of unique domains (y1): {num_domains}")
    print(f"Unique domains: {sorted(list(unique_domains))}")
    
    print(f"Number of unique subdomains (y2): {num_subdomains}")
    print(f"Unique subdomains range: min={min(unique_subdomains)}, max={max(unique_subdomains)}")

if __name__ == "__main__":
    count_classes()
