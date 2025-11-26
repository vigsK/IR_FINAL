import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from hybrid_search import HybridSearcher

if __name__ == "__main__":
    print("Starting re-indexing...")
    searcher = HybridSearcher()
    # The __init__ calls index_documents if files are missing, which we just deleted.
    print("Re-indexing complete.")
