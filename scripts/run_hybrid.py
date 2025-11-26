# run_hybrid.py
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from hybrid_search import HybridSearcher

def main():
    print("=== Hybrid Retrieval System (BM25 + FAISS + Reranker) ===")
    searcher = HybridSearcher()
    
    while True:
        print("\n" + "="*50)
        query = input("Enter query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue
            
        results = searcher.search(query)
        
        print("\nTop 10 Reranked Results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"{rank}. {doc_id} (Score: {score:.4f})")
            
        answer = searcher.generate_answer(query, results)
        print("\n" + "-"*20 + " LLM Answer " + "-"*20)
        print(answer)

if __name__ == "__main__":
    main()
