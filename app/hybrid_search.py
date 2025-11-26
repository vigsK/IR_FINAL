# app/hybrid_search.py
import os
import sys
import glob
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import operator

# Ensure Task1 is in path for BM25
task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Task1'))
if task1_path not in sys.path:
    sys.path.insert(0, task1_path)

try:
    from BM25 import BM25
except ImportError:
    print("Error: Could not import BM25 from Task1.")
    BM25 = None

class HybridSearcher:
    def __init__(self):
        print("Initializing HybridSearcher...")
        # 1. Load BM25
        if BM25:
            self.bm25 = BM25()
            pass
        else:
            self.bm25 = None
            print("Warning: BM25 not available.")

        # 2. Load/Create FAISS Index
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_path = "data/indexes/faiss_index.bin"
        self.doc_ids_path = "data/indexes/doc_ids.json"
        self.dimension = 384 # for all-MiniLM-L6-v2
        
        self.index = None
        self.doc_ids = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.doc_ids_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            import json
            with open(self.doc_ids_path, 'r') as f:
                self.doc_ids = json.load(f)
        else:
            print("FAISS index not found. Indexing documents now (this may take a while)...")
            self.index_documents()

        # 3. Load Cross-Encoder for Reranking
        print("Loading Cross-Encoder...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("HybridSearcher initialized.")

    def index_documents(self):
        """Read parsed pages, embed them, and build FAISS index."""
        docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/documents'))
        files = glob.glob(os.path.join(docs_dir, "*.txt"))
        
        documents = []
        self.doc_ids = []
        
        print(f"Found {len(files)} documents to index.")
        
        # Limit for demo purposes if too many? No, let's try all, it's ~3k docs, should be fast.
        # Actually there are 6408 files in parsed_pages based on list_dir.
        # Batch processing might be better.
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_texts = []
            for f_path in batch_files:
                fname = os.path.basename(f_path)
                try:
                    with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        # Simple truncation to avoid token limit issues (though SentenceTransformer handles it usually)
                        batch_texts.append(text[:1000]) 
                        self.doc_ids.append(fname)
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
            
            if batch_texts:
                embeddings = self.embedding_model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.append(embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i} documents...")

        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            
            # Build Index
            self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (Cosine Similarity if normalized)
            faiss.normalize_L2(final_embeddings)
            self.index.add(final_embeddings)
            
            # Save
            faiss.write_index(self.index, self.index_path)
            import json
            with open(self.doc_ids_path, 'w') as f:
                json.dump(self.doc_ids, f)
            print("Indexing complete and saved.")

    def search(self, query: str, top_k_candidates=100, top_k_final=10, use_reranker=True):
        """
        1. BM25 top-k
        2. Dense top-k
        3. Union
        4. Rerank (optional)
        5. Return top-k_final
        """
        print(f"Searching for: '{query}'")
        
        # 1. BM25 Search
        bm25_results = {}
        if self.bm25:
            raw_bm25 = self.bm25.search(query) # Returns top 100 usually
            for doc_id, score in raw_bm25:
                if not doc_id.endswith('.txt'):
                    doc_id += '.txt'
                bm25_results[doc_id] = score
        
        print(f"BM25 found {len(bm25_results)} candidates.")

        # 2. Dense Search
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k_candidates)
        
        dense_results = {}
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                dense_results[doc_id] = float(score)
        
        print(f"Dense search found {len(dense_results)} candidates.")

        # 3. Union Candidates
        all_candidates = set(bm25_results.keys()) | set(dense_results.keys())
        print(f"Total unique candidates: {len(all_candidates)}")
        
        # OPTIMIZATION: If we have too many candidates, first reduce by simple scoring
        # before expensive reranking
        if use_reranker and len(all_candidates) > 50:
            # Score each candidate by max of normalized BM25 and Dense scores
            candidate_scores = {}
            max_bm25 = max(bm25_results.values()) if bm25_results else 1
            max_dense = max(dense_results.values()) if dense_results else 1
            
            for doc_id in all_candidates:
                bm25_norm = bm25_results.get(doc_id, 0) / max_bm25
                dense_norm = dense_results.get(doc_id, 0) / max_dense
                candidate_scores[doc_id] = max(bm25_norm, dense_norm)
            
            # Take top-50 for reranking
            top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:50]
            all_candidates = [doc_id for doc_id, _ in top_candidates]
        else:
            all_candidates = list(all_candidates)
        
        # Prepare for Reranking
        docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/documents'))
        
        # Cache document content to avoid re-reading
        doc_cache = {}
        
        if use_reranker:
            candidate_pairs = []
            valid_candidates = []
            
            for doc_id in all_candidates:
                path = os.path.join(docs_dir, doc_id)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()[:500] # Reduced from 1000 for faster processing
                        doc_cache[doc_id] = text
                        candidate_pairs.append([query, text])
                        valid_candidates.append(doc_id)
            
            if not candidate_pairs:
                return []

            # 4. Rerank
            print("Reranking candidates...")
            scores = self.reranker.predict(candidate_pairs)
            
            # Combine doc_id with score
            ranked_results = list(zip(valid_candidates, scores))
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_results[:top_k_final]
        else:
            # Skip reranking, just return by combined score
            candidate_scores = {}
            max_bm25 = max(bm25_results.values()) if bm25_results else 1
            max_dense = max(dense_results.values()) if dense_results else 1
            
            for doc_id in all_candidates:
                bm25_norm = bm25_results.get(doc_id, 0) / max_bm25
                dense_norm = dense_results.get(doc_id, 0) / max_dense
                candidate_scores[doc_id] = bm25_norm + dense_norm
            
            ranked_results = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            return ranked_results[:top_k_final]

    def generate_answer(self, query, top_docs):
        """
        Simulate LLM answer generation.
        In a real scenario, we would send the query and the text of top_docs to GPT-4/Gemini.
        Here we will generate a synthetic response citing the docs.
        """
        print("\nGenerating Answer using LLM (Simulated)...")
        
        context = ""
        docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/documents'))
        
        for doc_id, score in top_docs:
            path = os.path.join(docs_dir, doc_id)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()[:200].replace('\n', ' ')
                context += f"[{doc_id}]: {text}...\n"
        
        # Simulated response
        response = f"Based on the retrieved documents, here is the answer to '{query}':\n\n"
        response += "The search results discuss various aspects related to your query. "
        response += f"For instance, document {top_docs[0][0]} mentions key details relevant to the topic. "
        response += f"Another source, {top_docs[1][0]}, provides further context. "
        response += "Overall, the hybrid retrieval system successfully identified these passages as the most pertinent.\n\n"
        response += "### Retrieved Context Snippets:\n"
        response += context
        
        return response
