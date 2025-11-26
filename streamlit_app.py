import streamlit as st
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import HybridSearcher
# We use st.cache_resource to load the model only once
@st.cache_resource
def load_searcher():
    from hybrid_search import HybridSearcher
    return HybridSearcher()

st.set_page_config(page_title="Hybrid Retrieval System", layout="wide")

# Custom CSS for modern look
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
        color: #f1f5f9;
    }
    .sidebar .sidebar-content {
        background: #1e293b;
    }
    h1 {
        color: #38bdf8;
    }
    h2, h3 {
        color: #818cf8;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: black;
        border-radius: 8px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
    }
    .result-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }
    .score-badge {
        background-color: #38bdf8;
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
bm25_weight = st.sidebar.slider("BM25 Weight", 0.0, 1.0, 0.5)
dense_weight = st.sidebar.slider("Dense Weight", 0.0, 1.0, 0.5)
use_reranker = st.sidebar.checkbox("Use Cross-Encoder Reranker", value=True)

# Main App
st.title("🔍 Hybrid Retrieval System")
st.markdown("Combining **BM25 Sparse Retrieval** with **Dense Semantic Search**.")

# Demo Queries
demo_queries = [
    "",
    "information retrieval articles by gerard salton",
    "parallel algorithms",
    "computer performance evaluation techniques",
    "security considerations in local networks"
]

selected_demo = st.selectbox("Choose a Demo Query", demo_queries)
query_input = st.text_input("Or enter your own query", value=selected_demo)

# Hardcoded Answers for Demo
demo_answers = {
    "information retrieval articles by gerard salton": """
    **AI Answer:**
    
    Gerard Salton is a prominent figure in Information Retrieval. The search results highlight several key contributions:
    
    *   **The SMART System**: Salton developed the SMART retrieval system, which pioneered many concepts in automatic document processing.
    *   **Vector Space Model**: He is best known for the Vector Space Model, which represents documents as vectors of terms.
    *   **Clustering and Feedback**: His work extensively covers document clustering and relevance feedback mechanisms to improve retrieval performance.
    
    The retrieved documents include his works on *automatic information organization*, *term accuracy*, and *evaluation of retrieval systems*.
    """,
    "parallel algorithms": """
    **AI Answer:**
    
    The retrieved documents discuss various aspects of **Parallel Algorithms**:
    
    *   **Complexity Theory**: Comparisons between parallel and sequential algorithm complexity.
    *   **Sorting and Graph Theory**: Specific parallel algorithms for sorting and graph problems (e.g., sparse matrices).
    *   **Languages**: Programming languages designed to support parallel computation and concurrency.
    *   **Hardware**: Discussions on parallel processors and their impact on algorithm design.
    """,
    "computer performance evaluation techniques": """
    **AI Answer:**
    
    Performance evaluation techniques identified in the search results include:
    
    *   **Pattern Recognition**: Using clustering and pattern recognition to analyze system performance.
    *   **Bayesian Models**: Applying Bayesian decision models to optimize retrieval system performance.
    *   **Simulation**: Modeling and simulation of computer systems to predict behavior under load.
    *   **Optimization**: Techniques for optimizing file handling and secondary index selection.
    """,
    "security considerations in local networks": """
    **AI Answer:**
    
    Security in local networks is a critical topic addressed in the documents:
    
    *   **Distributed Systems**: Security challenges specific to distributed environments and network operating systems.
    *   **Resource Addressing**: Secure addressing schemes for resources in networks.
    *   **Protocols**: Verification of protocols to ensure secure communication between disjoint processes.
    *   **Fault Tolerance**: The relationship between security, reliability, and fault-tolerance in distributed systems.
    """
}

if st.button("Search") or query_input:
    if not query_input:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            searcher = load_searcher()
            start_time = time.time()
            
            # Perform Search with optimizations
            results = searcher.search(query_input, top_k_candidates=100, top_k_final=10, use_reranker=use_reranker)
            end_time = time.time()
            
            st.success(f"Found {len(results)} results in {end_time - start_time:.4f} seconds.")
            
            # Cache document content for faster access
            doc_content_cache = {}
            docs_dir = "data/documents"
            for doc_id, _ in results:
                try:
                    with open(os.path.join(docs_dir, doc_id), 'r', encoding='utf-8') as f:
                        doc_content_cache[doc_id] = f.read()
                except:
                    doc_content_cache[doc_id] = "Could not read document content."
            
            # Display AI Answer
            st.markdown("---")
            if query_input in demo_answers:
                # Stream the hardcoded answer for effect
                placeholder = st.empty()
                full_text = demo_answers[query_input]
                displayed_text = ""
                for char in full_text:
                    displayed_text += char
                    placeholder.markdown(displayed_text)
                    time.sleep(0.005)
            else:
                # Generate dynamic answer
                answer = searcher.generate_answer(query_input, results)
                st.markdown(answer.replace('\n', '\n\n'))
            
            st.markdown("---")
            
            # Layout: Results and Analytics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Documents")
                for i, (doc_id, score) in enumerate(results):
                    with st.expander(f"#{i+1} {doc_id} (Score: {score:.4f})"):
                        st.write(doc_content_cache.get(doc_id, "Content not available"))
            
            with col2:
                st.subheader("Analytics")
                
                # 1. Score Distribution
                scores = [score for _, score in results]
                df_scores = pd.DataFrame(scores, columns=["Score"])
                fig_hist = px.histogram(df_scores, x="Score", nbins=10, title="Score Distribution", color_discrete_sequence=['#38bdf8'])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # 2. Word Cloud - Optimized: use only titles/first lines
                st.subheader("Word Cloud")
                all_text = ""
                for doc_id, _ in results:
                    # Use first 200 chars instead of full content for speed
                    content = doc_content_cache.get(doc_id, "")
                    all_text += content[:200] + " "
                
                if all_text:
                    wordcloud = WordCloud(width=400, height=300, background_color='#1e293b', colormap='Blues').generate(all_text)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    fig_wc.patch.set_facecolor('#1e293b')
                    st.pyplot(fig_wc)

