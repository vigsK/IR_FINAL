# utils.py
"""Utility functions that wrap the existing project modules for use in the Flask UI.
Each function returns data structures that can be rendered in templates.
"""
import os
import subprocess
import json
from typing import List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Helper to run a Python script and capture its stdout
def _run_script(script_path: str, args: List[str] = None) -> str:
    if args is None:
        args = []
    cmd = ["python", script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
    return result.stdout

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
    return result.stdout

# Import backend modules dynamically to handle pathing and conflicts
import sys

# Load Task1 BM25
task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Task1'))
if task1_path not in sys.path:
    sys.path.insert(0, task1_path)

try:
    # Clear Helper from sys.modules if it exists to ensure we load Task1's Helper
    if 'Helper' in sys.modules:
        del sys.modules['Helper']
    from BM25 import BM25
    bm25_model = BM25()
    print("BM25 model loaded successfully.")
except Exception as e:
    print(f"Error loading BM25: {e}")
    bm25_model = None

# Remove Task1 from path to avoid confusion
if task1_path in sys.path:
    sys.path.remove(task1_path)

# Load Task2 BM25WithFeedback
task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Task2'))
if task2_path not in sys.path:
    sys.path.insert(0, task2_path)

try:
    # Clear Helper from sys.modules to ensure we load Task2's Helper
    if 'Helper' in sys.modules:
        del sys.modules['Helper']
    from BM25WithFeedback import BM25WithFeedback
    bm25_feedback_model = BM25WithFeedback()
    print("BM25WithFeedback model loaded successfully.")
except Exception as e:
    print(f"Error loading BM25WithFeedback: {e}")
    bm25_feedback_model = None

# Remove Task2 from path
if task2_path in sys.path:
    sys.path.remove(task2_path)


def run_bm25(query: str) -> List[Tuple[str, float]]:
    """Run Task1 BM25 ranking for a given query."""
    if bm25_model:
        try:
            return bm25_model.search(query)
        except Exception as e:
            print(f"Error executing BM25 search: {e}")
            return []
    return []

def run_bm25_feedback(query: str, find_type: int = 1, k: int = 0) -> List[Tuple[str, float]]:
    """Run Task2 BM25 with relevance feedback."""
    if bm25_feedback_model:
        try:
            return bm25_feedback_model.search(query, find_type, k)
        except Exception as e:
            print(f"Error executing BM25WithFeedback search: {e}")
            return []
    return []

def list_parsed_documents() -> List[str]:
    """List parsed page text files for the docs viewer."""
    base_dir = os.path.abspath("../Snippet generation/parsed_pages")
    if not os.path.isdir(base_dir):
        return []
    files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
    return sorted(files)

def read_parsed_document(filename: str) -> str:
    """Read the content of a parsed document file."""
    base_dir = os.path.abspath("../Snippet generation/parsed_pages")
    path = os.path.join(base_dir, filename)
    if not os.path.isfile(path):
        return "Document not found."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_update_dates() -> str:
    """Execute the automation script that updates dates and author comments.
    Returns the script stdout.
    """
    script = os.path.abspath("../update_dates_authors.py")
    return _run_script(script)


def get_evaluation_metrics():
    """Return evaluation metrics for display on the metrics page.
    The data is hard‑coded based on the values provided by the user.
    Returns a dict with two keys:
        - 'recall': list of dicts with method and recall values at thresholds 0, 0.5, 1
        - 'summary': dict with MAP and MRR values per method
    """
    recall = [
        {"method": "lucene", "0": 0.0, "0.5": 0.032109, "1": 0.060889},
        {"method": "tfidf_stop", "0": 0.0, "0.5": 0.065622, "1": 0.063085},
        {"method": "bm_stop", "0": 0.0, "0.5": 0.031406, "1": 0.060434},
        {"method": "bm25", "0": 0.0, "0.5": 0.033538, "1": 0.061065},
        {"method": "jm_stop", "0": 0.0, "0.5": 0.021798, "1": 0.060884},
        {"method": "bm_queryEnrichment", "0": 0.0, "0.5": 0.030623, "1": 0.059854},
        {"method": "tfidf", "0": 0.0, "0.5": 0.1307, "1": 0.099106},
        {"method": "jm", "0": 0.0, "0.5": 0.031939, "1": 0.059692},
    ]
    summary = {
        "tfidf": {"MAP": 5.187743, "MRR": 16.49683},
        "bm": {"MAP": 13.7392, "MRR": 41.2417},
        "jm": {"MAP": 1.726855, "MRR": 5.998921},
        "bm_stop": {"MAP": 0.5655, "MRR": 1.2689},
        "lucene": {"MAP": 0.2105, "MRR": 0.25},
        "tfidf_stop": {"MAP": 2.472471, "MRR": 8.248473},
        "jm_stop": {"MAP": 1.07886, "MRR": 2.749338},
        "bm_queryEnrichment": {"MAP": 2.363213, "MRR": 6.598571},
    }

    
    # Generate Plots
    graphs = {}
    
    # 1. Recall Comparison (Grouped Bar Chart)
    recall_df = pd.DataFrame(recall)
    # Melt for plotting: method, threshold, value
    recall_melted = recall_df.melt(id_vars=['method'], var_name='Threshold', value_name='Recall')
    
    fig_recall = go.Figure()
    for threshold in ['0', '0.5', '1']:
        subset = recall_melted[recall_melted['Threshold'] == threshold]
        fig_recall.add_trace(go.Bar(
            x=subset['method'],
            y=subset['Recall'],
            name=f'Recall @ {threshold}'
        ))
    fig_recall.update_layout(
        title='Recall at Different Thresholds by Method',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e0e0e0")
    )
    graphs['recall_plot'] = pio.to_html(fig_recall, full_html=False, include_plotlyjs='cdn')

    # 2. MAP vs MRR (Grouped Bar Chart)
    methods = list(summary.keys())
    map_vals = [summary[m]['MAP'] for m in methods]
    mrr_vals = [summary[m]['MRR'] for m in methods]
    
    fig_summary = go.Figure()
    fig_summary.add_trace(go.Bar(
        x=methods,
        y=map_vals,
        name='MAP',
        marker_color='#00cec9'
    ))
    fig_summary.add_trace(go.Bar(
        x=methods,
        y=mrr_vals,
        name='MRR',
        marker_color='#6c5ce7'
    ))
    fig_summary.update_layout(
        title='MAP vs MRR by Method',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e0e0e0")
    )
    graphs['summary_plot'] = pio.to_html(fig_summary, full_html=False, include_plotlyjs=False)

    return {"recall": recall, "summary": summary, "graphs": graphs}

def get_dataset_stats():
    """Generate statistics and plots for the dataset."""
    stats = {'total_docs': 0, 'avg_len': 0}
    graphs = {}

    # 1. Top 20 Terms from unigram_index.txt
    try:
        index_path = os.path.abspath("../Task1/unigram_index.txt")
        term_data = []
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 5000: break # Limit to 5000 terms for performance
                    if line.startswith("Updated on"): continue
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        term = parts[0].strip()
                        postings = parts[1].strip()
                        # Count total frequency
                        count = 0
                        import re
                        matches = re.findall(r'\((.*?):(\d+)\)', postings)
                        for _, freq in matches:
                            count += int(freq)
                        term_data.append((term, count))
            
            # Sort by count desc
            term_data.sort(key=lambda x: x[1], reverse=True)
            top_20 = term_data[:20]
            
            if top_20:
                fig_terms = go.Figure(go.Bar(
                    x=[t[0] for t in top_20],
                    y=[t[1] for t in top_20],
                    marker_color='#00cec9'
                ))
                fig_terms.update_layout(
                    title='Top 20 Terms by Frequency',
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", color="#e0e0e0")
                )
                graphs['top_terms'] = pio.to_html(fig_terms, full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error generating term stats: {e}")

    # 2. Document Length Distribution
    try:
        docs_dir = os.path.abspath("../Snippet generation/parsed_pages")
        doc_lengths = []
        if os.path.isdir(docs_dir):
            for f_name in os.listdir(docs_dir):
                if f_name.endswith('.txt'):
                    with open(os.path.join(docs_dir, f_name), 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc_lengths.append(len(content.split()))
        
        if doc_lengths:
            fig_hist = go.Figure(go.Histogram(
                x=doc_lengths,
                nbinsx=50,
                marker_color='#6c5ce7'
            ))
            fig_hist.update_layout(
                title='Document Length Distribution',
                xaxis_title='Word Count',
                yaxis_title='Number of Documents',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color="#e0e0e0")
            )
            graphs['doc_len_hist'] = pio.to_html(fig_hist, full_html=False, include_plotlyjs=False)
            
            stats['total_docs'] = len(doc_lengths)
            stats['avg_len'] = sum(doc_lengths) / len(doc_lengths)

    except Exception as e:
        print(f"Error generating doc stats: {e}")

    # 3. Clustering Visualization
    try:
        graphs['clustering_plot'] = generate_clustering_plot()
    except Exception as e:
        print(f"Error generating clustering plot: {e}")

    return {"stats": stats, "graphs": graphs}

def generate_clustering_plot():
    """Generate a clustering visualization using TF-IDF, PCA, and K-Means."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import pandas as pd
        
        docs_dir = os.path.abspath("../Snippet generation/parsed_pages")
        documents = []
        doc_ids = []
        
        if os.path.isdir(docs_dir):
            files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
            # Limit to 1000 docs for performance
            files = files[:1000]
            
            for f_name in files:
                with open(os.path.join(docs_dir, f_name), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    doc_ids.append(f_name.replace('.txt', ''))
        
        if not documents:
            return None
            
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # PCA Reduction to 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf_matrix.toarray())
        
        # K-Means Clustering
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': clusters,
            'doc_id': doc_ids,
            'snippet': [d[:100] + '...' for d in documents]
        })
        
        # Plot
        fig = go.Figure()
        
        colors = ['#38bdf8', '#818cf8', '#f472b6', '#34d399', '#fbbf24']
        
        for i in range(num_clusters):
            cluster_data = df[df['cluster'] == i]
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {i+1}',
                marker=dict(size=8, color=colors[i % len(colors)]),
                text=cluster_data['doc_id'] + '<br>' + cluster_data['snippet'],
                hoverinfo='text'
            ))
            
        fig.update_layout(
            title='Document Clustering (TF-IDF + PCA + K-Means)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#e0e0e0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)
        
    except ImportError:
        return "<p>scikit-learn not installed. Cannot generate clustering.</p>"
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"



def get_sample_queries() -> List[str]:
    """Return a list of sample queries for testing."""
    return [
        "computer science",
        "information retrieval",
        "artificial intelligence",
        "machine learning",
        "database systems",
        "operating systems",
        "programming languages",
        "network security",
        "algorithm design",
        "data structures",
        "software engineering",
        "compiler construction",
        "parallel processing",
        "distributed systems",
        "computer graphics"
    ]
