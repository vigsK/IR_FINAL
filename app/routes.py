# app/routes.py
from flask import render_template, request, redirect, url_for, flash
from . import app
from .hybrid_search import HybridSearcher

# Initialize searcher once (global for simplicity in this demo)
searcher = None

def get_searcher():
    global searcher
    if searcher is None:
        searcher = HybridSearcher()
    return searcher

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    answer = ""
    query = ""
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            s = get_searcher()
            results = s.search(query)
            answer = s.generate_answer(query, results)
    
    return render_template('search.html', query=query, results=results, answer=answer)
