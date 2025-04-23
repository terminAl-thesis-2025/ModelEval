#!/usr/bin/env python3
"""
benchmark_chromadb_multilingual.py

Benchmarks embedding speed and retrieval quality on a small synthetic
English-context + German-instruction ↔ bash/SQL dataset, comparing:

  • ChromaDB default embedder (all-MiniLM-L6-v2 via ONNXRuntime)
  • intfloat/multilingual-e5-large
  • intfloat/multilingual-e5-base
  • intfloat/multilingual-e5-small

Requirements:
  pip install chromadb transformers torch haystack sentence-transformers
"""

import time
import numpy as np

# Chroma client & embedding functions
import chromadb
from chromadb.utils import embedding_functions

# For retrieval evaluation
from haystack.utils import convert_labels_to_dict, evaluate_retriever

# ------------------------------------------------------------------------------
# 1) Create a tiny synthetic dataset
# ------------------------------------------------------------------------------
def make_synthetic_dataset():
    """
    Returns:
      docs: list of {'id': str, 'text': str}
      queries: list of {'query_id': str, 'query': str}
      labels: list of (query_id, [relevant_doc_id, ...])
    """
    raw = [
        # (English context, German instruction, target code)
        ("File ops",         "Liste alle Dateien im aktuellen Verzeichnis auf",     "ls -la"),
        ("DB read",          "Zeige alle Einträge aus der Tabelle Benutzer",           "SELECT * FROM users;"),
        ("Permission change","Ändere die Zugriffsrechte der Datei geheim.txt auf 600",   "chmod 600 geheim.txt"),
        ("DB create",        "Erstelle eine neue Tabelle für Produkte mit Spalten id,name",
                                                                          "CREATE TABLE products (id INT, name TEXT);"),
        ("Networking",       "Zeige alle aktiven Netzwerkverbindungen an",             "netstat -tunapl"),
    ]
    docs, queries, labels = [], [], []
    for i, (ctx, instr, _) in enumerate(raw):
        doc_id = f"doc_{i}"
        text = f"{ctx}. {instr}"
        docs.append({"id": doc_id, "text": text})
        # We'll query by the German instruction alone
        queries.append({"query_id": f"q_{i}", "query": instr})
        labels.append((f"q_{i}", [doc_id]))
    return docs, queries, labels

# ------------------------------------------------------------------------------
# 2) Embed & index + measure embedding time + retrieval performance
# ------------------------------------------------------------------------------
def benchmark():
    docs, queries, labels = make_synthetic_dataset()

    # Define our four embedding functions
    models = {
        "chromadb-default": embedding_functions.DefaultEmbeddingFunction(),
        "e5-large": embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda", "batch_size": 8}
        ),
        "e5-base": embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cuda", "batch_size": 16}
        ),
        "e5-small": embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cuda", "batch_size": 32}
        ),
    }

    results = {}

    for name, ef in models.items():
        print(f"\n=== Testing {name} ===")
        # 1) New in-memory Chroma client & collection
        client = chromadb.Client()
        col = client.create_collection(
            name=f"benchmark_{name}",
            embedding_function=ef
        )

        # 2) Embedding + adding documents (sequential!)
        texts = [d["text"] for d in docs]
        ids   = [d["id"]   for d in docs]

        t0 = time.time()
        col.add(
            documents=texts,
            ids=ids,
        )
        # ensure GPU kernels finish
        if hasattr(ef, "model_kwargs") and ef.model_kwargs.get("device", "").startswith("cuda"):
            import torch; torch.cuda.synchronize()
        embed_time = time.time() - t0
        print(f"Embedding + indexing time: {embed_time:.3f}s")

        # 3) Retrieval: get top-3 for each query
        all_preds = {}
        for q in queries:
            qr = col.query(query_texts=[q["query"]], n_results=3)
            # qr["ids"] is [[id1,id2,id3]]
            all_preds[q["query_id"]] = qr["ids"][0]

        # 4) Evaluate using Haystack's util
        label_dict = convert_labels_to_dict(labels)
        metrics = evaluate_retriever(
            query_id2preds=all_preds,
            query_id2relevant_docs=label_dict,
            metric="recall",
            k_values=[1, 3]
        )
        # MRR
        mrr = evaluate_retriever(
            query_id2preds=all_preds,
            query_id2relevant_docs=label_dict,
            metric="mrr"
        )["mrr"]

        print(f"Recall@1: {metrics['recall_1']:.3f}  /  Recall@3: {metrics['recall_3']:.3f}  /  MRR: {mrr:.3f}")

        # store
        results[name] = {
            "embed_time": embed_time,
            "recall@1": metrics["recall_1"],
            "recall@3": metrics["recall_3"],
            "mrr": mrr
        }

        # clean up
        client.delete_collection(name=f"benchmark_{name}")

    # 5) Summary
    print("\n=== Summary ===")
    print(f"{'Model':<20} {'Time(s)':>8}  {'R@1':>5}  {'R@3':>5}  {'MRR':>5}")
    for name, m in results.items():
        print(f"{name:<20} {m['embed_time']:8.3f}  {m['recall@1']:5.3f}  {m['recall@3']:5.3f}  {m['mrr']:5.3f}")

if __name__ == "__main__":
    benchmark()
