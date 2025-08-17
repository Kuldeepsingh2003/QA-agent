import pandas as pd
from typing import List, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class LocalRetrievalQASystem:
    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
                 qa_model_name: str = "distilbert-base-cased-distilled-squad"):
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Load QA model
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name)

        self.index = None
        self.chunks = []
        self.is_indexed = False


    def build_index(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided to index.")
        self.corpus_texts = texts
        emb = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.corpus_embeddings = emb
        # Cosine similarity via dot-product if normalized
        self.nn = NearestNeighbors(n_neighbors=min(10, len(texts)), metric="cosine")
        self.nn.fit(emb)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call build_index first.")
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.nn.kneighbors(q_emb, n_neighbors=min(top_k, len(self.corpus_texts)))
        idxs = indices[0].tolist()
        dists = distances[0].tolist()
        # cosine distance -> similarity
        results = []
        for i, dist in zip(idxs, dists):
            sim = 1.0 - float(dist)
            results.append({"text": self.corpus_texts[i], "score": sim})
        # sort by similarity descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def build_context(self, matches: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        context = ""
        for m in matches:
            piece = m["text"].strip()
            if len(context) + len(piece) + 2 <= max_chars:
                context += ("\n" + piece) if context else piece
            else:
                break
        return context

    def answer(self, question: str, top_k: int = 5, max_context_chars: int = 2000):
    # 1. Retrieve with NLP
        matches = self.retrieve(question, top_k=top_k)
        context = self.build_context(matches, max_chars=max_context_chars)

    # 2. Try calculation on retrieved matches
        calc_result = self.calculate_from_matches(question, matches)
        if calc_result is not None:
            return {
            "answer": calc_result,
            "score": 1.0,
            "context": context,
            "matches": matches
             }

    # 3. If no calc found → use QA model
        if not context:
            return {"answer": "No relevant info found.", "score": 0.0, "context": "", "matches": matches}

        qa_out = self.qa(question=question, context=context)
        return {
            "answer": qa_out.get("answer", ""),
            "score": float(qa_out.get("score", 0.0)),
            "context": context,
            "matches": matches
            }

    def calculate_from_matches(self, question, matches):
        max_kw = ["highest", "max", "maximum", "largest", "top"]
        min_kw = ["lowest", "min", "minimum", "smallest", "bottom"]
        sum_kw = ["total", "sum", "overall"]
        avg_kw = ["average", "mean"]

        q_lower = question.lower()
        if any(k in q_lower for k in max_kw): op = "max"
        elif any(k in q_lower for k in min_kw): op = "min"
        elif any(k in q_lower for k in sum_kw): op = "sum"
        elif any(k in q_lower for k in avg_kw): op = "mean"
        else: return None

    # Convert retrieved chunks back into DataFrame for calculation
        rows = []
        for m in matches:
            row_data = self.text_to_dict(m["text"])
            if row_data: rows.append(row_data)

        if not rows: return None
        df = pd.DataFrame(rows)
        numeric_cols = df.select_dtypes(include="number").columns
        if numeric_cols.empty: return None

    # Pick salary/revenue if exists
        target_col = numeric_cols[0]
        for col in numeric_cols:
            if "salary" in col.lower() or "revenue" in col.lower():
                target_col = col
                break

        if op in ["max", "min"]:
            val = getattr(df[target_col], op)()
            row = df[df[target_col] == val]
            return row.to_dict(orient="records")[0]
        else:
            val = getattr(df[target_col], op)()
            return f"{target_col} {op} = {val}"

    def text_to_dict(self, text):
            """
            Simple parser to convert 'col: value; col: value' into dict
            """
            try:
                pairs = [p.strip() for p in text.split(";")]
                return {kv.split(":")[0].strip(): self.try_num(kv.split(":")[1].strip()) for kv in pairs if ":" in kv}
            except:
                return None

    def try_num(self, val):
        try:
            return float(val.replace(",", "")) if any(c.isdigit() for c in val) else val
        except:
            return val
