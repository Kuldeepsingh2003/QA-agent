from typing import List, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class LocalRetrievalQASystem:
    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 qa_model_name: str = "deepset/roberta-base-squad2"):
        self.embedder = SentenceTransformer(embedding_model_name)
        # Extractive QA pipeline: finds answer span inside context
        self.qa = pipeline("question-answering", model=qa_model_name)
        self.nn = None
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: np.ndarray = None
        self.is_indexed: bool = False

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

    def answer(self, question: str, top_k: int = 5, max_context_chars: int = 2000) -> Dict[str, Any]:
        matches = self.retrieve(question, top_k=top_k)
        context = self.build_context(matches, max_chars=max_context_chars)
        if not context:
            return {"answer": "I could not find relevant information in the documents.", "score": 0.0, "context": "", "matches": matches}
        try:
            qa_out = self.qa(question=question, context=context)
            # qa_out has keys: answer, score, start, end
            return {"answer": qa_out.get("answer", ""), "score": float(qa_out.get("score", 0.0)), "context": context, "matches": matches}
        except Exception as e:
            logger.exception("QA pipeline failed")
            return {"answer": f"QA model error: {e}", "score": 0.0, "context": context, "matches": matches}