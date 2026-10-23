# Requirements: pip install numpy onnxruntime tokenizers huggingface_hub
import math
import re
from collections import Counter

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from embedding_pretrained import EmbeddingModel

TRANSCRIPT = "files/luckin_coffee_transcript_cleaned.txt"


# --- Chunking (Part 1), measured in the embedding model's tokens (the lesson of Part 3) ---
embedder = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", max_tokens=256)


def token_chunk(text, tokenizer, chunk_tokens=128, overlap=32):
    tokenizer.no_truncation()
    enc = tokenizer.encode(text, add_special_tokens=False)
    tokenizer.enable_truncation(max_length=256)
    step = chunk_tokens - overlap
    chunks = []
    for start in range(0, len(enc.offsets), step):
        window = enc.offsets[start:start + chunk_tokens]
        chunks.append(text[window[0][0]:window[-1][1]])
    return chunks


with open(TRANSCRIPT, encoding="utf-8") as f:
    text = " ".join(f.read().split())  # one line, single spaces
    chunks = token_chunk(text, embedder.tokenizer)

# --- A small test set: a question, and a phrase the right chunk MUST contain ---
TEST_SET = [
    ("How many shops does the company operate?", "33,596"),
    ("How many stores were opened in China this quarter?", "2,531 net new stores"),
    ("How many stores are in Malaysia?", "83 franchised stores in Malaysia"),
    ("Where does the company buy its coffee beans?", "Yunnan"),
    ("What is the total roasting capacity?", "155,000 tons"),
    ("How many people work for Luckin?", "190,000 Luckin team members"),
    ("How many customers buy every month?", "93 million"),
    ("What was the GMV?", "RMB14.1 billion"),
    ("How much did S&M expenses increase?", "RMB732 million"),
    ("What was the non-GAAP net profit?", "RMB686 million"),
    ("How much cash does the company have?", "RMB9.1 billion"),
    ("Will shareholders receive dividends?", "not able to pay dividends"),
]


def relevant_ids(evidence):
    return {i for i, c in enumerate(chunks) if evidence in c}


for q, ev in TEST_SET:
    assert relevant_ids(ev), f"no chunk contains: {ev}"


# --- 1. Keyword search done right: BM25 from scratch ---
def tokenize(text):
    return re.findall(r"[a-z0-9&$.,]+", text.lower().replace("?", ""))


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs])
        self.avg_len = self.lengths.mean()
        n = len(docs)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d[w]
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    out[i] += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return out


bm25 = BM25(chunks)

# --- 2. Semantic search (Part 3) ---
chunk_vectors = embedder.embed(chunks)


def semantic_scores(query):
    return chunk_vectors @ embedder.embed([query])[0]


# --- 3. Hybrid search: Reciprocal Rank Fusion ---
def rrf(*score_lists, k=60):
    fused = np.zeros(len(chunks))
    for scores in score_lists:
        ranks = np.argsort(np.argsort(-scores))  # 0 = best
        fused += 1 / (k + ranks + 1)
    return fused


# --- 4. Re-ranking with a cross-encoder ---
class CrossEncoder:
    def __init__(self, model_id="cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.tokenizer = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(hf_hub_download(model_id, "onnx/model.onnx"))

    def score(self, query, docs):
        # The query and the chunk are read TOGETHER, in one input
        enc = self.tokenizer.encode_batch([(query, d) for d in docs])
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        return self.session.run(None, feeds)[0][:, 0]


reranker = CrossEncoder()


def rerank(query, candidate_scores, top_n=20):
    candidates = np.argsort(-candidate_scores)[:top_n]
    new_scores = np.full(len(chunks), -np.inf)
    new_scores[candidates] = reranker.score(query, [chunks[i] for i in candidates])
    return new_scores


# --- 5. Evaluation: Hit@1, Hit@3 and MRR ---
def evaluate(name, search):
    hits1 = hits3 = rr = 0
    per_query = []
    for query, evidence in TEST_SET:
        ranking = list(np.argsort(-search(query)))
        relevant = relevant_ids(evidence)
        rank = min(ranking.index(i) for i in relevant) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        rr += 1 / rank
        per_query.append(rank)
    n = len(TEST_SET)
    print(f"{name:22} Hit@1 = {hits1:2}/{n}   Hit@3 = {hits3:2}/{n}   MRR = {rr / n:.2f}")
    return per_query


if __name__ == "__main__":
    print(f"{len(chunks)} chunks, {len(TEST_SET)} test questions\n")
    results = {
        "BM25 (keyword)": evaluate("BM25 (keyword)", bm25.scores),
        "Semantic": evaluate("Semantic", semantic_scores),
        "Hybrid (RRF)": evaluate("Hybrid (RRF)", lambda q: rrf(bm25.scores(q), semantic_scores(q))),
        "Hybrid + re-rank": evaluate("Hybrid + re-rank",
                                     lambda q: rerank(q, rrf(bm25.scores(q), semantic_scores(q)))),
    }
    print("\nRank of the right chunk, per question:")
    print(f"{'question':48}" + "".join(f"{name[:10]:>12}" for name in results))
    for qi, (query, _) in enumerate(TEST_SET):
        print(f"{query[:47]:48}" + "".join(f"{r[qi]:>12}" for r in results.values()))

    # --- Why did BM25 miss the "shops" question? ---
    query = "How many shops does the company operate?"
    print(f"\nBM25 weights (IDF) for: {query}")
    print("   " + ", ".join(f"{w}={bm25.idf.get(w, 0):.2f}" for w in tokenize(query)))
    best = int(np.argmax(bm25.scores(query)))
    print(f"   BM25 top chunk: ...{chunks[best][:90]}...")

    # --- What did the re-ranker put first for the same question? ---
    best = int(np.argmax(rerank(query, rrf(bm25.scores(query), semantic_scores(query)))))
    i = chunks[best].find("In China")
    print(f"   Re-ranker top chunk: ...{chunks[best][i:i + 95]}...")
