# Requirements: pip install numpy onnxruntime tokenizers huggingface_hub
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TRANSCRIPT = "files/luckin_coffee_transcript_cleaned.txt"


# --- STEP 1: An embedding model = a tokenizer + a neural network + pooling ---
class EmbeddingModel:
    def __init__(self, model_id, max_tokens):
        self.tokenizer = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(hf_hub_download(model_id, "onnx/model.onnx"))

    def embed(self, texts):
        # Tokenize (Part 2 of the series!)
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # The network returns one vector PER TOKEN: shape (texts, tokens, dims)
        token_vectors = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        })[0]

        # Mean pooling: average the token vectors (ignoring padding) -> one vector per text
        mask = attention_mask[..., None]
        vectors = (token_vectors * mask).sum(axis=1) / mask.sum(axis=1)

        # L2-normalize, so cosine similarity becomes a simple dot product
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --- STEP 2: Chunk the transcript (Part 1 of the series!) ---
def fixed_size_chunk(text, chunk_size=120, overlap=20):
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]


with open(TRANSCRIPT, encoding="utf-8") as f:
    chunks = fixed_size_chunk(f.read())


# --- STEP 3: Keyword search vs. semantic search ---
def keyword_search(query):
    q_words = set(query.lower().replace("?", "").split())
    overlaps = [q_words & set(c.lower().split()) for c in chunks]
    best = int(np.argmax([len(o) for o in overlaps]))
    return best, sorted(overlaps[best])


def semantic_search(model, chunk_vectors, query):
    scores = chunk_vectors @ model.embed([query])[0]  # cosine similarity
    best = int(np.argmax(scores))
    return best, scores[best]


def snippet(text, keyword):
    pos = max(0, text.find(keyword) - 40)
    return "..." + text[pos:pos + 110] + "..."


if __name__ == "__main__":
    english = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", max_tokens=256)
    chunk_vectors = english.embed(chunks)
    print(f"{len(chunks)} chunks -> embedding matrix of shape {chunk_vectors.shape}\n")

    tests = [
        ("How many shops does the company operate?", "total store count"),
        ("Is the company expanding abroad?", "overseas"),
        ("Where does the company source its coffee?", "Yunnan"),
    ]
    for query, evidence in tests:
        k_idx, k_words = keyword_search(query)
        s_idx, s_score = semantic_search(english, chunk_vectors, query)
        print(f"QUERY: {query}")
        print(f"  keyword  -> chunk {k_idx:2}, matched on {k_words}")
        print(f"  semantic -> chunk {s_idx:2}, cosine {s_score:.2f}: {snippet(chunks[s_idx], evidence)}\n")

    # --- STEP 4: The tokenizer decides what the model can "see" ---
    multilingual = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_tokens=128)
    query = "公司有多少家门店？"  # "How many stores does the company have?"
    print(f"QUERY: {query}")
    for name, model in [("all-MiniLM-L6-v2", english),
                        ("paraphrase-multilingual-MiniLM-L12-v2", multilingual)]:
        tokens = model.tokenizer.encode(query).tokens
        best, score = semantic_search(model, model.embed(chunks), query)
        print(f"  {name}")
        print(f"    tokens -> {tokens}")
        print(f"    best   -> chunk {best:2}, cosine {score:.2f}: {chunks[best][:70]}...")
    print()

    # --- STEP 5: Silent truncation: what lies beyond the token limit is invisible ---
    chunk = chunks[7]
    english.tokenizer.no_truncation()
    print(f"Chunk 7: {len(chunk.split())} words -> {len(english.tokenizer.encode(chunk).ids)} tokens (limit: 256)")
    english.tokenizer.enable_truncation(max_length=256)
    original = english.embed([chunk])[0]
    off_topic = "The striker scored a late goal and the fans celebrated in the stadium."
    appended = english.embed([chunk + " " + off_topic])[0]
    prepended = english.embed([off_topic + " " + chunk])[0]
    print(f"  cosine(chunk, chunk + football sentence) = {original @ appended:.4f}")
    print(f"  cosine(chunk, football sentence + chunk) = {original @ prepended:.4f}")
