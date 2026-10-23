import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. The raw text (our tiny "training corpus")
corpus = [
    "Revenue grew strongly in the first quarter",
    "Sales grew strongly across our new stores",
    "Quarterly revenue increased thanks to new stores",
    "Sales increased in the first quarter",
    "The company reported higher revenue and a better margin",
    "The company reported higher sales and a better margin",
    "Operating margin improved during the quarter",
    "The team scored a late goal in the match",
    "The players scored a goal in the final",
    "The striker scored twice during the match",
    "The team won the final after a late goal",
    "Fans celebrated the goal in the stadium",
]
STOP_WORDS = {"the", "a", "in", "and", "to", "our", "after", "during", "across", "thanks"}


def tokenize(text):
    words = text.lower().replace("?", "").replace(",", "").split()
    return [w for w in words if w not in STOP_WORDS]


sentences = [tokenize(s) for s in corpus]
vocab = sorted({w for s in sentences for w in s})
word_to_id = {w: i for i, w in enumerate(vocab)}

# --- STEP 1: Word-Context Co-occurrence Matrix (X) ---
WINDOW = 2
X = np.zeros((len(vocab), len(vocab)))
for s in sentences:
    for i, word in enumerate(s):
        for j in range(max(0, i - WINDOW), min(len(s), i + WINDOW + 1)):
            if i != j:
                X[word_to_id[word], word_to_id[s[j]]] += 1

print(f"Step 1: Co-occurrence matrix shape: {X.shape}")
print(f"        'sales' and 'revenue' appear together {int(X[word_to_id['sales'], word_to_id['revenue']])} times\n")


# --- STEP 2: PPMI ---
def compute_ppmi(matrix):
    total = matrix.sum()
    p_word = matrix.sum(axis=1, keepdims=True) / total
    p_context = matrix.sum(axis=0, keepdims=True) / total
    p_joint = matrix / total
    with np.errstate(divide="ignore"):
        pmi = np.log(p_joint / (p_word * p_context))
    return np.maximum(pmi, 0)  # log(0) = -inf -> clipped to 0


X_ppmi = compute_ppmi(X)

# --- STEP 3: SVD -> dense vectors ---
K = 5
U, S, Vt = np.linalg.svd(X_ppmi)
word_vectors = U[:, :K] * S[:K]
print(f"Step 3: Each word is now a dense vector of {K} floats")
for w in ["revenue", "sales", "goal"]:
    print(f"   {w:8} -> {np.round(word_vectors[word_to_id[w]], 2) + 0.0}")
print()


# --- STEP 4: Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def nearest(word, n=3):
    v = word_vectors[word_to_id[word]]
    scores = [(cosine_similarity(v, word_vectors[i]), w) for i, w in enumerate(vocab) if w != word]
    return sorted(scores, reverse=True)[:n]


print("Step 4: Nearest neighbours")
for w in ["sales", "goal", "margin"]:
    print(f"   {w:8} -> " + ", ".join(f"{o} ({s:.2f})" for s, o in nearest(w)))
print(f"\n   sales vs revenue: {cosine_similarity(word_vectors[word_to_id['sales']], word_vectors[word_to_id['revenue']]):.2f}")
print(f"   sales vs goal   : {cosine_similarity(word_vectors[word_to_id['sales']], word_vectors[word_to_id['goal']]):.2f}")


# --- STEP 5: Visualize the first two dimensions ---
plt.figure(figsize=(8, 6))
finance = ["revenue", "sales", "stores", "margin", "company"]
sports = ["goal", "team", "match", "stadium"]
for w in finance + sports:
    x, y = word_vectors[word_to_id[w], 0], word_vectors[word_to_id[w], 1]
    plt.scatter(x, y, color="tab:orange" if w in finance else "tab:blue", s=80)
    # alternate label placement so close neighbours stay readable
    dx, dy = (-0.45, 0.04) if (finance + sports).index(w) % 2 else (0.06, 0.04)
    plt.text(x + dx, y + dy, w, fontsize=11)
plt.xlim(-3.6, 0.6)
plt.ylim(-2.5, 0.3)
plt.title("Word embeddings built from scratch (PPMI + SVD)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig("embeddings_from_scratch.png", dpi=200)


# --- STEP 6: Semantic search over a tiny knowledge base ---
def embed_text(text):
    ids = [word_to_id[w] for w in tokenize(text) if w in word_to_id]
    if not ids:
        return np.zeros(K)  # every word is out-of-vocabulary
    return word_vectors[ids].mean(axis=0)


knowledge_base = [
    "Revenue grew strongly in the first quarter",
    "Operating margin improved during the quarter",
    "The team won the final after a late goal",
    "Fans celebrated the goal in the stadium",
]
query = "How did sales evolve?"
q_vec = embed_text(query)
print(f"\nStep 6: Query = '{query}'")
print(f"   words the model knows: {[w for w in tokenize(query) if w in word_to_id]}")
for doc in knowledge_base:
    keyword_hits = len(set(tokenize(query)) & set(tokenize(doc)))
    print(f"   keyword={keyword_hits}  cosine={cosine_similarity(q_vec, embed_text(doc)):+.2f}  -> {doc}")
