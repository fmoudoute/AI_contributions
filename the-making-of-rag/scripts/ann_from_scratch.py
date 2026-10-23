# Requirements: pip install numpy
import time
import numpy as np

rng = np.random.default_rng(42)

# --- A fake "knowledge base": 200,000 normalized 384-dim vectors, grouped in 2,000 topics ---
N, DIM, TOPICS = 200_000, 384, 2_000
topic_centers = rng.normal(size=(TOPICS, DIM))
NOISE = 1.2  # how far a text can drift from its topic


def sample_texts(n):  # a random topic + a lot of noise, like real texts
    v = topic_centers[rng.integers(0, TOPICS, n)] + NOISE * rng.normal(size=(n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


vectors = sample_texts(N)
queries = sample_texts(100)  # new "questions", not copies of stored vectors
print(f"Knowledge base: {N:,} vectors x {DIM} dims = {vectors.nbytes / 1e6:.0f} MB\n")


# --- 1. Exact (brute-force) search: compare the query with EVERY vector ---
def exact_search(q, k=10):
    scores = vectors @ q
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


start = time.perf_counter()
truth = [exact_search(q) for q in queries]
exact_ms = (time.perf_counter() - start) / len(queries) * 1000
print(f"Exact search : {exact_ms:6.2f} ms per query, scanned {N:,} vectors")


# --- 2. IVF (inverted file index): cluster first, then only search the closest clusters ---
def kmeans(data, n_clusters, iterations=10):
    centroids = data[rng.choice(len(data), n_clusters, replace=False)]
    for _ in range(iterations):
        assignment = np.argmax(data @ centroids.T, axis=1)  # closest centroid (cosine)
        for c in range(n_clusters):
            members = data[assignment == c]
            if len(members):
                centroid = members.mean(axis=0)
                centroids[c] = centroid / np.linalg.norm(centroid)
    return centroids


N_LISTS = 450  # rule of thumb: about sqrt(N)
start = time.perf_counter()
sample = vectors[rng.choice(N, 50_000, replace=False)]
centroids = kmeans(sample.copy(), N_LISTS)
assignment = np.concatenate([np.argmax(chunk @ centroids.T, axis=1)
                             for chunk in np.array_split(vectors, 20)])
# Store each cluster's vectors next to each other in memory (like a real vector DB)
order = np.argsort(assignment, kind="stable")
sorted_vectors = vectors[order]
bounds = np.searchsorted(assignment[order], np.arange(N_LISTS + 1))
print(f"IVF index built in {time.perf_counter() - start:.1f} s ({N_LISTS} clusters)\n")


def ivf_search(q, k=10, n_probe=5):
    closest_clusters = np.argsort(-(centroids @ q))[:n_probe]
    candidates = np.concatenate([np.arange(bounds[c], bounds[c + 1]) for c in closest_clusters])
    scores = np.concatenate([sorted_vectors[bounds[c]:bounds[c + 1]] @ q for c in closest_clusters])
    top = np.argsort(-scores)[:k]
    return order[candidates[top]], len(candidates)


for n_probe in [1, 5, 20, 50]:
    start = time.perf_counter()
    results = [ivf_search(q, n_probe=n_probe) for q in queries]
    ms = (time.perf_counter() - start) / len(queries) * 1000
    recall = np.mean([len(set(r[0]) & set(t)) / 10 for r, t in zip(results, truth)])
    scanned = np.mean([r[1] for r in results])
    print(f"IVF n_probe={n_probe:2}: {ms:6.2f} ms per query, scanned {scanned:8,.0f} vectors, recall@10 = {recall:.0%}")
