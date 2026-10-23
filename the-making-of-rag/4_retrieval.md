# The Making of RAG: Retrieval (Part 4)

## Introduction

This is the fourth article in my *Making of RAG* series. So far, we have prepared our text with
[chunking](https://medium.com/@fmoudoute/chunking-the-making-of-rag-part-1-d273260d1c9c) and
[tokenization](https://medium.com/@fmoudoute/tokenization-the-making-of-rag-part-2-cea980ba5e03), and in Part 3 we
turned every chunk into a vector with **embeddings**. Our building has its foundation, its materials and its blueprint.

Now someone walks into the building with a question. Somebody has to know, among thousands of rooms, which ones hold the
answer, and has to find them in a few milliseconds. That somebody is **retrieval**, the concierge of our building (yes,
another metaphor... I promise this is the last one for this article :).

At the end of Part 3, I promised three things for this article: how we search vectors at scale, why the top result is
not always the best one, and how to combine semantic and keyword search. We will do all three, still on the same Luckin
Coffee Q1 2026 earnings transcript. But this time, I want to go one step further than "look, it works on my example".
We will **measure** it.

As in the previous articles, every number you'll see comes from code I ran myself, and the full scripts are at the end.

*(As always, if you find this helpful, a thumbs-up helps others find it and keeps me writing!)*

---

## Retrieval, or finding the needle

Retrieval is the step that takes the user's question and returns the K chunks (usually 3 to 10) that will be given to
the LLM as context. It sounds simple, and in Part 3 we already did a version of it with a single dot product. But
retrieval is where a RAG system succeeds or fails, and it has the same impacts I used in the previous articles:

### 1 - Accuracy

The LLM can only answer from what it receives. If the right chunk is not in the top K, the model will either say "I
don't know" (the good case) or answer confidently from the wrong chunk (the bad case). In my experience, when people
say "the LLM hallucinated", the cause very often turns out to be a retrieval problem.

### 2 - Cost

Every chunk we retrieve is pasted into the prompt, and as we saw in Part 2, every token is paid for. Retrieving 10
chunks of 128 tokens adds about 1,300 tokens to *every* question. Better retrieval means we can afford a smaller K,
and a cheaper prompt.

### 3 - Process Efficiency

Retrieval runs while the user is waiting. Comparing a question with 107 chunks is instant; comparing it with 10 million
chunks is not. The way we search (exactly, or approximately) decides the latency of the whole system.

---

## Before anything: a test set

You can't improve what you don't measure. In Part 3, I tested semantic search with three questions that I had chosen
myself, and I could see that it worked. That's a demo, not a measurement. So before comparing retrieval methods, I built
a small **test set**: 12 questions about the transcript, each with a short phrase that the right chunk *must* contain.

```python
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
```

I mixed two kinds of questions on purpose. Some use words that are *not* in the transcript ("shops", "people work for"),
and some use exact terms, acronyms and numbers ("GMV", "S&M", "non-GAAP").

For each question, we look at the **rank** of the right chunk in the results, and we compute three classic metrics:

* **Hit@1**: how many times the right chunk is ranked first.
* **Hit@3**: how many times it is in the top 3 (if we give 3 chunks to the LLM, this is what matters).
* **MRR (Mean Reciprocal Rank)**: the average of 1/rank. Rank 1 gives 1.0, rank 2 gives 0.5, rank 4 gives 0.25. An MRR
  of 1.0 means "always first".

Twelve questions is small, and I'll come back to that. But it's already enough to see real differences, and it's
infinitely better than zero.

### Chunking, the Part 3 way

In Part 3, we saw that one of our 120-word chunks was 307 tokens long because of the Chinese paragraphs, and that the
embedding model silently ignored everything after token 256. So this time I took my own advice and chunked **in the
tokens of the embedding model**: 128 tokens per chunk, with 32 tokens of overlap.

```python
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
```

The trick is in `enc.offsets`: for every token, the tokenizer tells us where it starts and ends in the original text, so
we can cut the text exactly on token boundaries. The result: **107 chunks, and none goes over 130 tokens** (128 plus
the two special tokens). Nothing is truncated anymore.

---

## Keyword search, done right: BM25

In Part 3, our keyword search simply counted the words in common between the question and the chunk, and it got
fooled by words like "the" and "company". The fix has existed for decades, and it's still used everywhere today
(Elasticsearch, OpenSearch, Lucene...): **BM25**.

BM25 improves on word counting with three ideas:

1. **Rare words matter more (IDF).** A word that appears in every chunk ("the") says nothing; a word that appears in one
   chunk ("Yunnan") says a lot.
2. **Repetition has diminishing returns (k1).** A chunk that says "revenue" 10 times is not 10 times more relevant than a
   chunk that says it once.
3. **Long chunks are penalized (b).** A long chunk has more chances to contain any word, so its score is normalized by
   its length.

Here it is from scratch, in about 25 lines:

```python
import math
import re
from collections import Counter


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
```

Note that my `tokenize` keeps characters like `&`, `$`, `.` and `,`, so that "S&M", "$300" or "33,596" stay in one
piece. As we saw in Part 2, the way you tokenize decides what you can find!

And for the semantic side, nothing new: it's exactly the `all-MiniLM-L6-v2` model and the dot product from Part 3.

```python
chunk_vectors = embedder.embed(chunks)


def semantic_scores(query):
    return chunk_vectors @ embedder.embed([query])[0]
```

---

## Round 1: keyword vs. semantic

Here are the results on our 12 questions (the full evaluation code is in the script at the end):

#### After:

```text
BM25 (keyword)         Hit@1 =  6/12   Hit@3 =  9/12   MRR = 0.66
Semantic               Hit@1 =  7/12   Hit@3 = 10/12   MRR = 0.73
```

Semantic search wins, but not by much. And that's where it gets interesting, because they don't fail on the same
questions. Here is the rank of the right chunk for each question (1 = perfect):

```text
question                                          BM25 (keyw    Semantic
How many shops does the company operate?                  17           1
How many stores are in Malaysia?                           4           1
How much cash does the company have?                       4           1
How many people work for Luckin?                           2           4
How many customers buy every month?                        1           5
What was the non-GAAP net profit?                          2           3
```

**Where BM25 fails.** "How many shops does the company operate?" puts the right chunk at rank **17**. Let's look at the
weights BM25 gives to each word of the question:

```text
BM25 weights (IDF) for: How many shops does the company operate?
   how=4.28, many=0.00, shops=0.00, does=0.00, the=0.48, company=3.43, operate=0.00
   BM25 top chunk: ...to know whether the company will consider adapting other shareholder reward initiatives su...
```

This is the funniest result of the article. "Shops", "many", "does" and "operate" never appear in the transcript, so they
score zero. And "how" appears only **once** in the whole transcript, which makes it, according to IDF, one of the
"rarest and most meaningful" words of the question! BM25 ended up returning an analyst's question about dividends,
because it contains "how" and "company". IDF is a great idea on a large corpus; on 107 chunks, it's easy to fool.

**Where semantic search fails.** "How many people work for Luckin?" ranks the right chunk 4th. The answer is in a
sentence about "our 190,000 Luckin team members", and the top 3 were the introduction chunks of the call, full of
"Luckin Coffee". The word "Luckin" dominated the vector, and the meaning of "people work for" ≈ "team members" was not
strong enough. BM25 ranked it 2nd, but mostly by luck: only "for" and "Luckin" matched, and the chunk it ranked 1st
matched exactly the same two words.

So each method sees something the other misses. Can we get the best of both?

---

## Round 2: hybrid search with Reciprocal Rank Fusion

The obvious idea would be to add the two scores. But the scores don't live on the same scale: a BM25 score of 5.2 and
a cosine similarity of 0.48 can't be compared. The elegant solution is to **ignore the scores and only keep the
ranks**. That's **Reciprocal Rank Fusion (RRF)**, introduced by Cormack, Clarke and Büttcher in 2009:

`RRF(chunk) = Σ 1 / (k + rank of the chunk in each result list)`, with k = 60 by convention.

A chunk ranked first by both methods gets the highest score; a chunk ranked first by one method and 50th by the other
still gets a decent score. And the code is tiny:

```python
def rrf(*score_lists, k=60):
    fused = np.zeros(len(chunks))
    for scores in score_lists:
        ranks = np.argsort(np.argsort(-scores))  # 0 = best
        fused += 1 / (k + ranks + 1)
    return fused


hybrid_scores = rrf(bm25.scores(query), semantic_scores(query))
```

(The double `argsort` is a small numpy trick: the first one sorts the chunks, the second one gives the rank of each
chunk.)

#### After:

```text
BM25 (keyword)         Hit@1 =  6/12   Hit@3 =  9/12   MRR = 0.66
Semantic               Hit@1 =  7/12   Hit@3 = 10/12   MRR = 0.73
Hybrid (RRF)           Hit@1 =  9/12   Hit@3 = 12/12   MRR = 0.86
```

**12 out of 12 questions have the right chunk in the top 3.** Neither method could do that alone. "How many people work
for Luckin?" jumps from rank 4 (semantic) and rank 2 (BM25) to rank 1, and "How many shops?" goes from rank 17 (BM25)
to rank 2. This is why hybrid search has become the default in most production RAG systems I have seen, and why most
vector databases now offer it out of the box.

---

## Round 3: re-ranking, or why the top result is not always the best one

There is one more limit, and it comes from how embeddings work. In Part 3, the question and the chunk were embedded
**separately**, each one into its own vector, and then compared. This type of model is called a **bi-encoder**. It's
fast (chunk vectors are computed once, in advance), but the model never sees the question and the chunk *together*.
Everything the chunk says has to be squeezed into one vector *before* anyone knows what the question will be, so small
but important details (is it the China store count or the total store count?) can get lost.

A **cross-encoder** reads the question and the chunk together, in a single input, and outputs one relevance score.
It's much more accurate, but much slower: it must run the full model once per (question, chunk) pair, at query time,
and nothing can be computed in advance. Running it on millions of chunks is out of the question.

So we use both, in two stages:

1. **Retrieve** a few candidates fast (here, the top 20 of the hybrid search).
2. **Re-rank** only those 20 with the cross-encoder.

```python
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
```

Notice the `(query, d)` pair given to the tokenizer: this is what makes it a cross-encoder. The tokenizer builds a
single input `[CLS] question [SEP] chunk [SEP]`, and the `type_ids` tell the model which part is which.

#### After:

```text
BM25 (keyword)         Hit@1 =  6/12   Hit@3 =  9/12   MRR = 0.66
Semantic               Hit@1 =  7/12   Hit@3 = 10/12   MRR = 0.73
Hybrid (RRF)           Hit@1 =  9/12   Hit@3 = 12/12   MRR = 0.86
Hybrid + re-rank       Hit@1 = 10/12   Hit@3 = 11/12   MRR = 0.89
```

The re-ranker improves Hit@1 (10 out of 12) and the MRR... but **Hit@3 went down** from 12 to 11. One question got
worse: "How many shops does the company operate?" went from rank 2 to rank **5**. So I looked at what the re-ranker put
first:

```text
   Re-ranker top chunk: ...In China, we added 2,531 net new stores during the quarter, bringing our total store count to 3...
```

The sentence continues with "...to 33,419". And honestly? **That is a correct answer.** 33,419 is the number of stores
in China, and 33,596 is the total including the 177 overseas stores. The question "how many shops does the company
operate?" is ambiguous, and my test set only accepted one of the two answers. The re-ranker wasn't wrong; **my test set
was too strict**.

I decided to leave this result as it is in the article, because it teaches two lessons I find more valuable than a
perfect score:

* **A re-ranker is not magic.** It usually improves the top of the ranking, but it can move things in ways you don't
  expect, so measure it on your own data before adding it (it also adds latency at every query).
* **Your evaluation is only as good as your test set.** Before blaming the model, look at the failures one by one.
  Sometimes the model is right and the "expected answer" is wrong. In a real project, I would accept both chunks for this
  question, and add more questions: 12 is enough to see trends, not to make a final decision.

---

## Retrieval at scale: exact vs. approximate search

So far, we compared each question with all 107 chunks. This is called **exact** (or brute-force) search, and for 107
chunks it's instant. But what happens with millions of chunks?

To test this, I needed a much bigger knowledge base than one transcript. So I generated **200,000 synthetic vectors** of
384 dimensions (like `all-MiniLM-L6-v2`), grouped into 2,000 "topics" with a lot of noise, to imitate real texts that
talk about the same subjects in different ways. Let me be clear: these are not real embeddings, and the timings depend
on my laptop. But the *trade-off* they show is real.

### Exact search

```python
def exact_search(q, k=10):
    scores = vectors @ q
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]
```

### Approximate search: IVF from scratch

The idea of an **IVF index (Inverted File index)** is simple. Before any question arrives, we group the vectors into
clusters with k-means. When a question arrives, we first find the few clusters closest to it, and we only search
inside those clusters. It's like a library: you don't read every book, you go to the right shelf first.

```python
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
sample = vectors[rng.choice(N, 50_000, replace=False)]
centroids = kmeans(sample.copy(), N_LISTS)
assignment = np.concatenate([np.argmax(chunk @ centroids.T, axis=1)
                             for chunk in np.array_split(vectors, 20)])
# Store each cluster's vectors next to each other in memory (like a real vector DB)
order = np.argsort(assignment, kind="stable")
sorted_vectors = vectors[order]
bounds = np.searchsorted(assignment[order], np.arange(N_LISTS + 1))


def ivf_search(q, k=10, n_probe=5):
    closest_clusters = np.argsort(-(centroids @ q))[:n_probe]
    candidates = np.concatenate([np.arange(bounds[c], bounds[c + 1]) for c in closest_clusters])
    scores = np.concatenate([sorted_vectors[bounds[c]:bounds[c + 1]] @ q for c in closest_clusters])
    top = np.argsort(-scores)[:k]
    return order[candidates[top]], len(candidates)
```

The `n_probe` parameter is the number of clusters we visit. To measure the quality, we compare the 10 results of IVF with
the 10 results of the exact search: that's the **recall@10** (10 out of 10 = 100%).

#### After:

```text
Knowledge base: 200,000 vectors x 384 dims = 307 MB

Exact search :   5.90 ms per query, scanned 200,000 vectors
IVF index built in 2.6 s (450 clusters)

IVF n_probe= 1:   0.14 ms per query, scanned      468 vectors, recall@10 = 89%
IVF n_probe= 5:   0.43 ms per query, scanned    2,283 vectors, recall@10 = 92%
IVF n_probe=20:   1.51 ms per query, scanned    8,978 vectors, recall@10 = 95%
IVF n_probe=50:   3.72 ms per query, scanned   22,361 vectors, recall@10 = 97%
```

This table is, for me, the essence of approximate search. With `n_probe=1`, we scan **0.2%** of the vectors and search
**about 40 times faster**, but we lose 11% of the true top-10 results. With `n_probe=50`, we get 97% of them back, but
the speed-up is almost gone. **There is no free lunch: approximate search trades a bit of accuracy for a lot of speed**,
and `n_probe` is the knob that sets the balance.

At 200,000 vectors, exact search at 6 ms is still perfectly fine, and that is an important point: **you don't always
need approximate search**. For a few hundred thousand chunks, a brute-force search (or pgvector without an index) is
simpler and exact. Approximate search becomes necessary in the millions, where exact search grows linearly with the
number of chunks.

In production, vector databases like Qdrant, Milvus, Weaviate, Pinecone or pgvector implement these ideas in optimized
code, and many of them rely on another family of indexes called **HNSW** (a graph where each vector is connected
to its neighbours, and the search "walks" toward the question). Different algorithm, same trade-off, and the same kind
of knob (`ef_search` instead of `n_probe`).

---

## Practical considerations

1. **Build a test set first.** Even 20 to 50 questions with their expected chunks will tell you more than any
   leaderboard. Re-run it every time you change the chunking, the embedding model or the retrieval method.
2. **Start with hybrid search.** BM25 catches exact terms (product names, acronyms, numbers, IDs); embeddings catch
   meaning. RRF combines them without any tuning.
3. **Add a re-ranker if your top-1 matters**, and measure it. It adds latency at every query (here, 20 extra model calls
   per question), so retrieve a reasonable number of candidates (20 to 50) and re-rank only those.
4. **Choose K with the cost in mind.** Every retrieved chunk is paid for in the prompt. If your Hit@3 is already high,
   giving 10 chunks to the LLM mostly adds cost and noise.
5. **Don't rely on a fixed similarity threshold.** As we saw in Part 3, a good match can have a cosine of 0.48. Scores
   depend on the model and on the length of the texts; ranks are much more robust.
6. **Exact first, approximate when you need it.** Below a few hundred thousand chunks, exact search is simple and fast
   enough. Above that, use the approximate index of your vector database, and tune its knob with your test set.

---

## Conclusion

Retrieval is the moment where all the previous steps meet. The chunking decides *what* can be found, the tokenization
decides what each method can *see*, and the embeddings decide what "similar" means. And as we saw, no single method
wins everywhere: BM25 was fooled by the word "how", the embedding model by the word "Luckin", and the re-ranker
"failed" on a question where it was actually right.

What I take from this experiment is that retrieval is less about choosing *the best* algorithm, and more about
**combining** complementary methods and **measuring** them honestly on your own data. Going from 9 to 12 questions in the
top 3 didn't require a bigger model, only a better combination of two simple ones.

In the next article, I'll cover the last step of the pipeline: **generation**. How we turn the retrieved chunks into a
prompt, how to make the LLM answer *only* from them (and cite them), and what happens when retrieval gives it the
wrong context.

If you've found this series helpful, I'd really appreciate a follow. It helps me know that people are reading and
encourages me to keep writing.

---

## Run it yourself

The scripts are in the `scripts/` folder of my project and should be run from that folder:

* `retrieval_eval.py`: BM25, semantic search, RRF, re-ranking and the evaluation on the 12 questions. It requires
  `pip install numpy onnxruntime tokenizers huggingface_hub`, and the `embedding_pretrained.py` file from Part 3 in the
  same folder. The models are downloaded automatically from Hugging Face on the first run.
* `ann_from_scratch.py`: exact search vs. IVF on 200,000 synthetic vectors. It requires only `numpy` (and about 2 GB of
  free memory). Your timings will differ from mine, but the trade-off should look the same.

## References
- S. Robertson & H. Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*
- G. Cormack, C. Clarke & S. Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*
- R. Nogueira & K. Cho (2019), *Passage Re-ranking with BERT*
- H. Jégou, M. Douze & C. Schmid (2011), *Product Quantization for Nearest Neighbor Search* (IVF indexes)
- Y. Malkov & D. Yashunin (2016), *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
- [cross-encoder/ms-marco-MiniLM-L6-v2 model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
