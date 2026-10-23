# The Making of RAG: Tokenization

## Introduction

This is my second article on Medium, and it continues my *Making of RAG* series. For those reading me for the first time,
my aim in these articles—and in any conference where I speak or join panels—is to bring ML, AI (including GenAI) to life
in a way I wish I had been introduced to when I started.

In my previous article in this series, I covered chunking: the segmentation of source material into discrete textual
units that serve as the foundation for retrieval and generation. I presented several chunking methods without explicitly
clarifying whether some of them should be applied before or after tokenization. For basic methods like fixed-size chunking,
tokenization should be performed *prior* to chunking. Together, tokenization and chunking enable embedding. The tokenization
phase is therefore the first and most fundamental step in the entire RAG pipeline.

To revisit the "foundation of a building" example from the previous article, tokenization can be perceived as the raw
material—the bricks, the steel, the concrete—from which everything else is constructed. Before building a foundation,
we need to understand the properties of the materials. Without understanding how our text is being represented at the
most basic level, we cannot build an effective RAG system.

Just like with chunking, I noticed that many LLM users are not familiar with the tokenization process—and why should
they? Yet, as we will see here (and I'll even try to put a dollar amount on some of these effects), the choice of
tokenizer and tokenization strategy has profound implications for everything that follows: chunk boundaries, embedding
quality, retrieval accuracy, generation cost, and ultimately the quality of the final response.

If you find this article helpful and it clarifies something for you, I'd really appreciate a thumbs-up at the top/bottom of
the page. It helps others find this content and encourages me to keep writing more in this series.


## Tokenization, another hidden ingredient

I also like to compare tokenization in RAG to the preparation of ingredients before cooking (ok...building, cooking... :). There is no
single correct way to prepare every ingredient for every dish. Some ingredients need to be chopped finely, others roughly;
some need to be peeled, others not. The same principle applies to tokenization. Depending on the nature of the text,
the model we are using, and the specific requirements of our application, we may choose different tokenization
strategies.

But unlike chunking, where the engineer has a high degree of control over the strategy and implementation, tokenization
is often constrained by the underlying model that we plan to use. Most modern language models come with a predetermined tokenizer
that was trained alongside the model itself, where the model has been trained to learn to understand text through the lens of a
specific tokenization scheme.

There are three well-documented impacts of tokenization quality on a RAG pipeline:

### 1 - Accuracy

Tokenization directly impacts what the model can "see." If a tokenizer splits a word into multiple pieces in a way
that alters its meaning, or if it treats two semantically similar words as completely unrelated tokens, the model may
struggle to understand the text correctly. For example, consider the word "tokenization." A subword tokenizer might split
this into "token" + "ization," preserving the morphological relationship to "token." A character-level tokenizer might
split it into individual letters, losing that semantic connection entirely. A word-level tokenizer might treat "tokenization"
and "token" as completely unrelated tokens, missing the clear relationship between them. That says it all!


### 2 - Cost

Tokenization has a direct and sometimes quite surprising impact on cost. Most commercial LLM APIs (OpenAI, Anthropic,
Google Gemini, DeepSeek, Alibaba Qwen, Moonshot Kimi..) charge by the token, but the number of tokens that are required
to represent a given piece of text varies dramatically depending on the tokenizer. For example, the same English text
might require 10 tokens with one tokenizer and 15 tokens with another. Over millions of documents, the bill will add up!!!

As a quick example, a tokenizer that splits "co-operate" as "co" + "operate" might use fewer tokens than one that splits
it as "co-" + "operate," but the difference of a single token per occurrence across a large corpus adds up quickly.

This is particularly relevant for RAG because the cost of generation is often dominated by the length of the context
being fed to the model. A chunking strategy that produces 500 tokens per chunk with one tokenizer might produce 800
tokens per chunk with another, which would dramatically increase generation costs.


### 3 - Context Preservation

Tokenization also affects how well the model can preserve the semantic context of a chunk (as you can see from the previous
article, there are a bunch of similarities between the costs implied by **chunking** and **tokenization**). If a tokenizer
breaks a word into meaningless pieces, or if it groups unrelated characters into a single token that doesn't correspond
to a meaningful unit, the model loses information that might be crucial for understanding the input.

Context preservation might also be affected by the tokenizer's ability to handle out-of-vocabulary words.
A tokenizer that relies heavily on a fixed vocabulary may split unfamiliar words into many small pieces, losing the coherence
of the original term. This is why subword tokenization has become the dominant approach: it balances the efficiency of
word-level tokenization with the flexibility of character-level tokenization.

The question came in the past, as in what do I mean by subword tokenization... let's look at a quick example:

```text
"tokenization" → ["token", "ization"]
"token" → ["token"]
"tokens" → ["token", "s"]
"tokenized" → ["token", "ized"]
```


## Tokenization and the RAG pipeline

To understand why tokenization matters for RAG, we need to understand where it fits in the pipeline. The typical flow
looks like this:

* **Document Ingestion**: Raw text enters the system
* **Tokenization**: Text is converted to tokens (the focus of this article)
* **Chunking**: Tokens are grouped into chunks (covered in the previous article)
* **Embedding**: Each chunk is converted to a vector representation
* **Indexing**: Vectors are stored in a vector database
* **Retrieval**: User queries are tokenized, embedded, and matched against stored vectors
* **Generation**: Retrieved context is fed to a language model for response generation

Notice that tokenization happens twice: first during ingestion (when documents are processed for indexing) and again at query time (when user queries are processed). Both the ingestion tokenizer and the query tokenizer are typically determined by the underlying model.

A subtle but important point is that the tokenizer used for chunking and the tokenizer used for the final generation
model may be different. If we are using a different model for embedding than for generation, we need to ensure that
the tokenization is handled correctly at each stage.


## Different methods of tokenization

Let me be honest about something upfront (repeating myself here..., but it is quite important): unlike chunking, where the engineer has significant choice in strategy,
tokenization choices are largely determined by the model we are using. Modern transformer models come with a specific
tokenizer that was trained alongside the model, and you cannot easily replace it.

That said, understanding the different tokenization approaches is essential for making informed decisions about which
models to use and how to configure your RAG pipeline. Here are the main tokenization methods - feel free to comment if I
missed some methods:

### Character-Level Tokenization

Character-level tokenization is the most granular approach, treating each character as a separate token. A sentence like
"Hello world" would be tokenized as ["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] - this is an entry-level approach
that I have never used.

This approach has significant advantages: it has no out-of-vocabulary issues (everything can be represented), it works
well for any language, and it's simple to implement. However, it also has major drawbacks: sequences become much longer
(a 100-word document might become 500+ tokens), and the model must learn to combine characters into meaningful units,
which is computationally expensive and difficult.

Now, let's be honest, character-level tokenization is rarely used in modern RAG systems due to its inefficiency,
but it's worth understanding as a baseline.


### Word-Level Tokenization

Word-level tokenization treats each word as a separate token, splitting on whitespace and sometimes punctuation.
"Hello world" becomes ["Hello", "world"].

This approach is intuitive and efficient, but it poses significant problems. The vocabulary size becomes enormous, leading
to memory and computational issues. It also has severe out-of-vocabulary problems: any word not in the vocabulary cannot
be represented, forcing the model to use an "unknown token" and losing information. Additionally, it handles morphological
variations poorly: "run," "running," and "ran" are all different tokens, even though they share a common root.

I believe we need to mention word-level tokenization, just for the sake of it, but it is pretty much obsolete for modern
NLP systems, replaced by subword tokenization, which we jump right into now!


## Subword Tokenization

Subword tokenization sits right between word-level and character-level tokenization. It breaks words into smaller units that
are still meaningful enough to preserve linguistic patterns. As a result, the same word can be split differently depending on how
common it is in the training data: frequent words stay intact, while rare words are decomposed into smaller, reusable
subwords. This approach keeps the vocabulary at a manageable size, reduces out-of-vocabulary issues, and still allows
models to learn efficiently from reasonably short sequences.

I will go through the most common subword methods from oldest to most modern. The first and historically most influential
is Byte Pair Encoding (BPE), which introduced the idea of iteratively merging the most frequent token pairs.
BPE was later refined and adapted into methods like WordPiece and the Unigram Language Model, and it remains the foundation for many modern tokenizers,
including those used in our beloved GPT-style models. The next sections walk us through these methods in historical order.

As per the previous article on chunking, I will add a concrete example where possible.


### Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is an iterative algorithm that starts with a vocabulary of individual characters and then
repeatedly merges the most frequent adjacent pair of tokens. The process is simple to describe:

1. We start with a vocabulary of all characters in the training data.
2. Then we count how often each adjacent token pair appears in the corpus.
3. Then we merge the most frequent pair into a new token and add it to the vocabulary.
4. We repeat steps 2–3 until we reach a target vocabulary size or no pair is frequent enough.

For example, if the pair `"t"` + `"o"` appears very often (as in `"to"`, `"token"`, `"together"`), BPE will merge them
into `"to"`. Later, if `"to"` + `"ken"` is frequent, it might merge into `"token"`. Words that never get fully merged
are represented as a sequence of subwords, like `"token" → ["token"]`, `"tokens" → ["token", "s"]`, `"tokenized" → ["token", "ized"]`.

...**THAT'S EASY**

The main strengths of BPE are that it helps uncover and handle rare and unseen words by splitting them into known subwords.
It also keeps the vocabulary fairly compact by prioritizing frequent patterns. Last but not least, it preserves some morphological structure,
since common prefixes and suffixes often become their own tokens.

For a bit of history, BPE was introduced by Philip Gage in an article titled "A New Algorithm for Data Compression,"
published in the C Users Journal in February 1994. The concept was then revisited by Rico Sennrich, Barry Haddow, and
Alexandra Birch in their 2015 paper, "Neural Machine Translation of Rare Words with Subword Units" (if you have plenty of time this coming weekend: [https://arxiv.org/pdf/1508.07909](https://arxiv.org/pdf/1508.07909)).

BPE appeared first in the context of neural machine translation and later became the default tokenizer for models like
GPT-2 and GPT-3. While modern variants have refined the details, the core idea remains the same: build a vocabulary by
repeatedly merging the most frequent token pairs.

#### BPE Tokenization Example (Simplified)

```python
from collections import defaultdict
import re

# ============================================================
# EXAMPLE : Investment Management Corpus
# ============================================================

corpus = """
investment investment investment investment
investor investor investor
investing investing
invested
reinvestment

investment management
investment committee
investment strategy
investment portfolio
investment process
investment performance
investment opportunity
investment thesis
"""

# ============================================================
# Step 1 - Build Initial Vocabulary
# ============================================================

def build_vocab(text):

    words = re.findall(r'\w+', text.lower())

    vocab = defaultdict(int)

    for word in words:
        chars = tuple(list(word) + ['</w>'])
        vocab[chars] += 1

    return vocab


# ============================================================
# Step 2 - Count Pair Frequencies
# ============================================================

def get_pair_stats(vocab):

    pairs = defaultdict(int)

    for word, freq in vocab.items():

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq

    return pairs


# ============================================================
# Step 3 - Merge Most Frequent Pair
# ============================================================

def merge_pair(vocab, pair):

    merged_token = ''.join(pair)

    new_vocab = {}

    for word, freq in vocab.items():

        new_word = []

        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                new_word.append(merged_token)
                i += 2

            else:
                new_word.append(word[i])
                i += 1

        new_vocab[tuple(new_word)] = freq

    return new_vocab


# ============================================================
# Build Vocabulary
# ============================================================

vocab = build_vocab(corpus)

print("INITIAL VOCABULARY")
print("=" * 60)

for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:20]:
    print(f"{''.join(word):20} freq={freq}")

# ============================================================
# Run BPE
# ============================================================

num_merges = 50

print("\n")
print("BPE TRAINING")
print("=" * 60)

merges = []

for step in range(num_merges):

    pair_stats = get_pair_stats(vocab)

    best_pair = max(pair_stats, key=pair_stats.get)

    merged = ''.join(best_pair)

    print(
        f"Merge {step+1:2d}: "
        f"{best_pair[0]} + {best_pair[1]}"
        f" -> {merged}"
        f" ({pair_stats[best_pair]} occurrences)"
    )

    merges.append(best_pair)

    vocab = merge_pair(vocab, best_pair)

# ============================================================
# Tokenize New Words Using Learned Merges
# ============================================================

def tokenize(word, merges):

    tokens = list(word) + ['</w>']

    for merge in merges:

        merged_token = ''.join(merge)

        i = 0
        new_tokens = []

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == merge[0]
                and tokens[i + 1] == merge[1]
            ):
                new_tokens.append(merged_token)
                i += 2

            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return [t for t in tokens if t != '</w>']


# ============================================================
# Results
# ============================================================

print("\n")
print("TOKENIZATION RESULTS")
print("=" * 60)

test_words = [
    "investor",
    "investment",
    "investing",
    "invested",
    "reinvestment",
    "management"
]

for word in test_words:

    tokens = tokenize(word, merges)

    print(f"{word:30} -> {' + '.join(tokens)}")
```

#### After (BPE Tokenization):

```text
INITIAL VOCABULARY
============================================================
investment</w>       freq=12
investor</w>         freq=3
investing</w>        freq=2
invested</w>         freq=1
reinvestment</w>     freq=1
management</w>       freq=1
committee</w>        freq=1
strategy</w>         freq=1
portfolio</w>        freq=1
process</w>          freq=1
performance</w>      freq=1
opportunity</w>      freq=1
thesis</w>           freq=1


BPE TRAINING
============================================================
Merge  1: i + n -> in (21 occurrences)
Merge  2: e + s -> es (21 occurrences)
Merge  3: in + v -> inv (19 occurrences)
Merge  4: inv + es -> inves (19 occurrences)
Merge  5: inves + t -> invest (19 occurrences)
Merge  6: m + e -> me (14 occurrences)
Merge  7: me + n -> men (14 occurrences)
Merge  8: men + t -> ment (14 occurrences)
Merge  9: ment + </w> -> ment</w> (14 occurrences)
Merge 10: invest + ment</w> -> investment</w> (13 occurrences)
Merge 11: o + r -> or (6 occurrences)
Merge 12: invest + or -> investor (3 occurrences)
Merge 13: investor + </w> -> investor</w> (3 occurrences)
Merge 14: invest + in -> investin (2 occurrences)
Merge 15: investin + g -> investing (2 occurrences)
Merge 16: investing + </w> -> investing</w> (2 occurrences)
Merge 17: m + a -> ma (2 occurrences)
Merge 18: ma + n -> man (2 occurrences)
Merge 19: i + t -> it (2 occurrences)
Merge 20: t + e -> te (2 occurrences)
Merge 21: e + </w> -> e</w> (2 occurrences)
Merge 22: y + </w> -> y</w> (2 occurrences)
Merge 23: p + or -> por (2 occurrences)
Merge 24: por + t -> port (2 occurrences)
Merge 25: s + </w> -> s</w> (2 occurrences)
Merge 26: invest + e -> investe (1 occurrences)
Merge 27: investe + d -> invested (1 occurrences)
Merge 28: invested + </w> -> invested</w> (1 occurrences)
Merge 29: r + e -> re (1 occurrences)
Merge 30: re + investment</w> -> reinvestment</w> (1 occurrences)
Merge 31: man + a -> mana (1 occurrences)
Merge 32: mana + g -> manag (1 occurrences)
Merge 33: manag + e -> manage (1 occurrences)
Merge 34: manage + ment</w> -> management</w> (1 occurrences)
Merge 35: c + o -> co (1 occurrences)
Merge 36: co + m -> com (1 occurrences)
Merge 37: com + m -> comm (1 occurrences)
Merge 38: comm + it -> commit (1 occurrences)
Merge 39: commit + te -> committe (1 occurrences)
Merge 40: committe + e</w> -> committee</w> (1 occurrences)
Merge 41: s + t -> st (1 occurrences)
Merge 42: st + r -> str (1 occurrences)
Merge 43: str + a -> stra (1 occurrences)
Merge 44: stra + te -> strate (1 occurrences)
Merge 45: strate + g -> strateg (1 occurrences)
Merge 46: strateg + y</w> -> strategy</w> (1 occurrences)
Merge 47: port + f -> portf (1 occurrences)
Merge 48: portf + o -> portfo (1 occurrences)
Merge 49: portfo + l -> portfol (1 occurrences)
Merge 50: portfol + i -> portfoli (1 occurrences)


TOKENIZATION RESULTS
============================================================
investor                       -> investor</w>
investment                     -> investment</w>
investing                      -> investing</w>
invested                       -> invested</w>
reinvestment                   -> reinvestment</w>
management                     -> management</w>
```

### WordPiece

WordPiece, introduced as part of the BERT model, is another subword method that is similar to BPE but uses
a different merging criterion. Instead of always merging the most frequent pair, WordPiece selects the pair that maximizes
a score based on how much it improves the overall likelihood of the training data. This allows WordPiece to favor merges
that are more semantically or linguistically useful, not just statistically frequent.

WordPiece also introduces a special convention for marking continued subwords: tokens that are not at the beginning
of a word are prefixed with `##`. For example:

* `"token"` → `["token"]`
* `"tokens"` → `["token", "##s"]`
* `"tokenization"` → `["token", "##ization"]`

This makes it very explicit which subwords are prefixes and which are suffixes, helping the model learn morphological
relationships more easily.

WordPiece is the default tokenizer for many transformer models out there. It tends to produce slightly more coherent splits than raw BPE,
and it is still a greedy, iterative merging algo with a slightly different objective function — in other words,
WordPiece optimizes for linguistic coherence while BPE optimizes purely for frequency.

In practice, WordPiece and BPE often produce similar tokenizations, but WordPiece's scoring criterion can lead to
different choices for ambiguous or rare words.

#### Python Code: WordPiece Tokenization (Simplified Simulation)

```python
from collections import defaultdict
import re
import math

corpus = """
investment investment investment investment
investor investor investor
investing investing
invested
reinvestment

investment management
investment committee
investment strategy
investment portfolio
investment process
investment performance
investment opportunity
investment thesis
"""

def build_vocab(text):
    words = re.findall(r'\w+', text.lower())
    vocab = defaultdict(int)
    for word in words:
        chars = tuple(list(word) + ['</w>'])
        vocab[chars] += 1
    return vocab

def get_pair_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs

def get_wordpiece_scores(vocab, pairs):
    """
    WordPiece uses a different scoring function.
    It scores each pair based on how much it increases the likelihood
    of the training data under the model.
    """
    # Count total tokens
    total_tokens = sum(len(word) for word in vocab.keys())
    
    scores = {}
    for pair, freq in pairs.items():
        # Get frequency of the merged token
        merged = ''.join(pair)
        merged_freq = 0
        for word, word_freq in vocab.items():
            word_str = ''.join(word)
            if merged in word_str:
                merged_freq += word_freq
        
        # WordPiece score: (frequency of pair) / (frequency of first token * frequency of second token)
        # This is the log-likelihood ratio
        first_freq = sum(1 for word in vocab.keys() if pair[0] in word)
        second_freq = sum(1 for word in vocab.keys() if pair[1] in word)
        
        if first_freq > 0 and second_freq > 0:
            # WordPiece maximizes the likelihood of the segmentation
            score = freq / (first_freq * second_freq)
            scores[pair] = score
    
    return scores

def merge_pair(vocab, pair):
    merged_token = ''.join(pair)
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab

vocab = build_vocab(corpus)

print("INITIAL VOCABULARY")
print("=" * 60)
for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:20]:
    print(f"{''.join(word):20} freq={freq}")

num_merges = 50
print("\n")
print("WORD PIECE TRAINING")
print("=" * 60)

merges = []

for step in range(num_merges):
    pairs = get_pair_stats(vocab)
    if not pairs:
        break
    
    # WordPiece: score pairs instead of just counting frequency
    scores = get_wordpiece_scores(vocab, pairs)
    best_pair = max(scores, key=scores.get)
    
    merged = ''.join(best_pair)
    print(
        f"Merge {step+1:2d}: "
        f"{best_pair[0]} + {best_pair[1]}"
        f" -> {merged}"
        f" (score: {scores[best_pair]:.4f})"
    )
    
    merges.append(best_pair)
    vocab = merge_pair(vocab, best_pair)

def tokenize(word, merges):
    tokens = list(word) + ['</w>']
    for merge in merges:
        merged_token = ''.join(merge)
        i = 0
        new_tokens = []
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == merge[0]
                and tokens[i + 1] == merge[1]
            ):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return [t for t in tokens if t != '</w>']

print("\n")
print("TOKENIZATION RESULTS")
print("=" * 60)

test_words = [
    "investor",
    "investment",
    "investing",
    "invested",
    "reinvestment",
    "management"
]

for word in test_words:
    tokens = tokenize(word, merges)
    print(f"{word:30} -> {' + '.join(tokens)}")
```

#### After (WordPiece Tokenization):

```text
INITIAL VOCABULARY
============================================================
investment</w>       freq=12
investor</w>         freq=3
investing</w>        freq=2
invested</w>         freq=1
reinvestment</w>     freq=1
management</w>       freq=1
committee</w>        freq=1
strategy</w>         freq=1
portfolio</w>        freq=1
process</w>          freq=1
performance</w>      freq=1
opportunity</w>      freq=1
thesis</w>           freq=1


WORD PIECE TRAINING
============================================================
Merge  1: n + v -> nv (score: 0.4750)
Merge  2: i + nv -> inv (score: 0.4222)
Merge  3: inv + e -> inve (score: 0.3455)
Merge  4: inve + s -> inves (score: 0.4750)
Merge  5: inves + t -> invest (score: 0.3455)
Merge  6: invest + m -> investm (score: 0.5200)
Merge  7: investm + e -> investme (score: 0.7222)
Merge  8: investme + n -> investmen (score: 1.0833)
Merge  9: investmen + t -> investment (score: 0.8125)
Merge 10: investment + </w> -> investment</w> (score: 0.5000)
Merge 11: u + n -> un (score: 0.2500)
Merge 12: n + g -> ng (score: 0.2222)
Merge 13: i + ng -> ing (score: 0.4000)
Merge 14: invest + ing -> investing (score: 0.6667)
Merge 15: a + n -> an (score: 0.3333)
Merge 16: m + an -> man (score: 0.3333)
Merge 17: invest + o -> investo (score: 0.2500)
Merge 18: investo + r -> investor (score: 0.4286)
Merge 19: investor + </w> -> investor</w> (score: 0.2727)
Merge 20: man + a -> mana (score: 0.2500)
Merge 21: mana + g -> manag (score: 0.5000)
Merge 22: g + y -> gy (score: 0.5000)
Merge 23: man + c -> manc (score: 0.3333)
Merge 24: m + m -> mm (score: 0.2500)
Merge 25: mm + i -> mmi (score: 0.2500)
Merge 26: l + i -> li (score: 0.3333)
Merge 27: un + i -> uni (score: 0.5000)
Merge 28: s + i -> si (score: 0.3333)
Merge 29: si + s -> sis (score: 0.3333)
Merge 30: s + s -> ss (score: 0.2500)
Merge 31: investing + </w> -> investing</w> (score: 0.2000)
Merge 32: o + mmi -> ommi (score: 0.2000)
Merge 33: c + ommi -> commi (score: 0.5000)
Merge 34: f + o -> fo (score: 0.2500)
Merge 35: fo + li -> foli (score: 0.5000)
Merge 36: foli + o -> folio (score: 0.3333)
Merge 37: o + c -> oc (score: 0.3333)
Merge 38: p + o -> po (score: 0.2500)
Merge 39: o + p -> op (score: 0.3333)
Merge 40: op + po -> oppo (score: 0.5000)
Merge 41: n + t -> nt (score: 0.1667)
Merge 42: commi + t -> commit (score: 0.2000)
Merge 43: commit + t -> committ (score: 0.2000)
Merge 44: s + t -> st (score: 0.2500)
Merge 45: a + t -> at (score: 0.2500)
Merge 46: t + folio -> tfolio (score: 0.3333)
Merge 47: t + uni -> tuni (score: 0.5000)
Merge 48: tuni + t -> tunit (score: 0.5000)
Merge 49: tunit + y -> tunity (score: 1.0000)
Merge 50: t + h -> th (score: 1.0000)


TOKENIZATION RESULTS
============================================================
investor                       -> investor</w>
investment                     -> investment</w>
investing                      -> investing</w>
invested                       -> invest + e + d
reinvestment                   -> r + e + investment</w>
management                     -> manag + e + m + e + nt
```

### Unigram Language Model

The third tokenization approach I will bring now is the Unigram LM. The Unigram Language Model was introduced
alongside SentencePiece, and it takes a different approach from BPE and WordPiece.
Instead of iteratively merging token pairs, ULM starts with a large vocabulary which will often include all characters and common substrings,
then iteratively removes tokens that decrease the likelihood of the training data under a unigram model (if you are not
familiar with unigram model: it estimates the probability of each subword token based on how often it appears in the training data — that's it!).

The process can be summarized as:

1. Start with a large initial vocabulary (e.g., all characters plus frequent substrings).
2. Fit a unigram language model, where each token is assumed to be independent of the others.
3. Evaluate which tokens, if removed, would have the smallest impact on the model likelihood.
4. Remove those tokens and repeat until the vocabulary reaches the target size.

Once the vocabulary is trained, tokenization is done by finding a high-probability segmentation of the text, often using
beam search (if you wish to know more about beam search, I am sharing the following article that does a great job explaining:
[A Call for Clarity in Beam Search: How It Works and When It Stops](https://arxiv.org/pdf/2204.05424v3)). This means that a word like `"tokenization"` might be segmented in different ways depending on which path
has the highest probability under the unigram model.

Now let's have a quick look at Unigram's advantages:

* It is not constrained to a fixed merging order like BPE or WordPiece.
* It can produce multiple valid segmentations for the same word, which can be more flexible.
* It often yields slightly better token distributions for certain datasets.

However, it is more complex to implement and train, and in practice, many models still prefer BPE or WordPiece for
their simplicity and speed. You will notice that in the following example I change our corpus, because Unigram needs significantly more data to learn meaningful patterns.

#### Python Code: Unigram Tokenization (Simplified Simulation)

```python
from collections import defaultdict
import re
import math

# Simpler corpus with clear patterns
corpus = """
low low low low low low low low low low
lower lower lower lower
lowest lowest lowest
high high high high high high high high
higher higher higher higher
highest highest highest
"""

def build_word_freq(text):
    words = re.findall(r'\w+', text.lower())
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    return word_freq

def build_initial_vocab(word_freq):
    vocab = set()
    for word in word_freq.keys():
        for char in word:
            vocab.add(char)
    
    # Add common substrings
    subword_counts = defaultdict(int)
    for word, freq in word_freq.items():
        for i in range(len(word)):
            for j in range(i+2, min(i+5, len(word)+1)):
                subword = word[i:j]
                subword_counts[subword] += freq
    
    for subword, freq in subword_counts.items():
        if freq >= 3:
            vocab.add(subword)
    
    return list(vocab)

def get_token_freqs(vocab, word_freq):
    token_freqs = defaultdict(int)
    vocab_set = set(vocab)
    
    for word, freq in word_freq.items():
        # Greedy longest match
        tokens = []
        i = 0
        while i < len(word):
            found = False
            for j in range(len(word), i, -1):
                if word[i:j] in vocab_set:
                    tokens.append(word[i:j])
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(word[i])
                i += 1
        
        for token in tokens:
            token_freqs[token] += freq
    
    return token_freqs

def compute_loss(word_freq, vocab):
    token_freqs = get_token_freqs(vocab, word_freq)
    total_freq = sum(token_freqs.values())
    
    loss = 0
    for word, freq in word_freq.items():
        tokens = []
        i = 0
        vocab_set = set(vocab)
        while i < len(word):
            found = False
            for j in range(len(word), i, -1):
                if word[i:j] in vocab_set:
                    tokens.append(word[i:j])
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(word[i])
                i += 1
        
        word_loss = 0
        for token in tokens:
            prob = token_freqs.get(token, 1) / total_freq
            word_loss += -math.log(prob)
        loss += freq * word_loss
    
    return loss

def unigram_training(word_freq, initial_vocab, target_size):
    vocab = set(initial_vocab)
    chars = set()
    for word in word_freq.keys():
        for char in word:
            chars.add(char)
    
    print(f"Initial vocabulary: {len(vocab)} tokens")
    
    while len(vocab) > target_size:
        current_loss = compute_loss(word_freq, vocab)
        
        impact = {}
        for token in list(vocab):
            if token in chars:
                continue
            test_vocab = vocab - {token}
            test_loss = compute_loss(word_freq, test_vocab)
            impact[token] = test_loss - current_loss
        
        sorted_tokens = sorted(impact.items(), key=lambda x: x[1])
        num_to_remove = max(1, int(len(vocab) * 0.15))
        
        for token, _ in sorted_tokens[:num_to_remove]:
            vocab.remove(token)
        
        print(f"Vocab: {len(vocab)}, Loss: {current_loss:.2f}")
    
    return list(vocab)

def tokenize_word(word, vocab):
    tokens = []
    i = 0
    vocab_set = set(vocab)
    while i < len(word):
        found = False
        for j in range(len(word), i, -1):
            if word[i:j] in vocab_set:
                tokens.append(word[i:j])
                i = j
                found = True
                break
        if not found:
            tokens.append(word[i])
            i += 1
    return tokens

word_freq = build_word_freq(corpus)

print("CORPUS:")
for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
    print(f"  {word}: {freq}")

initial_vocab = build_initial_vocab(word_freq)
print(f"\nInitial vocabulary: {len(initial_vocab)} tokens")

final_vocab = unigram_training(word_freq, initial_vocab, 20)

print(f"\nFINAL VOCABULARY ({len(final_vocab)} tokens):")
print(sorted(final_vocab))

print("\nTOKENIZATION RESULTS")
print("=" * 50)

test_words = ["low", "lower", "lowest", "high", "higher", "highest"]

for word in test_words:
    tokens = tokenize_word(word, final_vocab)
    print(f"{word:10} -> {' + '.join(tokens)}")

print("\nTOKEN FREQUENCIES")
print("=" * 50)
token_freqs = get_token_freqs(final_vocab, word_freq)
for token, freq in sorted(token_freqs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {token}: {freq}")
```

#### After (Unigram Tokenization):

```text
CORPUS:
  low: 10
  high: 8
  lower: 4
  higher: 4
  lowest: 3
  highest: 3

Initial vocabulary: 39 tokens
Initial vocabulary: 39 tokens
Vocab: 34, Loss: 81.17
Vocab: 29, Loss: 59.95
Vocab: 25, Loss: 59.95
Vocab: 22, Loss: 59.95
Vocab: 19, Loss: 59.95

FINAL VOCABULARY (19 tokens):
['e', 'er', 'est', 'g', 'h', 'he', 'her', 'high', 'i', 'ig', 'l', 'low', 'o', 'r', 's', 'st', 't', 'w', 'wes']

TOKENIZATION RESULTS
==================================================
low        -> low
lower      -> low + er
lowest     -> low + est
high       -> high
higher     -> high + er
highest    -> high + est

TOKEN FREQUENCIES
==================================================
  low: 17
  high: 15
  er: 8
  est: 6

```

### SentencePiece (framework for BPE and Unigram)

SentencePiece is not a new subword algorithm in the same sense as BPE or WordPiece; rather, it is an open-source,
unsupervised tokenizer and detokenizer framework developed by Google that can implement both BPE and the Unigram language
model (**great**, isn't it?). It was designed to be **language-agnostic** and to handle the entire tokenization pipeline end-to-end, including
whitespace management, vocabulary training, encoding, and decoding. This is why many modern LLMs out there use SentencePiece for
their foundation models.

#### Python Code: SentencePiece Tokenization (Simplified Simulation)

```python
import sentencepiece as spm
import os

# Define training options
options = dict(
    input="./files/luckin_coffee_transcript_cleaned.txt",             
    input_format="text",
    model_prefix="custom500",   
    model_type="unigram",       
    vocab_size=2268,  # The vocab size depends on the training data, in my case here i have 443 sentences and 30,575, it is the final size after EM pruning      
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,
    max_sentence_length=4192,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=0.99995,
    byte_fallback=True,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
    num_threads=os.cpu_count(),
)

# Train the SentencePiece model
spm.SentencePieceTrainer.train(**options)

# Load and use the trained model
sp = spm.SentencePieceProcessor(model_file='custom500.model')

# Encode a sentence
encoded = sp.encode("In the first quarter, total net revenues grew 35% year-over-year")
print("Encoded:", encoded)

decoded = sp.decode(encoded)
print("Decoded:", decoded)
```

#### After (SentencePiece Tokenization):

```text
Encoded: [312, 270, 332, 305, 260, 267, 350, 331, 361, 753, 647, 259, 284, 288, 291, 304, 264, 338, 264, 328]
Decoded: In the first quarter, total net revenues grew 35% year-over-year
```

Key features that set SentencePiece apart from traditional tokenizers:

* **Whitespace-agnostic (lossless)**: As we discussed at the start of this article, traditional tokenizers assume words are separated by spaces and often discard or
  ignore them. SentencePiece treats spaces as standard characters, explicitly replacing them with a specific meta-symbol
  (like `_` or `▁`). This makes it perfectly lossless—you can turn tokens back into the exact original text without losing
  spacing information.

* **No pre-tokenization required**: SentencePiece learns to tokenize directly from raw, unsegmented text.
  This is particularly useful for languages with no explicit word boundaries, like Chinese, Japanese, or Thai, where
  traditional word-level tokenizers struggle.

* **Two core algorithms**: It supports training via Byte-Pair Encoding (BPE) and the Unigram language model, giving you
  flexibility to choose the algorithm that best fits your data and task.

* **Fixed vocabulary size**: Unlike algorithms that conceptually assume an infinite vocabulary, SentencePiece is trained
  so that your target model has a predetermined vocabulary size (e.g., 32,000 tokens). This makes it easier to control
  model size and inference cost.

* **End-to-end system**: SentencePiece creates an end-to-end pipeline where you can train, encode, and decode using a
  single tool. You don't need to stitch together separate components for tokenization and detokenization, which simplifies
  the workflow and reduces the risk of inconsistencies.

Because of these properties, SentencePiece is especially powerful in multilingual settings and in scenarios where you
want a single, consistent tokenization pipeline across many languages. It is widely used in models like ALBERT, T5, and
some versions of other transformer families.

In the literature, SentencePiece is often mentioned alongside BPE and WordPiece because many modern models use a
SentencePiece-trained BPE or Unigram tokenizer. From a historical perspective, it comes after the original BPE and
WordPiece algorithms, as a more general and flexible implementation framework that unifies tokenization and detokenization
in a language-agnostic way.

## Conclusion

Tokenization is one of those topics that often stays in the background, yet it has a very real impact on how a RAG
system behaves in practice. As we have seen, the way text is split into tokens affects not only vocabulary size and
model efficiency, but also retrieval quality, generation cost, and the amount of context the model can actually preserve.

What I find interesting is that there is no single tokenization method that wins everywhere. Word-level tokenization
is too rigid, character-level tokenization is too granular, and subword tokenization became the practical middle ground
that modern models rely on. Within that family, BPE, WordPiece, Unigram, and SentencePiece each reflect a slightly different
way of solving the same problem.

In the end, tokenization is not just a preprocessing step. It is part of the foundation of the system. If the foundation
is weak, everything built on top of it will inherit that weakness. This is why I think it is worth understanding properly, especially when building RAG pipelines that need to be both accurate and cost-efficient.
