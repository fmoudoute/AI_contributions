# The Making of RAG: Generation (Part 5)

## Introduction

This is the fifth article in my *Making of RAG* series, and we have finally reached the step everyone thinks about when
they hear "RAG": the moment the LLM writes the answer. In the previous articles, we chunked the text (Part 1), tokenized
it (Part 2), embedded it (Part 3) and built a retrieval pipeline that we measured on 12 real questions (Part 4). All of
that work had one purpose: putting the right chunks in front of the language model.

If retrieval was the concierge of our building, generation is the conversation with the visitor. The concierge found the
right rooms; now someone has to read what is inside and give a clear, honest answer, without making things up.

At the end of Part 4, I promised to cover how we turn the retrieved chunks into a prompt, how to make the LLM answer
*only* from them (and cite them), and what happens when retrieval gives it the wrong context. We'll do all of that, on
the same Luckin Coffee Q1 2026 earnings transcript.

And this time, there is a twist: **everything runs locally, for free.** I used [Ollama](https://ollama.com) with
`qwen2.5:7b`, a 7-billion-parameter open model that runs on a laptop. No API key, no bill, and nothing to install in
Python: we talk to Ollama with the standard library. As always, every output you'll see is real, and the scripts are at
the end.

*(As always, if you find this helpful, a thumbs-up helps others find it and keeps me writing!)*

---

## Generation, or the last mile

Generation is the step where the retrieved chunks and the user's question are combined into one prompt, and the LLM
writes the answer. It's the only step the user actually sees, and it has the same impacts as the other steps:

### 1 - Accuracy

A perfect retrieval can still produce a wrong answer if the LLM misreads the context, mixes two numbers, or ignores the
chunks and answers from its own memory. And as we'll see, an answer can be completely faithful to the chunks... and still
be wrong.

### 2 - Cost

The retrieved chunks are pasted into the prompt, so every extra chunk is extra tokens on every question. With a paid API,
that's money (Part 2); with a local model, it's time. Either way, K (the number of chunks) is a cost decision.

### 3 - Trust

This one is new in the series. A RAG system is only useful if people can trust its answers. That means two things: the
model must say "I don't know" when the answer is not in the documents, and it must show *where* each fact comes from,
so a human can check it.

---

## Talking to a local LLM

Ollama runs a small web server on your machine (`localhost:11434`). Sending it a prompt is a simple HTTP request, so the
Python standard library is enough:

```python
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def ask_llm(prompt, model=MODEL, num_ctx=8192):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # temperature 0 + a fixed seed: the same prompt gives the same answer
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx},
    }
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    return {
        "answer": result["response"].strip(),
        "prompt_tokens": result["prompt_eval_count"],
        "answer_tokens": result["eval_count"],
        "seconds": result["total_duration"] / 1e9,
        "load_seconds": result.get("load_duration", 0) / 1e9,  # time spent loading the model into memory
    }
```

Two details matter here. First, `temperature: 0` and a fixed `seed` make the answers reproducible: I ran every
experiment of this article several times, and got the same answers each time. Second, `num_ctx` is the size of the
model's context window in tokens. Keep an eye on it; it will come back to bite us at the end of the article.

---

## Without retrieval: what the LLM knows on its own

Before adding any context, let's ask the model directly:

#### After:

```text
Q: How many stores does Luckin Coffee have?
A: As of the latest available data, Luckin Coffee operates thousands of stores globally. However, the exact number can
fluctuate due to various factors such as store openings, closures, and acquisitions. For the most accurate and
up-to-date information, you can refer to Luckin Coffee's official announcements or their latest financial reports.

Q: How many people work for Luckin Coffee?
A: As of my last update in October 2023, the exact number of employees at Luckin Coffee is not publicly disclosed, and it
can vary over time. [...]
```

I have to say, I expected a hallucination here, and Qwen surprised me: it stayed vague rather than inventing a number.
But look at what we got: "thousands of stores" and "As of my last update in October 2023". The model's knowledge is
frozen at its training date, and it knows nothing about a call from April 2026. **This is the whole reason RAG exists**:
the LLM brings the language skills, and our documents bring the facts.

---

## The augmented prompt

Now let's give it the context. The prompt is just text, and I think it's worth looking at it as text, because that's
all the model receives:

```python
PROMPT_TEMPLATE = """You are an assistant that answers questions about the Luckin Coffee Q1 2026 earnings call.

Rules:
- Answer ONLY with information from the context below.
- After each fact, cite the chunk it comes from, like [chunk 12].
- If the context does not contain the answer, say exactly: "I don't know based on the provided context."

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question, chunk_ids):
    context = "\n\n".join(f"[chunk {i}] {r.chunks[i]}" for i in chunk_ids)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def rag(question, k=3, model=MODEL):
    chunk_ids = retrieve(question, k)  # hybrid search + re-rank, from Part 4
    result = ask_llm(build_prompt(question, chunk_ids), model=model)
    result["chunk_ids"] = chunk_ids
    return result
```

Three rules, and each one has a job: the first one limits the model to the context (accuracy), the second one makes
every fact checkable (trust), and the third one gives the model an exit when the answer isn't there (no guessing). Notice
also that every chunk is labelled with its ID, so the model can cite it.

Let's ask the question that Part 3 and Part 4 found the hardest, "How many people work for Luckin?":

#### After:

```text
Retrieved chunks: [52, 51, 24]
Prompt: 206 words -> 465 tokens for the LLM
A: According to chunk 52, Luckin has 190,000 team members. [chunk 52]
(465 prompt tokens + 28 answer tokens, 3.2s)
```

Correct, short, and cited. Notice the count: 206 words became 465 tokens. Our transcript contains Chinese paragraphs,
and as we saw in Part 2 and Part 3, Chinese characters cost a lot more tokens than English words. Also notice that the
prompt is 465 tokens and the answer only 28: **in RAG, the input is usually much bigger than the output**, which is why
the number of chunks matters so much for cost.

---

## Trust, but verify: checking the citations

Asking for citations is good. Checking them is better. An LLM can cite a chunk that doesn't contain what it claims, so I
wrote a small "grounding check": for every number in the answer, verify that it really appears in one of the chunks the
answer cites.

```python
import re

NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def check_grounding(question, answer):
    cited = sorted({int(c) for c in re.findall(r"chunk (\d+)", answer)})
    text = re.sub(r"\[?chunks? [\d, chunk]+\]?", " ", answer)  # remove the citations themselves
    numbers = sorted({n.rstrip(".,") for n in NUMBER.findall(text)} - set(NUMBER.findall(question)))
    cited_text = " ".join(r.chunks[c] for c in cited if c < len(r.chunks))
    return {"cited": cited, "grounded": [n for n in numbers if n in cited_text],
            "not_grounded": [n for n in numbers if n not in cited_text]}
```

I tested it on the real answer above, and on an answer I made up myself:

#### After:

```text
A: According to chunk 52, Luckin has 190,000 team members. [chunk 52]
   {'cited': [52], 'grounded': ['190,000'], 'not_grounded': []}
A: Luckin operates 35,000 stores [chunk 20].
   {'cited': [20], 'grounded': [], 'not_grounded': ['35,000']}
```

The fake "35,000 stores" is caught immediately: chunk 20 contains no such number. It's simple (it only checks numbers,
not sentences), but in a financial use case, numbers are exactly what we can't afford to get wrong. And it costs nothing:
no extra LLM call, just string matching. In Part 6, this check will catch a smaller model citing the wrong chunks.

---

## Faithful... but wrong

Now the question that has followed us since Part 4: "How many shops does the company operate?"

#### After:

```text
Retrieved chunks: [20, 25, 58]
A: The company operates a total of 33,419 stores, which includes 21,713 self-operated stores and 11,706 partnership stores. [chunk 20]
Chunk 20 says: ...In China, we added 2,531 net new stores during the quarter, bringing our total store count to 33,419. This included 21,7...
```

This is, for me, the most important result of this article. Every number in the answer is in chunk 20, and the citation
is correct; our grounding check would say "all good". **And yet the answer is wrong.** The chunk says "*In China*, ...
our total store count to 33,419". The model dropped the two words "In China" and turned a China number into the
company's total. The real total is 33,596 (with the 177 overseas stores), and it's in chunk 19, which retrieval didn't
bring back.

So, two lessons:

* **Faithful is not the same as correct.** An answer can be perfectly grounded in the context and still be wrong, because
  the context was incomplete and the model filled the gap with a confident reading of what it had.
* **Generation inherits retrieval's mistakes.** Remember that in Part 4, the re-ranker picked chunk 20 first for this
  question. The LLM didn't fix it; it turned it into a fluent, cited, wrong answer.

---

## How many chunks should we give to the LLM?

So, should we just give more chunks? Let's measure it. I ran the 12 questions of Part 4 with K = 1, 3 and 10, and
checked whether each answer contains the expected fact:

#### After:

```text
K= 1: correct  9/12, avg prompt   251 tokens, avg 1.7s
       x How many shops does the company operate? -> The company operates a total of 33,419 stores. [chunk 20]
       x What was the non-GAAP net profit? -> I don't know based on the provided context.
       x Will shareholders receive dividends? -> I don't know based on the provided context.
K= 3: correct 11/12, avg prompt   499 tokens, avg 3.0s
       x How many shops does the company operate? -> The company operates a total of 33,419 stores, which includes 21,713 s
K=10: correct 12/12, avg prompt  1313 tokens, avg 6.4s
```

More chunks means more correct answers, but also **5 times more tokens** and almost 4 times more time per question
between K=1 and K=10. And the K=1 failures are worth a closer look, because they are three different problems:

* **"How many shops?"**: the retrieval problem we just discussed.
* **"What was the non-GAAP net profit?"**: the best chunk ends with *"...On a non-GAAP basis, net profit"*, and the number
  is at the start of the *next* chunk. The chunk boundary cut the answer in two. That's Part 1 coming back: the overlap
  wasn't enough here.
* **"Will shareholders receive dividends?"**: the chunk *did* contain "we are not able to pay dividends in the short
  term", and the model still answered "I don't know". This is a generation failure: the model was too cautious. (The
  chunk starts with a Chinese paragraph, which may have confused it, but that's only my guess.)

With K=10, all 12 questions are answered correctly, including "How many shops?", which now gets the right 33,596 from
chunk 19. My take: **K=3 is a good default** for this kind of factual question, and if your users ask broader questions
("summarize the CFO's comments"), measure a larger K on your own test set.

---

## Knowing when to say "I don't know"

A RAG system that always answers is dangerous. So I asked four questions that the transcript cannot answer:

#### After:

```text
Q: What is the revenue guidance for the second quarter of 2026?
A: I don't know based on the provided context.
Q: How many stores does Luckin have in Japan?
A: I don't know based on the provided context.
Q: Who is the chairman of the board?
A: I don't know based on the provided context.
Q: What is the price of a latte?
A: I don't know based on the provided context.
```

Four out of four. The Japan question is the one I like the most: the top retrieved chunk talks about international
expansion and "our total overseas store count to 177", which is a perfect trap for a model that wants to be helpful
("Luckin has 177 stores abroad, so maybe some in Japan..."). Qwen didn't fall for it. This is the
third rule of the prompt doing its job, and in my experience, it's worth spelling out the exact sentence you want: it
makes "I don't know" easy to detect in code (we'll use that in Part 6).

---

## The context window: silent truncation, again

Do you remember Part 3, where the embedding model silently ignored everything after 256 tokens? The same thing happens
with LLMs. Every model has a context window, and with Ollama you set it with `num_ctx`. If the prompt is longer, it is
cut, without any error.

To show it, I retrieved K=30 chunks for "How many stores are in Malaysia?" (the answer is in chunk 25, ranked first, so
it is at the very top of the context) and sent the same prompt with two context sizes:

#### After:

```text
Chunk 25 (the answer) is at position 0 of 30
num_ctx=2048: the LLM read 1026 prompt tokens
   A: The provided text does not contain any information about the number of stores in Malaysia. The text focuses on Luckin Coffee's store expansion and performance in China, and does not mention any operations in Malaysia.
num_ctx=8192: the LLM read 3287 prompt tokens
   A: According to chunk 25, there are 83 franchised stores in Malaysia. [chunk 25]
```

With a 2,048-token window, the model read only **1,026 of the 3,287 tokens**. It still answered the question (which is
at the end of the prompt), but it never saw chunk 25 at the top. And look at the answer: no citation, no "I don't know
based on the provided context". The rules were at the top of the prompt too, so **they were most likely cut as well.** The model answered from a truncated context, without its
instructions, and confidently said the transcript doesn't mention Malaysia.

That's the third time in this series that silent truncation has bitten us (chunks in Part 1, embeddings in Part 3, and now
the LLM). So my advice is always the same: **know the limit of every model in your pipeline, set it explicitly, and log
the token counts.** Ollama gives them to you for free (`prompt_eval_count`); compare it with the length of the prompt you
sent. The default context size has changed between Ollama versions, so never rely on it.

---

## Practical considerations

1. **Write the rules in the prompt, and keep them short.** Answer only from the context, cite the sources, and say a
   fixed sentence when the answer isn't there.
2. **Label every chunk with an ID**, so the model can cite it, and so you can check it.
3. **Check the citations in code.** A simple number check catches the most dangerous errors for free.
4. **Choose K with your test set.** More chunks means more recall but more tokens, more time, and more noise.
5. **Set the context window explicitly** and compare the token count the model reports with what you sent.
6. **Use temperature 0 for factual Q&A.** You want the same answer every time, especially when you evaluate.
7. **Remember that faithful is not correct.** A cited answer can still be wrong if retrieval brought the wrong chunk.

---

## Conclusion

Generation is the last mile of RAG, and it's where everything we built before becomes visible to the user. What I find
striking in these experiments is that the LLM was rarely the weak link: a 7B model running on a laptop answered
correctly, cited its sources, and refused the four questions it couldn't answer. Most of the failures came from earlier
steps: a chunk boundary (Part 1), a missed chunk (Part 4), a context that was too small.

But the "33,419 stores" answer shows why we can't stop here. It was fluent, cited, grounded... and wrong. Looking at a
few answers one by one isn't enough anymore. We need to evaluate the **whole pipeline**, end to end, and separate the
retrieval failures from the generation failures.

That's exactly what the next article is about: **evaluation**. We'll measure correctness, faithfulness and refusals,
compare three local models, and see whether we can trust an LLM to grade another LLM.

If you've found this series helpful, I'd really appreciate a follow. It helps me know that people are reading and
encourages me to keep writing.

---

## Run it yourself

The scripts are in the `scripts/` folder of my project and should be run from that folder:

1. Install [Ollama](https://ollama.com), then run `ollama pull qwen2.5:7b` (about 4.7 GB).
2. Keep the Part 4 files (`retrieval_eval.py`, `embedding_pretrained.py`) in the same folder, with their requirements
   (`pip install numpy onnxruntime tokenizers huggingface_hub`).
3. Run `python generation_demo.py`. It reproduces every output of this article, using `rag_generation.py`.

With temperature 0 and a fixed seed, you should get the same answers as me with the same model version; the timings will
of course depend on your machine.

## References
- P. Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
