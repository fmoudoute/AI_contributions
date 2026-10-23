# The Making of RAG: Evaluation (Part 6)

## Introduction

This is the sixth article in my *Making of RAG* series, and the last one of the pipeline. Over five articles, we built
a complete RAG system, piece by piece: chunking (Part 1), tokenization (Part 2), embeddings (Part 3), retrieval (Part 4)
and generation (Part 5), all on the same Luckin Coffee Q1 2026 earnings transcript, and all running on a laptop.

So, is it good?

That's a harder question than it looks. In Part 4, we measured retrieval: is the right chunk in the top 3? In Part 5, we
checked whether the answer contains the expected number. But Part 5 also gave us an answer that was fluent, cited,
grounded in its chunk... and wrong ("33,419 stores", which is the China count, not the total). A RAG system can fail in
many ways, and a single number can't tell us which one.

To finish my building metaphor (I promised, it's the last time :), evaluation is the **inspection** before people move
in. You don't just look at the facade; you check the foundation, the materials, the plumbing and the electricity,
separately, because each one can fail on its own.

In this article, we will evaluate the whole pipeline end to end: correctness, refusals, faithfulness of the citations,
and the question everyone asks today: **can we trust an LLM to grade another LLM?** As always, all the numbers come from
code I ran myself, with free local models through [Ollama](https://ollama.com), and the scripts are at the end.

*(As always, if you find this helpful, a thumbs-up helps others find it and keeps me writing!)*

---

## What exactly are we measuring?

When a RAG answer is wrong, there are only two places where things can go wrong:

* **Retrieval**: the right chunk was not given to the LLM. The model can't answer what it never saw.
* **Generation**: the right chunk was there, and the model still answered wrongly (misread it, mixed numbers, refused,
  or ignored the context).

This distinction is the most useful thing an evaluation can give you, because the fixes are completely different. A
retrieval failure is fixed with better chunking, hybrid search or a bigger K. A generation failure is fixed with a better
prompt or a better model. If you don't know which one you have, you will optimize the wrong step.

So, for every question, I measure:

1. **Retrieved**: is the evidence in the K chunks given to the LLM? (the Part 4 metric)
2. **Correct**: is the answer right? (we'll see that this one is surprisingly hard to measure)
3. **Refused**: for questions the transcript can't answer, did the model say "I don't know"?
4. **Unsupported numbers**: numbers in the answer that are *not* in the chunks it cites (the Part 5 grounding check).

---

## The test set

I extended the 12 questions of Part 4 with three things: a **reference answer** (what a human expert would answer), the
**key facts** the answer must contain, and four **unanswerable** questions.

```python
EVAL_SET = [
    # (question, evidence in the right chunk, reference answer, key facts: ALL groups must match, ANY alternative per group)
    ("How many shops does the company operate?", "33,596",
     "33,596 stores in total at the end of Q1 2026 (33,419 in China and 177 overseas).", [["33,596"]]),
    ("How much did S&M expenses increase?", "RMB732 million",
     "S&M expenses rose 48% year-over-year to RMB732 million.", [["48%"], ["732"]]),
    ("Will shareholders receive dividends?", "not able to pay dividends",
     "Not in the short term: for historical reasons the company cannot pay dividends, but it is evaluating alternatives.",
     [["not able to pay", "unable to pay", "cannot pay", "can't pay", "not pay dividends"]]),
    # ... 9 more answerable questions ...
    # Questions the transcript can NOT answer: the right behaviour is to say so
    ("What is the revenue guidance for the second quarter of 2026?", None, REFUSAL, None),
    ("How many stores does Luckin have in Japan?", None, REFUSAL, None),
    ("Who is the chairman of the board?", None, REFUSAL, None),
    ("What is the price of a latte?", None, REFUSAL, None),
]
```

Writing the reference answers took me longer than writing the code, and I think that's normal. **The test set is the
most valuable part of an evaluation**, and it's the part nobody can write for you: you need to know the documents.

---

## Two ways to grade an answer

### 1. String matching

The simplest grader: the answer is correct if it contains the key facts. For unanswerable questions, it's correct if it
says "I don't know".

```python
def is_refusal(answer):
    return "don't know" in answer.lower() or "do not know" in answer.lower()


def string_match(answer, keys):
    if keys is None:
        return is_refusal(answer)
    return all(any(alternative in answer for alternative in group) for group in keys)
```

It's fast, free and reproducible. But it's also blind: it can't tell "93 million customers" from "the context mentions 93
million, but I don't know". Keep that example in mind.

### 2. LLM-as-a-judge

The popular alternative is to ask another LLM to compare the answer with the reference answer. I used `llama3.1` (8B) as
the judge, a different model family from Qwen, our main model:

```python
JUDGE_MODEL = "llama3.1:latest"
JUDGE_PROMPT = """You are grading the answer of an assistant against a reference answer.

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Is the assistant answer correct and consistent with the reference answer? Minor extra details are fine,
but a wrong number, a wrong entity or a missing key fact is not.
Reply with one word: CORRECT or INCORRECT."""


def judge(question, reference, answer):
    verdict = g.ask_llm(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
                        model=JUDGE_MODEL)["answer"].upper()
    return verdict.startswith("CORRECT")
```

It understands meaning, so in theory it should be better than string matching. In theory.

---

## Round 1: did Part 4 actually help?

In Part 4, hybrid search + re-ranking beat semantic search on retrieval metrics (MRR 0.89 vs 0.73). But users don't see
MRR; they see answers. So here is the same model (`qwen2.5:7b`, K=3) with the two retrieval pipelines:

#### After:

```text
Part 3: semantic only, K=3         correct(match) 10/12  correct(judge) 10/12  final 10/12  refused 4/4  unsupported numbers  0  avg 3.7s
Part 4: hybrid + re-rank, K=3      correct(match) 11/12  correct(judge) 11/12  final 11/12  refused 4/4  unsupported numbers  0  avg 3.5s
```

(The "final" column is explained below; here, both graders agree on every answer. The times are per answer, without
the time Ollama spends loading a model into memory: since the judge is a different model, Ollama keeps switching
between the two, and my first run measured that switching more than the models themselves.)

Yes, the Part 4 pipeline gives one more correct answer, 11 vs 10. Smaller than the retrieval gap, and that's
an important lesson: **better retrieval metrics don't translate one-to-one into better answers**, because the LLM can
sometimes answer from an imperfect set of chunks, and sometimes fails even with a perfect one.

And the failures tell the story:

```text
[Part 3 / qwen] How many people work for Luckin?
    match=False judge=False final=False retrieved=False unsupported=[]
    -> "I don't know based on the provided context."
[Part 3 / qwen] How many customers buy every month?
    match=False judge=False final=False retrieved=False unsupported=[]
    -> "I don't know based on the provided context."
```

Both are retrieval failures (`retrieved=False`): semantic search didn't bring the right chunk, and Qwen correctly said
it didn't know. That's actually the *good* way to fail: no hallucination, just an honest "I don't know". These two
questions are exactly the ones where, in Part 4, semantic search ranked the right chunk 4th and 5th.

---

## Round 2: does the model size matter?

Now the same pipeline (hybrid + re-rank, K=3) with three local models: a tiny one, `LFM2.5` with 1.2 billion parameters
(730 MB), and two 7 to 8 billion parameter models, `qwen2.5:7b` and `llama3.1:8b` (about 4.7 GB each).

#### After:

```text
LFM2.5 1.2B                        correct(match)  8/12  correct(judge)  8/12  final  7/12  refused 4/4  unsupported numbers  9  avg 0.8s
qwen2.5 7B                         (same run as above)
llama3.1 8B                        correct(match) 10/12  correct(judge) 10/12  final 10/12  refused 4/4  unsupported numbers  1  avg 1.3s
```

(For Qwen, it's the "Part 4" line of Round 1: 11/12 correct, 0 unsupported numbers, 3.5s.)

The small model is the fastest (0.8s per answer), but it gets only 7 answers right, and it produced **9 unsupported numbers**.
Let's look at its answers, because they show every type of generation failure in one place:

```text
[Part 4 / LFM2.5 1.2B] What was the non-GAAP net profit?
    -> "The non-GAAP net profit was RMB506 million, with a net margin of 4.2%, compared to RMB524 million and 5.9% in the
       prior-year period. I don't know based on the provided context."
[Part 4 / LFM2.5 1.2B] How much cash does the company have?
    -> "The company has a total cash position of about RMB9.1 billion, [...] [chunk 12]  I don't know based on the provided "
[Part 4 / LFM2.5 1.2B] How much did S&M expenses increase?
    -> "S&M expenses rose 48% year over year to RMB732 million, according to the context [chunk 63]. I don't know based
       on the provided context."
```

* **Mixed-up numbers**: RMB506 million is the *GAAP* net profit. The non-GAAP figure (RMB686 million) was in the same
  context, a few words later. The model picked the wrong one.
* **Wrong citations**: RMB9.1 billion is in chunks 65 and 66, not chunk 12. RMB732 million is in chunk 62, not 63. The
  numbers are right, but the sources are wrong, and our grounding check from Part 5 caught both.
* **Broken instructions**: it appends "I don't know based on the provided context" after almost every answer, even
  correct ones. It clearly struggled to follow the three rules of the prompt at once.

That last point also exposes a flaw in *my* metric: the small model scores a perfect **4/4 on refusals**, but only
because it says "I don't know" to everything! A model that refuses every question would get a perfect refusal score.
**A metric that can be gamed by a bad model must always be read together with another one** (here, correctness).

So which model would I choose? For this use case, Qwen: the best correctness, zero unsupported numbers. But Llama is
almost 3 times faster on my laptop (1.3s vs 3.5s per answer) for one answer less, and that could be the right trade-off for a chat interface. That's the kind
of decision an evaluation lets you make with numbers instead of impressions.

---

## Who judges the judge?

In both rounds, the string match and the LLM judge give the same totals... except for the small model (8 vs 7 "final").
But the totals hide something. Across the 48 answerable answers of this article (4 runs × 12 questions), the two graders
disagreed **4 times**. So I read the four answers myself:

```text
[Part 4 / LFM2.5 1.2B] How many customers buy every month?
    match=True judge=False final=False
    -> "The context mentions that the average monthly transacting customers increased by 25% year-over-year to over
       93 million, but it does not specify the exact number for the current quarter. Therefore, I don't know [...]"
[Part 4 / LFM2.5 1.2B] Will shareholders receive dividends?
    match=False judge=True final=False
    -> "I don't know based on the provided context. [chunk 98]  I don't know based on the provided context."
[Part 4 / llama3.1 8B] How many shops does the company operate?
    match=False judge=True final=False
    -> 'According to [chunk 20], the company has a total store count of 33,419.'
[Part 4 / llama3.1 8B] Where does the company buy its coffee beans?
    match=True judge=False final=True
    -> 'According to [chunk 42], the company buys its coffee beans from Yunnan, procuring over 30,000 tons of coffee
       beans during the current harvest season.'
```

My verdicts, one by one:

1. **Customers**: the answer contains "93 million" but concludes "I don't know". **The judge is right**, the string match
   is fooled. This is exactly the blindness I mentioned above.
2. **Dividends**: the answer is just "I don't know", and the judge called it CORRECT. **The judge is wrong.**
3. **Shops**: "33,419" is the China count (our old friend from Parts 4 and 5), and the judge called it CORRECT, even
   though the reference answer explicitly says "33,596 stores in total (33,419 in China...)". **The judge is wrong.** And
   there's an uncomfortable detail: this answer was written by `llama3.1`... and graded by `llama3.1`. I can't prove it
   from one example, but self-preference (a model grading its own answers more kindly) is a known bias of LLM judges,
   and a good reason to never let a model judge itself.
4. **Coffee beans**: "from Yunnan, over 30,000 tons" is incomplete (the company also buys international beans), but it's
   not wrong. The judge was too strict. **I sided with the string match**, but I admit it's debatable.

So on the four cases where it mattered, the "smart" judge was right **once**, and the "dumb" string match was right
**three times**. That's 4 cases, far too few to conclude that string matching is better in general. But it's enough to
conclude that **an LLM judge is not ground truth**. It's another model, with its own errors and biases, and it needs to
be evaluated too.

What I did, and what I would recommend: run both graders, accept the answers where they agree, and **have a human read
only the disagreements**. Here, that's 4 answers out of 48 (8%). In the script, my verdicts are stored in a small
dictionary, so the evaluation stays reproducible:

```python
# I read every disagreement myself (they are printed at the end of the run) and wrote my verdict here.
HUMAN_LABELS = {
    ("hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M", "How many customers buy every month?"): False,  # quotes 93M, then says "I don't know"
    ("hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M", "Will shareholders receive dividends?"): False,  # only says "I don't know"
    ("llama3.1:latest", "How many shops does the company operate?"): False,  # 33,419 is the China count, not the total
    ("llama3.1:latest", "Where does the company buy its coffee beans?"): True,  # incomplete, but not wrong
}


def final_verdict(model, question, match, judged):
    if match == judged:
        return match
    return HUMAN_LABELS.get((model, question))  # None = not reviewed yet
```

That's the "final" column in the results above.

---

## Where do the failures come from?

Now that we have a "final" verdict we can trust, we can do the analysis I mentioned at the start: for each pipeline,
how many failures come from retrieval, and how many from generation?

```python
both = sum(x["retrieved"] and x["final"] for x in rows)
gen_fail = sum(x["retrieved"] and not x["final"] for x in rows)
ret_fail = sum(not x["retrieved"] and not x["final"] for x in rows)
lucky = sum(not x["retrieved"] and x["final"] for x in rows)
```

#### After:

```text
Part 3 pipeline / qwen2.5 7B     OK 10 | generation failed 0 | retrieval failed 2 | right anyway 0
Part 4 pipeline / LFM2.5 1.2B    OK  7 | generation failed 4 | retrieval failed 1 | right anyway 0
Part 4 pipeline / qwen2.5 7B     OK 11 | generation failed 0 | retrieval failed 1 | right anyway 0
Part 4 pipeline / llama3.1 8B    OK 10 | generation failed 1 | retrieval failed 1 | right anyway 0
```

This little table is, for me, the most useful result of the whole series, because it tells you **where to work next**:

* With **Qwen**, there are zero generation failures. Changing the LLM would be a waste of time; every remaining failure
  is retrieval. The one left is the "How many shops?" question, so the next improvement is on the retrieval side (in Part 5,
  K=10 fixed it).
* With the **small model**, 4 of its 5 failures are generation failures. Better retrieval wouldn't help much; the model is
  the bottleneck.
* With **Llama**, it's balanced: one of each (the generation failure is the dividends question, where it said "I don't
  know" with the answer in front of it, exactly like Qwen did at K=1 in Part 5).

Without this split, all three pipelines just look "a bit wrong", and you'd probably start tuning the prompt, or buy a
bigger model.

---

## Practical considerations

1. **Write the test set first, and include unanswerable questions.** Start with 20 to 50 questions written by someone who
   knows the documents. It's the most valuable and least glamorous part of the work.
2. **Always separate retrieval failures from generation failures.** Log whether the evidence was in the context for
   every question; the fix depends on it.
3. **Use more than one metric**, and look for metrics that a bad model could game (like my refusal score).
4. **Don't treat an LLM judge as ground truth.** Use a different model from the one being evaluated, compare it with a
   simple metric, and have a human review the disagreements.
5. **Check the citations automatically.** It's free, and it caught wrong sources that both graders missed.
6. **Make it reproducible.** Temperature 0, fixed seeds, stored human labels: you want to re-run the evaluation after
   every change and compare the numbers.
7. **Remember the size of your test set.** With 12 questions, one answer is 8 points. Small differences (10 vs 11) are a
   signal to investigate, not a proof.

---

## Conclusion

When I started this series, I wanted to bring RAG to life in a way I wish I had been introduced to it: one step at a
time, with real code, and real results, including the ones that didn't go as planned. Six articles later, we have a
complete pipeline running on a laptop, for free, that answers 11 out of 12 questions correctly and refuses the 4 it can't
answer.

But if there is one thing I'd like you to take from this last article, it's this: **a RAG system is only as good as your
ability to measure it.** The "33,419 stores" answer survived chunking, retrieval, re-ranking, generation, citations and
an LLM judge. Only a carefully written reference answer and a human looking at the disagreements caught it.

I would like to thank everyone who read, commented and followed this series. It gave me the motivation to go further
than I had planned. I have a few ideas for what comes next (agents, structured data, multimodal RAG...), and I'd love to
hear in the comments which one you'd like me to cover.

If you've found this series helpful, I'd really appreciate a follow. It helps me know that people are reading and
encourages me to keep writing.

---

## Run it yourself

The scripts are in the `scripts/` folder of my project and should be run from that folder:

1. Install [Ollama](https://ollama.com) and pull the three models:
   `ollama pull qwen2.5:7b`, `ollama pull llama3.1` and `ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M`.
2. Keep the files from Parts 3 to 5 (`embedding_pretrained.py`, `retrieval_eval.py`, `rag_generation.py`) in the same
   folder, with their requirements (`pip install numpy onnxruntime tokenizers huggingface_hub`).
3. Run `python eval_end_to_end.py`. It takes about 15 minutes on my laptop (each answer is generated, then judged).

## References
- L. Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (biases of LLM judges, including self-enhancement)
- S. Es et al. (2023), *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- [Ollama](https://ollama.com)
- [LFM2.5-1.2B-Instruct (Liquid AI)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)
