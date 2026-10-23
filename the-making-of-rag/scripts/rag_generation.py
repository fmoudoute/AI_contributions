# Requirements: the Part 4 setup + Ollama running locally (https://ollama.com), with: ollama pull qwen2.5:7b
import json
import re
import urllib.request

import numpy as np

import retrieval_eval as r

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


# --- Retrieval: the best method from Part 4 (hybrid + re-rank) ---
def retrieve(query, k=3):
    scores = r.rerank(query, r.rrf(r.bm25.scores(query), r.semantic_scores(query)))
    return [int(i) for i in np.argsort(-scores)[:k]]


# --- The augmented prompt ---
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
    chunk_ids = retrieve(question, k)
    result = ask_llm(build_prompt(question, chunk_ids), model=model)
    result["chunk_ids"] = chunk_ids
    return result


# --- Grounding check: is every number of the answer really in the chunks it cites? ---
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def check_grounding(question, answer):
    cited = sorted({int(c) for c in re.findall(r"chunk (\d+)", answer)})
    text = re.sub(r"\[?chunks? [\d, chunk]+\]?", " ", answer)  # remove the citations themselves
    numbers = sorted({n.rstrip(".,") for n in NUMBER.findall(text)} - set(NUMBER.findall(question)))
    cited_text = " ".join(r.chunks[c] for c in cited if c < len(r.chunks))
    return {"cited": cited, "grounded": [n for n in numbers if n in cited_text],
            "not_grounded": [n for n in numbers if n not in cited_text]}
