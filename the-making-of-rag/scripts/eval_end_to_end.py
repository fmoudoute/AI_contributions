# Requirements: the Part 5 setup (rag_generation.py) + ollama pull llama3.1 and LiquidAI LFM2.5 (see article)
import re

import numpy as np

import rag_generation as g
import retrieval_eval as r

REFUSAL = "I don't know based on the provided context"

# --- The test set, with a reference answer and the key fact(s) the answer must contain ---
EVAL_SET = [
    # (question, evidence in the right chunk, reference answer, key facts: ALL groups must match, ANY alternative per group)
    ("How many shops does the company operate?", "33,596",
     "33,596 stores in total at the end of Q1 2026 (33,419 in China and 177 overseas).", [["33,596"]]),
    ("How many stores were opened in China this quarter?", "2,531 net new stores",
     "2,531 net new stores in China.", [["2,531"]]),
    ("How many stores are in Malaysia?", "83 franchised stores in Malaysia",
     "83 franchised stores in Malaysia.", [["83"]]),
    ("Where does the company buy its coffee beans?", "Yunnan",
     "From origin regions, domestic and international; over 30,000 tons came from Yunnan this harvest season.", [["Yunnan"]]),
    ("What is the total roasting capacity?", "155,000 tons",
     "155,000 tons of total annual roasting capacity.", [["155,000"]]),
    ("How many people work for Luckin?", "190,000 Luckin team members",
     "About 190,000 Luckin team members.", [["190,000"]]),
    ("How many customers buy every month?", "93 million",
     "Over 93 million average monthly transacting customers.", [["93 million"]]),
    ("What was the GMV?", "RMB14.1 billion", "GMV was RMB14.1 billion.", [["14.1"]]),
    ("How much did S&M expenses increase?", "RMB732 million",
     "S&M expenses rose 48% year-over-year to RMB732 million.", [["48%"], ["732"]]),
    ("What was the non-GAAP net profit?", "RMB686 million",
     "Non-GAAP net profit was RMB686 million, a 5.7% net margin.", [["686"]]),
    ("How much cash does the company have?", "RMB9.1 billion",
     "About RMB9.1 billion as of March 31, 2026.", [["9.1"]]),
    ("Will shareholders receive dividends?", "not able to pay dividends",
     "Not in the short term: for historical reasons the company cannot pay dividends, but it is evaluating alternatives.",
     [["not able to pay", "unable to pay", "cannot pay", "can't pay", "not pay dividends"]]),
    # Questions the transcript can NOT answer: the right behaviour is to say so
    ("What is the revenue guidance for the second quarter of 2026?", None, REFUSAL, None),
    ("How many stores does Luckin have in Japan?", None, REFUSAL, None),
    ("Who is the chairman of the board?", None, REFUSAL, None),
    ("What is the price of a latte?", None, REFUSAL, None),
]


# --- Metric 1: correctness by string matching (cheap, strict, sometimes wrong) ---
def is_refusal(answer):
    return "don't know" in answer.lower() or "do not know" in answer.lower()


def string_match(answer, keys):
    if keys is None:
        return is_refusal(answer)
    return all(any(alternative in answer for alternative in group) for group in keys)


# --- Metric 2: LLM-as-a-judge (another LLM grades the answer against the reference answer) ---
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


# --- Metric 3: a human decides when the two metrics disagree ---
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


# --- Run a full pipeline on the test set ---
def run_pipeline(name, retrieve_fn, model=g.MODEL, k=3):
    rows = []
    for question, evidence, reference, keys in EVAL_SET:
        chunk_ids = retrieve_fn(question, k)
        result = g.ask_llm(g.build_prompt(question, chunk_ids), model=model)
        answer = result["answer"]
        grounding = g.check_grounding(question, answer)
        match, judged = string_match(answer, keys), judge(question, reference, answer)
        rows.append({
            "question": question,
            "answerable": evidence is not None,
            "retrieved": evidence is None or any(evidence in r.chunks[i] for i in chunk_ids),
            "match": match,
            "judge": judged,
            "final": final_verdict(model, question, match, judged),
            "cited": bool(grounding["cited"]),
            "unsupported": grounding["not_grounded"],
            "answer": answer,
            "seconds": result["seconds"] - result["load_seconds"],  # don't count model loading
        })
    answerable = [x for x in rows if x["answerable"]]
    unanswerable = [x for x in rows if not x["answerable"]]
    print(f"{name:34} "
          f"correct(match) {sum(x['match'] for x in answerable):2}/{len(answerable)}  "
          f"correct(judge) {sum(x['judge'] for x in answerable):2}/{len(answerable)}  "
          f"final {sum(bool(x['final']) for x in answerable):2}/{len(answerable)}  "
          f"refused {sum(x['match'] for x in unanswerable)}/{len(unanswerable)}  "
          f"unsupported numbers {sum(len(x['unsupported']) for x in rows):2}  "
          f"avg {np.mean([x['seconds'] for x in rows]):.1f}s")
    return rows


def semantic_only(question, k):
    return [int(i) for i in np.argsort(-r.semantic_scores(question))[:k]]


if __name__ == "__main__":
    print("--- Same model (qwen2.5:7b), two retrieval pipelines ---")
    part3 = run_pipeline("Part 3: semantic only, K=3", semantic_only)
    part4 = run_pipeline("Part 4: hybrid + re-rank, K=3", g.retrieve)

    print("\n--- Same pipeline (hybrid + re-rank, K=3), three generation models ---")
    models = {
        "LFM2.5 1.2B": "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M",
        "qwen2.5 7B": "qwen2.5:7b",
        "llama3.1 8B": "llama3.1:latest",
    }
    by_model = {}
    for name, model in models.items():
        by_model[name] = part4 if model == g.MODEL else run_pipeline(name, g.retrieve, model=model)
        if model == g.MODEL:
            print(f"{name:34} (same run as above)")

    # --- Where do the failures come from: retrieval or generation? ---
    print("\n--- Failure attribution (answerable questions) ---")
    for name, rows in [("Part 3 pipeline / qwen2.5 7B", part3)] + [(f"Part 4 pipeline / {m}", x) for m, x in by_model.items()]:
        rows = [x for x in rows if x["answerable"]]
        both = sum(x["retrieved"] and x["final"] for x in rows)
        gen_fail = sum(x["retrieved"] and not x["final"] for x in rows)
        ret_fail = sum(not x["retrieved"] and not x["final"] for x in rows)
        lucky = sum(not x["retrieved"] and x["final"] for x in rows)
        print(f"{name:32} OK {both:2} | generation failed {gen_fail} | retrieval failed {ret_fail} | right anyway {lucky}")

    # --- Every failure, so we can read them one by one ---
    print("\n--- Failures and disagreements ---")
    for name, rows in [("Part 3 / qwen", part3)] + [(f"Part 4 / {m}", x) for m, x in by_model.items()]:
        for x in rows:
            if not (x["match"] and x["judge"]) or x["unsupported"]:
                flag = (f"match={x['match']} judge={x['judge']} final={x['final']} "
                        f"retrieved={x['retrieved']} unsupported={x['unsupported']}")
                print(f"[{name}] {x['question']}\n    {flag}\n    -> {x['answer'][:260]!r}")
