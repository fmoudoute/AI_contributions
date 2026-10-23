# Requirements: the Part 4 setup + Ollama running locally, with: ollama pull qwen2.5:7b
import rag_generation as g
import retrieval_eval as r

ANSWER_KEYS = {
    "How many shops does the company operate?": ["33,596"],
    "How many stores were opened in China this quarter?": ["2,531"],
    "How many stores are in Malaysia?": ["83"],
    "Where does the company buy its coffee beans?": ["Yunnan"],
    "What is the total roasting capacity?": ["155,000"],
    "How many people work for Luckin?": ["190,000"],
    "How many customers buy every month?": ["93 million"],
    "What was the GMV?": ["14.1"],
    "How much did S&M expenses increase?": ["732"],
    "What was the non-GAAP net profit?": ["686"],
    "How much cash does the company have?": ["9.1"],
    "Will shareholders receive dividends?": ["not able to pay", "unable to pay", "cannot pay", "not pay dividends"],
}


def section(title):
    print(f"\n=== {title} ===")


# --- 1. The LLM alone, without retrieval ---
section("1. Without retrieval")
for question in ["How many stores does Luckin Coffee have?", "How many people work for Luckin Coffee?"]:
    print(f"Q: {question}\nA: {g.ask_llm(question)['answer']}\n")

# --- 2. The augmented prompt, and the answer ---
section("2. With retrieval (K=3)")
question = "How many people work for Luckin?"
chunk_ids = g.retrieve(question, k=3)
prompt = g.build_prompt(question, chunk_ids)
result = g.ask_llm(prompt)
print(f"Retrieved chunks: {chunk_ids}")
print(f"Prompt: {len(prompt.split())} words -> {result['prompt_tokens']} tokens for the LLM")
print(f"A: {result['answer']}")
print(f"({result['prompt_tokens']} prompt tokens + {result['answer_tokens']} answer tokens, {result['seconds']:.1f}s)")

# --- 3. Checking the citations ---
section("3. Grounding check")
for question, answer in [
    (question, result["answer"]),
    ("How many shops does the company operate?", "Luckin operates 35,000 stores [chunk 20]."),  # a made-up answer
]:
    print(f"A: {answer}\n   {g.check_grounding(question, answer)}")

# --- 4. Faithful to the context... but wrong ---
section("4. Faithful but wrong")
question = "How many shops does the company operate?"
res = g.rag(question, k=3)
print(f"Retrieved chunks: {res['chunk_ids']}")
print(f"A: {res['answer']}")
i = r.chunks[20].find("In China")
print(f"Chunk 20 says: ...{r.chunks[20][i:i + 120]}...")

# --- 5. How many chunks should we give to the LLM? ---
section("5. Choosing K")
for k in [1, 3, 10]:
    correct, tokens, seconds, failures = 0, 0, 0, []
    for question, keys in ANSWER_KEYS.items():
        res = g.rag(question, k=k)
        tokens += res["prompt_tokens"]
        seconds += res["seconds"]
        if any(key in res["answer"] for key in keys):
            correct += 1
        else:
            failures.append(f"{question} -> {res['answer'][:70]}")
    n = len(ANSWER_KEYS)
    print(f"K={k:2}: correct {correct:2}/{n}, avg prompt {tokens / n:5.0f} tokens, avg {seconds / n:.1f}s")
    for failure in failures:
        print(f"       x {failure}")

# --- 6. Questions the transcript can't answer ---
section("6. Unanswerable questions")
for question in ["What is the revenue guidance for the second quarter of 2026?",
                 "How many stores does Luckin have in Japan?",
                 "Who is the chairman of the board?",
                 "What is the price of a latte?"]:
    print(f"Q: {question}\nA: {g.rag(question)['answer']}")

# --- 7. The context window: silent truncation, again ---
section("7. Context window")
question = "How many stores are in Malaysia?"
chunk_ids = g.retrieve(question, k=30)
print(f"Chunk 25 (the answer) is at position {chunk_ids.index(25)} of {len(chunk_ids)}")
for num_ctx in [2048, 8192]:
    res = g.ask_llm(g.build_prompt(question, chunk_ids), num_ctx=num_ctx)
    print(f"num_ctx={num_ctx}: the LLM read {res['prompt_tokens']} prompt tokens")
    print(f"   A: {res['answer']}")
