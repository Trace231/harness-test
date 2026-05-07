"""Debug: 看 V2 init 后的 decl 质量、看一次 predict 的实际 prompt + LLM 输出。"""
import json
from llm_client import call_llm, count_tokens, count_messages_tokens
from solution import MyHarness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


train = load_jsonl("data/train_dev.jsonl")
dev = load_jsonl("data/test_dev.jsonl")

harness = MyHarness(call_llm, count_tokens, count_messages_tokens, 2048)
for item in train:
    harness.update(item["text"], item["label"])

print(">>> Triggering init...")
import time
t0 = time.time()
harness._ensure_init()
print(f"Init done in {time.time() - t0:.1f}s\n")

# 看 5 个 decl
print(">>> Sample decls:")
for label in ["card_arrival", "card_delivery_estimate", "atm_support",
              "lost_or_stolen_phone", "exchange_rate"]:
    if label in harness._decl:
        print(f"  [{label}]")
        print(f"    {harness._decl[label]}")
print()

# 看一个具体 query 的完整流程
def trace(text, true_label):
    print(f">>> Query: {text}")
    print(f"    True label: {true_label}")
    demos = harness._retrieve(text, k=harness.TOP_K_RETRIEVE)
    candidates = harness._build_candidates(demos)
    print(f"    Candidates: {candidates}")

    decl_lines = []
    for l in candidates:
        d = harness._decl.get(l, "").strip()
        decl_lines.append(f"- {l}: {d}" if d else f"- {l}")
    decl_block = "\n".join(decl_lines)

    system = (
        "You are a strict text classifier. Choose exactly one label "
        "from the candidates below. Each candidate has a short "
        "description focused on what distinguishes it from similar labels.\n\n"
        f"Candidates:\n{decl_block}\n\n"
        "Output exactly one label name from the list above. "
        "No explanation, no punctuation, no quotation marks — just the label."
    )

    messages = [{"role": "system", "content": system}]
    for d_text, d_label, _ in reversed(demos):
        messages.append({"role": "user", "content": f"Text: {d_text}"})
        messages.append({"role": "assistant", "content": d_label})
    messages.append({"role": "user", "content": f"Text: {text}"})

    print(f"    [SYSTEM]:\n{system[:600]}...\n")
    resp = call_llm(messages)
    print(f"    [LLM RAW]: {resp!r}")
    pred = harness._snap_to_label(resp.strip().splitlines()[0].strip())
    print(f"    Predicted: {pred}  ({'✓' if pred == true_label else '✗'})")
    print()


# 抽几个之前 V1 错的样本看看 V2
trace("My card has not arrived yet, where is it?", "card_arrival")
trace("How can I find the nearest ATM?", "atm_support")
trace("Why was my payment reversed?", "reverted_card_payment?")
trace("im not sure what this charge is for", "card_payment_fee_charged")
