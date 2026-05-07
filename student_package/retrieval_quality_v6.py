"""
retrieval_quality_v6.py — 量化 V6 (HyDE 加持) 的检索召回率

跑 539 次 HyDE 改写（约 4-5 min）+ 检索，统计 recall@k。
对比 V3 char-only，看 HyDE 把检索推到哪里。
"""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import call_llm, count_tokens, count_messages_tokens
from solution import MyHarness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    train = load_jsonl("data/train_dev.jsonl")
    test = load_jsonl("data/test_dev.jsonl")

    h = MyHarness(call_llm, count_tokens, count_messages_tokens, 2048)
    for it in train:
        h.update(it["text"], it["label"])
    print("Init (含 decl 生成)...")
    t0 = time.time()
    h._ensure_init()
    print(f"  Init done in {time.time()-t0:.1f}s")

    print(f"\nTest={len(test)}  Labels={len(h._label_set)}")
    print("跑 539 次 HyDE + 检索...\n")

    Ks = [1, 3, 5, 8, 10, 15, 20, 30, 50]
    recall = {k: 0 for k in Ks}
    rank_of_correct = []

    def eval_one(idx_item):
        idx, item = idx_item
        text = item["text"]
        true = item["label"]
        # 复用 solution.py 实际 predict 走的检索路径（含 RRF 选项）
        rewrites = h._hyde_rewrite(text)
        if rewrites and h.USE_RRF:
            retrieval_input = [text] + rewrites
        elif rewrites:
            retrieval_input = text + " " + " ".join(rewrites)
        else:
            retrieval_input = text
        retrieved = h._retrieve(retrieval_input, k=max(Ks))
        labels_in_order = [r[1] for r in retrieved]
        rank = next((i for i, l in enumerate(labels_in_order, 1) if l == true), -1)
        return idx, rank, labels_in_order

    t0 = time.time()
    results = [None] * len(test)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(eval_one, (i, x)): i for i, x in enumerate(test)}
        done = 0
        for fut in as_completed(futures):
            idx, rank, labels = fut.result()
            results[idx] = (rank, labels)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(test)}  ({time.time()-t0:.0f}s)")
    print(f"  完成 {time.time()-t0:.0f}s\n")

    for rank, labels in results:
        rank_of_correct.append(rank)
        for k in Ks:
            if rank > 0 and rank <= k:
                recall[k] += 1

    n = len(test)
    print(f"{'k':>4} | recall@k | demos cover true label")
    print("-" * 40)
    for k in Ks:
        print(f"{k:>4} | {recall[k]/n*100:6.2f}% | {recall[k]}/{n}")

    found = [r for r in rank_of_correct if r > 0]
    not_found = sum(1 for r in rank_of_correct if r == -1)
    if found:
        print(f"\n真值在 top-{max(Ks)} 中找到: {len(found)}/{n} ({len(found)/n*100:.1f}%)")
        print(f"找到时的平均 rank: {sum(found)/len(found):.2f}")
        print(f"中位 rank: {sorted(found)[len(found)//2]}")
        print(f"未找到（不在 top-{max(Ks)}）: {not_found}")


if __name__ == "__main__":
    main()
