"""
deep_error_analysis.py — 对 V6 全部错例做结构化分析

需要 preds_baseline.json (analyze.py 跑过)
对每个错例:
  - 真值 label / 预测 label
  - retrieve top-8 召回了哪些 label（真值是否在）
  - 错例的 query 文本
按 (true → pred) 错配对分组，显示每组所有样本
最后给出错误结构归类
"""
import json
from collections import Counter, defaultdict

from llm_client import count_tokens, count_messages_tokens
from solution import MyHarness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    train = load_jsonl("data/train_dev.jsonl")
    test = load_jsonl("data/test_dev.jsonl")
    preds = json.load(open("preds_baseline.json"))["predictions"]

    print(f"Test={len(test)}  Pred 数={len(preds)}\n")

    # 重建 retrieval 用（不调 LLM，纯检索）
    def fake_llm(m): return ""
    h = MyHarness(fake_llm, count_tokens, count_messages_tokens, 2048)
    for it in train:
        h.update(it["text"], it["label"])
    h._fit_tfidf()

    failures = []
    for i, item in enumerate(test):
        pred = preds[i]
        true = item["label"]
        if pred != true:
            # 跑 retrieval（无 HyDE）—— 只看 lexical retrieval 是否到位
            retrieved = h._retrieve(item["text"], k=15)
            top_labels = [r[1] for r in retrieved]
            true_in_top = next((rk for rk, l in enumerate(top_labels, 1) if l == true), None)
            pred_in_top = next((rk for rk, l in enumerate(top_labels, 1) if l == pred), None)
            failures.append({
                "idx": i,
                "text": item["text"],
                "true": true,
                "pred": pred,
                "true_rank": true_in_top,
                "pred_rank": pred_in_top,
                "top_labels": top_labels[:8],
            })

    print(f"失败总数: {len(failures)}\n")

    # ---- 错误类型分类 ----
    cat = Counter()
    for f in failures:
        if f["true_rank"] is None:
            cat["A: retrieval 完全没召回真值（true 不在 top-15）"] += 1
        elif f["true_rank"] > 8 and f["pred_rank"] and f["pred_rank"] <= 8:
            cat["B: 真值在 9-15 候选区，pred 在 top-8（LLM 没看到真值）"] += 1
        elif f["true_rank"] <= 8 and f["pred_rank"] and f["pred_rank"] <= 8:
            cat["C: 真值和 pred 都在 top-8（LLM 看到了真值但选错）"] += 1
        elif f["true_rank"] is not None and f["pred"] not in f["top_labels"][:15]:
            cat["D: pred 不在 candidates，是被 4 级 fallback snap 来的"] += 1
        else:
            cat["E: 其他"] += 1

    print("错误结构分类：")
    for k, v in cat.most_common():
        print(f"  {v:3d}  {k}")
    print()

    # ---- 按 (true → pred) 分组 ----
    by_pair = defaultdict(list)
    for f in failures:
        by_pair[(f["true"], f["pred"])].append(f)

    # 显示出现 ≥ 2 次的混淆对的所有样本
    print("=" * 80)
    print("出现 ≥ 2 次的 (真值 → 预测) 错配对的全部样本：")
    print("=" * 80)
    for pair, items in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        if len(items) < 2:
            continue
        print(f"\n[{pair[0]} → {pair[1]}]  ({len(items)} 次)")
        for f in items:
            rank_info = f"true rank={f['true_rank']}, pred rank={f['pred_rank']}"
            print(f"  [{f['idx']}] ({rank_info})")
            print(f"    text: {f['text'][:140]}")

    # ---- 真值 rank 分布 ----
    print("\n" + "=" * 80)
    print("真值 label 在 retrieval top-K 中的位置（用于看 retrieval 是否是瓶颈）")
    print("=" * 80)
    rank_buckets = Counter()
    for f in failures:
        r = f["true_rank"]
        if r is None:
            rank_buckets["不在 top-15"] += 1
        elif r == 1:
            rank_buckets["rank 1"] += 1
        elif r <= 3:
            rank_buckets["rank 2-3"] += 1
        elif r <= 8:
            rank_buckets["rank 4-8"] += 1
        else:
            rank_buckets["rank 9-15"] += 1
    for k, v in [("rank 1", rank_buckets["rank 1"]),
                 ("rank 2-3", rank_buckets["rank 2-3"]),
                 ("rank 4-8", rank_buckets["rank 4-8"]),
                 ("rank 9-15", rank_buckets["rank 9-15"]),
                 ("不在 top-15", rank_buckets["不在 top-15"])]:
        bar = "█" * v
        print(f"  {k:>12}: {v:>3}  {bar}")
    print(f"  小计 90 错例")

    # ---- 看每个 true label 的具体表现 ----
    print("\n" + "=" * 80)
    print("错得最多的真值 label（≥ 3 次）：")
    print("=" * 80)
    by_true = defaultdict(list)
    for f in failures:
        by_true[f["true"]].append(f)
    for label, items in sorted(by_true.items(), key=lambda x: -len(x[1])):
        if len(items) < 3:
            continue
        all_for_label = sum(1 for x in test if x["label"] == label)
        print(f"\n[{label}]  错 {len(items)}/{all_for_label}")
        # 训练样本
        train_examples = [t for t, l in [(x["text"], x["label"]) for x in train] if l == label]
        print(f"  训练样本:")
        for t in train_examples:
            print(f"    \"{t}\"")
        print(f"  错的 query:")
        for f in items:
            print(f"    [{f['idx']}] (pred={f['pred']}, true rank={f['true_rank']}) \"{f['text'][:100]}\"")


if __name__ == "__main__":
    main()
