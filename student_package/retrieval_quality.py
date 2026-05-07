"""
retrieval_quality.py — 量化 retrieval 召回质量

检索是 bottleneck 还是非瓶颈？计算 recall@k：
   recall@k = 测试样本中，正确 label 出现在 top-k 召回的 demo 标签集合里的比例

如果 recall@8 ~= 95% 而 accuracy = 81%，那瓶颈在 LLM 选择
如果 recall@8 ~= 81%，那 retrieval 是瓶颈（LLM 已经几乎挑得最优）
"""
import json
from collections import Counter

from llm_client import count_tokens, count_messages_tokens
from solution import MyHarness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    train = load_jsonl("data/train_dev.jsonl")
    test = load_jsonl("data/test_dev.jsonl")

    # 不调 LLM 的 dummy
    def fake_llm(messages): return ""

    h = MyHarness(fake_llm, count_tokens, count_messages_tokens, 2048)
    for it in train:
        h.update(it["text"], it["label"])
    h._fit_tfidf()
    h._compute_centroids()  # 让 centroid boost 生效

    print(f"Train={len(train)}  Test={len(test)}  Labels={len(h._label_set)}\n")

    # 测试不同 k
    Ks = [1, 3, 5, 8, 10, 15, 20, 30]
    recall = {k: 0 for k in Ks}
    rank_of_correct = []  # 真值在召回排名

    for item in test:
        text = item["text"]
        true = item["label"]
        # 按 max(K) 召回，然后对每个 K 切片
        retrieved = h._retrieve(text, k=max(Ks))
        labels_in_order = [r[1] for r in retrieved]

        # 找真值首次出现的 rank
        rank = next((i for i, l in enumerate(labels_in_order, 1) if l == true), -1)
        rank_of_correct.append(rank)

        for k in Ks:
            top_k_labels = set(labels_in_order[:k])
            if true in top_k_labels:
                recall[k] += 1

    n = len(test)
    print(f"{'k':>4} | recall@k | demos cover true label")
    print("-" * 40)
    for k in Ks:
        print(f"{k:>4} | {recall[k]/n*100:6.2f}% | {recall[k]}/{n}")

    # rank 分布
    found = [r for r in rank_of_correct if r > 0]
    not_found = sum(1 for r in rank_of_correct if r == -1)
    if found:
        print(f"\n真值在 top-{max(Ks)} 中找到: {len(found)}/{n} ({len(found)/n*100:.1f}%)")
        print(f"找到时的平均 rank: {sum(found)/len(found):.2f}")
        print(f"中位 rank: {sorted(found)[len(found)//2]}")
        print(f"未找到（不在 top-{max(Ks)}）: {not_found}")

    # rank 分布直方图
    print(f"\nRank 分布 (在 top-{max(Ks)} 内):")
    bins = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 10), (11, 15), (16, 20), (21, 30)]
    for lo, hi in bins:
        cnt = sum(1 for r in found if lo <= r <= hi)
        bar = "█" * int(cnt / max(1, max(found)) * 50)
        print(f"  rank {lo:>2}-{hi:>2}: {cnt:>4}  {bar}")

    # 对比：candidates (按 _build_candidates) 的 recall
    print("\n--- Candidates (predict 实际用的，top-8 unique label) ---")
    cand_recall = 0
    cand_size_dist = Counter()
    for item in test:
        retrieved = h._retrieve(item["text"], k=h.TOP_K_RETRIEVE)
        cands = h._build_candidates(retrieved)
        cand_size_dist[len(cands)] += 1
        if item["label"] in cands:
            cand_recall += 1
    print(f"recall (candidates): {cand_recall/n*100:.2f}% ({cand_recall}/{n})")
    print(f"candidate 数量分布: {dict(sorted(cand_size_dist.items()))}")


if __name__ == "__main__":
    main()
