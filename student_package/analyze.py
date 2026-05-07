"""
analyze.py — 跑一轮 baseline，dump 预测 + 错例分析
"""
import json
import sys
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import call_llm as _raw_call_llm, count_tokens, count_messages_tokens, truncate_to_tokens
from solution import MyHarness


def make_llm(max_prompt_tokens, tracker, lock):
    def _call(messages):
        prompt_text = " ".join(m.get("content", "") for m in messages)
        n = count_tokens(prompt_text)
        if n > max_prompt_tokens:
            messages = list(messages)
            excess = n - max_prompt_tokens
            for i in range(len(messages) - 1, -1, -1):
                if excess <= 0: break
                content = messages[i].get("content", "")
                t = count_tokens(content)
                if t <= excess:
                    messages[i] = {**messages[i], "content": ""}; excess -= t
                else:
                    messages[i] = {**messages[i], "content": truncate_to_tokens(content, t - excess)}; excess = 0
        resp = _raw_call_llm(messages)
        with lock:
            tracker["prompt"] += n
            tracker["completion"] += count_tokens(resp)
        return resp
    return _call


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    train = load_jsonl("data/train_dev.jsonl")
    dev = load_jsonl("data/test_dev.jsonl")
    label_set = sorted({x["label"] for x in train})

    tracker = {"prompt": 0, "completion": 0}
    lock = threading.Lock()
    llm = make_llm(2048, tracker, lock)

    harness = MyHarness(llm, count_tokens, count_messages_tokens, 2048)
    for item in train:
        harness.update(item["text"], item["label"])

    predictions = [None] * len(dev)
    errors = []
    t0 = time.time()

    def run_one(args):
        idx, item = args
        try:
            return idx, harness.predict(item["text"]).strip(), None
        except Exception as e:
            return idx, "", str(e)

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(run_one, (i, item)) for i, item in enumerate(dev)]
        done = 0
        for fut in as_completed(futures):
            idx, pred, err = fut.result()
            predictions[idx] = pred
            if err: errors.append((idx, err))
            done += 1
            if done % 50 == 0:
                sys.stderr.write(f"\r{done}/{len(dev)}")
                sys.stderr.flush()
    sys.stderr.write("\n")

    elapsed = time.time() - t0
    correct = sum(1 for item, p in zip(dev, predictions) if p == item["label"])
    acc = correct / len(dev) * 100
    print(f"\n准确率: {acc:.1f}%  耗时: {elapsed:.1f}s")
    print(f"prompt tok/条: {tracker['prompt']/len(dev):.1f}")
    print(f"completion tok/条: {tracker['completion']/len(dev):.1f}")
    print(f"调用错误: {len(errors)}")

    # 失败样本
    fails = [(i, dev[i]["text"], dev[i]["label"], predictions[i])
             for i in range(len(dev)) if predictions[i] != dev[i]["label"]]

    # 1. 输出是否在合法 label_set 中？
    in_set = sum(1 for _, _, _, p in fails if p in label_set)
    out_set = len(fails) - in_set
    print(f"\n失败 {len(fails)} 个 | 在 label_set 内: {in_set} | 不在内: {out_set}")

    # 2. label 错位分布（前 10）
    confusion = Counter((true, pred) for _, _, true, pred in fails)
    print("\n[Top 10 误判对] (true → pred : count)")
    for (t, p), c in confusion.most_common(10):
        print(f"  {c:3d}  {t}  →  {p}")

    # 3. 每类失败数（前 10）
    fail_by_label = Counter(true for _, _, true, _ in fails)
    print("\n[Top 10 错得最多的真值 label]")
    for lbl, c in fail_by_label.most_common(10):
        total = sum(1 for x in dev if x["label"] == lbl)
        print(f"  {c}/{total}  {lbl}")

    # 4. 不在 label_set 中的输出样例
    print("\n[预测输出不在 label_set 内的样例（前 8）]")
    out_of_set_samples = [(t, p) for _, _, t, p in fails if p not in label_set][:8]
    for true, pred in out_of_set_samples:
        print(f"  true={true}  pred={pred!r}")

    # 5. 随机抽 10 个错例看 query
    print("\n[随机错例 10 个]")
    import random
    random.seed(0)
    for i, text, true, pred in random.sample(fails, min(10, len(fails))):
        print(f"  [{i}] true={true}  pred={pred}")
        print(f"       text={text[:120]}")

    # dump 预测供后续 ablation 对比
    with open("preds_baseline.json", "w") as f:
        json.dump({"acc": acc, "predictions": predictions, "elapsed": elapsed,
                   "tracker": tracker}, f, indent=2)
    print("\n已保存 preds_baseline.json")


if __name__ == "__main__":
    main()
