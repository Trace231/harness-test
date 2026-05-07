"""
solution.py — 考生唯一需要提交的文件

规则
----
1. 只能修改 MyHarness 类内部；其余部分不可改动。考生可以先行查看 harness_base.py 以了解可用接口和调用约定。
2. 只允许 import Python 标准库（re, math, random, json, collections 等）、numpy
   以及 harness_base（已提供）。
3. 禁止 import 其他第三方库（openai, sklearn, torch …）。
4. 禁止通过任何途径读写磁盘文件。
5. call_llm 每次调用的 prompt token 数若超过 max_prompt_tokens，
   会被自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

import math
import re
from collections import Counter

import numpy as np

from harness_base import Harness


# ============================================================
# 工具函数
# ============================================================
def _char_ngrams(s: str, n_min: int = 3, n_max: int = 5):
    """生成 char n-gram 特征（lowercase 后切片）。"""
    s = s.lower()
    out = []
    for n in range(n_min, n_max + 1):
        if len(s) < n:
            continue
        out.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return out


def _normalize_label(s: str) -> str:
    """归一化用于比较：lower + 去非字母数字。"""
    return re.sub(r'[^\w]', '', s.lower())


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（迭代 DP）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + (ca != cb),  # substitute
            ))
        prev = cur
    return prev[-1]


# ============================================================
# 考生实现区
# ============================================================
class MyHarness(Harness):
    """Baseline + 改动 1 (4 级 parse fallback) + 改动 2 (system label list).

    核心:
      - char n-gram TF-IDF 检索 (k=5)
      - Multi-turn few-shot prompt
      - System prompt 列出全部合法 label (改动 2)
      - 输出经过 4 级 fallback snap 到合法 label_set (改动 1)
    """

    TOP_K = 5
    NGRAM_MIN = 3
    NGRAM_MAX = 5

    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        self._fitted = False
        self._vocab: dict = {}
        self._idf: np.ndarray | None = None
        self._X: np.ndarray | None = None  # (N, V), L2-normalized
        self._label_set: set = set()
        self._labels_sorted: list = []
        self._labels_block: str = ""
        self._norm_to_label: dict = {}  # 归一化 → 原 label

    # ----- TF-IDF 拟合 / 检索 ----------------------------------
    def _fit(self) -> None:
        if self._fitted or not self.memory:
            return
        texts = [t for t, _ in self.memory]
        labels = [l for _, l in self.memory]
        rows = [Counter(_char_ngrams(t, self.NGRAM_MIN, self.NGRAM_MAX)) for t in texts]

        # 词表
        for r in rows:
            for f in r:
                if f not in self._vocab:
                    self._vocab[f] = len(self._vocab)

        N = len(texts)
        V = len(self._vocab)

        # IDF
        df = np.zeros(V, dtype=np.float32)
        for r in rows:
            for f in r:
                df[self._vocab[f]] += 1.0
        self._idf = np.log(N / np.maximum(df, 1.0)).astype(np.float32)

        # TF-IDF 矩阵 (sub-linear tf)
        X = np.zeros((N, V), dtype=np.float32)
        for i, r in enumerate(rows):
            for f, c in r.items():
                j = self._vocab[f]
                X[i, j] = (1.0 + math.log(c)) * self._idf[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        self._X = X / norms

        # Label 集合
        self._label_set = set(labels)
        self._labels_sorted = sorted(self._label_set)
        self._labels_block = "\n".join(f"- {l}" for l in self._labels_sorted)
        self._norm_to_label = {_normalize_label(l): l for l in self._labels_sorted}

        self._fitted = True

    def _vectorize(self, text: str) -> np.ndarray:
        c = Counter(_char_ngrams(text, self.NGRAM_MIN, self.NGRAM_MAX))
        v = np.zeros(self._X.shape[1], dtype=np.float32)
        for f, cnt in c.items():
            j = self._vocab.get(f)
            if j is not None:
                v[j] = (1.0 + math.log(cnt)) * self._idf[j]
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v

    def _retrieve(self, text: str, k: int):
        q = self._vectorize(text)
        scores = self._X @ q
        k = min(k, len(self.memory))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.memory[i][0], self.memory[i][1], float(scores[i])) for i in idx]

    # ----- 4 级 parse fallback (改动 1) ------------------------
    def _snap_to_label(self, raw: str) -> str:
        """把 LLM 输出 snap 到合法 label_set 中。

        L1: 原样匹配
        L2: 归一化匹配（lower + 去标点）
        L3: 子串包含
        L4: 编辑距离最近邻
        """
        if not raw or not self._label_set:
            # 兜底：返回任意一个 label，避免空字符串
            return self._labels_sorted[0] if self._labels_sorted else ""

        # L1: 原样
        if raw in self._label_set:
            return raw

        # L2: 归一化
        norm = _normalize_label(raw)
        if norm in self._norm_to_label:
            return self._norm_to_label[norm]

        # L3: 子串（label 是 raw 的子串，或 raw 是 label 的子串）
        rl = raw.lower()
        # 优先：label 出现在 raw 中（更具体）。取最长匹配避免短 label 误命中。
        substring_hits = [l for l in self._labels_sorted if l.lower() in rl]
        if substring_hits:
            return max(substring_hits, key=len)
        # 反向：raw 在 label 中
        substring_hits = [l for l in self._labels_sorted if rl in l.lower()]
        if substring_hits:
            return min(substring_hits, key=len)  # 选最具体（最短）

        # L4: 编辑距离最近邻
        return min(self._labels_sorted,
                   key=lambda l: _edit_distance(norm, _normalize_label(l)))

    # ----- 主接口 ---------------------------------------------
    def predict(self, text: str) -> str:
        self._fit()
        if not self.memory:
            return ""

        demos = self._retrieve(text, k=self.TOP_K)
        # 升序：最相关的紧邻 query
        demos = list(reversed(demos))

        # 改动 2: system 列出全部合法 label
        system_content = (
            "You are a strict text classifier. "
            "Output exactly one label from the allowed set below. "
            "No explanation, no punctuation, no extra text — just the label string.\n\n"
            f"Allowed labels:\n{self._labels_block}\n\n"
            "Output exactly one label from the list above."
        )

        messages = [{"role": "system", "content": system_content}]
        for d_text, d_label, _ in demos:
            messages.append({"role": "user", "content": f"Text: {d_text}"})
            messages.append({"role": "assistant", "content": d_label})
        messages.append({"role": "user", "content": f"Text: {text}"})

        resp = self.call_llm(messages)
        if not resp:
            return self._snap_to_label("")

        first_line = resp.strip().splitlines()[0] if resp.strip() else ""
        # 改动 1: 4 级 snap
        return self._snap_to_label(first_line.strip())
