"""BLEU-4 and ROUGE-L for report generation evaluation."""

import re
from typing import List, Sequence

import nltk
import numpy as np

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize_for_bleu(text: str) -> List[str]:
    """Simple tokenization for BLEU (lowercase, split on non-alnum)."""
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens if tokens else [""]


def bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 (corpus-style single sentence)."""
    ref_tok = tokenize_for_bleu(reference)
    hyp_tok = tokenize_for_bleu(hypothesis)
    if not ref_tok or not hyp_tok:
        return 0.0
    smooth = SmoothingFunction()
    return float(
        sentence_bleu(
            [ref_tok],
            hyp_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth.method1,
        )
    )


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 (longest common subsequence)."""
    ref_tok = tokenize_for_bleu(reference)
    hyp_tok = tokenize_for_bleu(hypothesis)
    if not ref_tok or not hyp_tok:
        return 0.0
    lcs_len = _lcs_length(ref_tok, hyp_tok)
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / len(hyp_tok)
    rec = lcs_len / len(ref_tok)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of longest common subsequence."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def compute_metrics(references: List[str], hypotheses: List[str]) -> dict:
    """Compute BLEU-4 and ROUGE-L over lists of reference/hypothesis pairs."""
    assert len(references) == len(hypotheses)
    bleu_scores = [bleu4(r, h) for r, h in zip(references, hypotheses)]
    rouge_scores = [rouge_l(r, h) for r, h in zip(references, hypotheses)]
    return {
        "BLEU-4": float(np.mean(bleu_scores)),
        "ROUGE-L": float(np.mean(rouge_scores)),
    }
