"""Tests for the scoring formulas in src/zeroshot.py. Run from the repo root:

    python -m pytest tests/test_zeroshot.py -q

No model download and no data needed: a tiny fake language model with a hand-written logits table stands in
for GPT-2, so every expected number below can be checked by hand (or with plain numpy).

Recipe (same as tests/test_eval.py): known answers, padding must not change a score, bad cases fail loudly.
"""
import numpy as np
import pytest
import torch

from src.zeroshot import token_stats

# logits[i] = the fake LM's next-token logits after seeing token i (vocab of 4 tokens: 0 = BOS, 1, 2, 3)
TABLE = torch.tensor([
    [2.0, 1.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 0.0],      # uniform: token 1 gives no information about the next token
    [1.0, 3.0, -2.0, 0.5],
    [0.2, 0.1, 0.4, 0.3],
])


class FakeLM:
    """Looks like a Hugging Face causal LM: model(input_ids=..., attention_mask=...).logits.

    Logits depend only on the current token, so padding can never change a real position's score."""

    def __call__(self, input_ids, attention_mask):
        return type("Out", (), {"logits": TABLE[input_ids]})()


def _batch(seqs, pad=0):
    longest = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), longest), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids, mask


def _log_softmax(x):
    x = np.asarray(x, dtype=float)
    return x - np.log(np.exp(x).sum())


def _reference(seq, method):
    """Plain-numpy version of each method for one sequence [BOS, t1, t2, ...]."""
    lps, ranks, means, variances = [], [], [], []
    for cur, nxt in zip(seq[:-1], seq[1:]):
        lp_all = _log_softmax(TABLE[cur].numpy())
        p = np.exp(lp_all)
        lps.append(lp_all[nxt])
        ranks.append(1 + (TABLE[cur].numpy() > TABLE[cur].numpy()[nxt]).sum())
        means.append((p * lp_all).sum())
        variances.append((p * lp_all ** 2).sum() - (p * lp_all).sum() ** 2)
    if method == "ppl":
        return float(np.mean(lps))
    if method == "logrank":
        return float(-np.mean(np.log(ranks)))
    return float((np.sum(lps) - np.sum(means)) / np.sqrt(np.sum(variances)))  # fastdetectgpt


@pytest.mark.parametrize("method", ["ppl", "logrank", "fastdetectgpt"])
def test_scores_match_hand_computation(method):
    seq = [0, 1, 2, 3]
    ids, mask = _batch([seq])
    got = token_stats(FakeLM(), ids, mask, method)[0]
    assert got == pytest.approx(_reference(seq, method), abs=1e-5)


def test_logrank_hand_example():
    # After BOS (logits 2,1,0,-1) the actual token 1 is the LM's 2nd choice -> rank 2.
    # After token 1 (uniform) the actual token 3 ties with everything -> rank 1 (ties don't count against it).
    ids, mask = _batch([[0, 1, 3]])
    assert token_stats(FakeLM(), ids, mask, "logrank")[0] == pytest.approx(-np.log(2) / 2, abs=1e-6)


def test_ppl_of_uniform_prediction_is_minus_log_vocab():
    ids, mask = _batch([[1, 3]])    # token 1's row is uniform over 4 tokens
    assert token_stats(FakeLM(), ids, mask, "ppl")[0] == pytest.approx(-np.log(4), abs=1e-6)


@pytest.mark.parametrize("method", ["ppl", "logrank", "fastdetectgpt"])
def test_padding_does_not_change_a_score(method):
    short, long = [0, 2, 1], [0, 1, 2, 3, 1, 2]
    alone = token_stats(FakeLM(), *_batch([short]), method)[0]
    padded = token_stats(FakeLM(), *_batch([short, long]), method)[0]   # `short` now carries 3 pad tokens
    assert padded == pytest.approx(alone, abs=1e-6)


def test_more_ai_like_text_scores_higher():
    # Token 1 is the top choice after BOS and token 0 after token 2 (high probability): "AI-like".
    # The reverse picks the LM's least likely tokens: "human-like". Higher score = more AI for all three methods.
    ai, human = [0, 0, 0], [0, 3, 2]
    for method in ("ppl", "logrank", "fastdetectgpt"):
        s_ai = token_stats(FakeLM(), *_batch([ai]), method)[0]
        s_human = token_stats(FakeLM(), *_batch([human]), method)[0]
        assert s_ai > s_human, method
