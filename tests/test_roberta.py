"""Tests for the helpers in src/train_roberta.py. Run from the repo root:

    python -m pytest tests/test_roberta.py -q

No model download and no data needed: only the small pure functions are tested here. The full training path is
checked by running `python -m src.train_roberta --setting indist --dev` (about 1 minute on CPU).

Each test's docstring says what would be broken in the real training run if that test failed.
"""
import numpy as np
import pandas as pd
import pytest

from src.train_roberta import auroc_or_none, class_weights, collate, eval_steps, stratified_cap


def test_class_weights_balanced_and_skewed():
    """If this fails, the loss would no longer balance humans and machines, and RoBERTa could learn to answer "machine" for everything."""
    assert class_weights([0, 1, 0, 1]) == [1.0, 1.0]
    # 1 human vs 4 machine: the rare class gets the bigger weight, n / (2 * n_c)
    w = class_weights([0, 1, 1, 1, 1])
    assert w == pytest.approx([5 / 2, 5 / 8])
    assert w[0] > w[1]


def test_class_weights_needs_both_classes():
    """If this fails, a train split with only one class would give an infinite weight instead of a clear error."""
    with pytest.raises(ValueError):
        class_weights([1, 1, 1])


def test_stratified_cap_keeps_both_labels_and_is_reproducible():
    """If this fails, --dev runs could train on one class only, or pick different rows each run (not reproducible)."""
    df = pd.DataFrame({"id": [f"i{k}" for k in range(100)], "label": [k % 5 == 0 for k in range(100)]})
    df["label"] = df["label"].astype(int)           # 20 machine (1), 80 human (0)
    capped = stratified_cap(df, 10)
    assert len(capped) == 10 and set(capped["label"]) == {0, 1}
    assert (capped["label"] == 1).sum() == 5
    assert capped["id"].tolist() == stratified_cap(df, 10)["id"].tolist()


def test_stratified_cap_small_class_is_not_oversampled():
    """If this fails, --dev would repeat texts of the rarer class instead of keeping each one once."""
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": [0, 0, 0, 1]})
    capped = stratified_cap(df, 10)
    assert len(capped) == 4                         # only 1 machine text exists, it is kept once


def test_eval_steps_one_epoch_checks_twice_and_always_at_the_end():
    """If this fails, checkpoint selection would skip the final weights or check val at the wrong steps."""
    assert eval_steps(100, 1, 2) == {50, 100}
    assert eval_steps(7, 1, 2) == {4, 7}
    assert 200 in eval_steps(100, 2, 1)             # last step of the last epoch
    assert eval_steps(1, 1, 2) == {1}


def test_auroc_or_none():
    """If this fails, a one-class val set would crash training instead of skipping the checkpoint check."""
    assert auroc_or_none([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc_or_none([0, 0, 0], [0.1, 0.2, 0.3]) is None   # one class only: undefined


def test_collate_pads_with_pad_id_and_masks_padding():
    """If this fails, padding tokens would be attended to, and a text's score would depend on its batch-mates."""
    ids, mask = collate([[5, 6, 7], [8]], pad_id=1, device="cpu")
    assert ids.tolist() == [[5, 6, 7], [8, 1, 1]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
