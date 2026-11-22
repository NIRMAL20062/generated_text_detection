"""Tests for the pure helpers in src/paraphrase.py. Run from the repo root:

    python -m pytest tests/test_paraphrase.py -q

No model download and no data needed. The full path is checked by `python -m src.paraphrase --dev`.
"""
import pandas as pd
import pytest

from src.paraphrase import bleu, build_splits, chunk_long, make_rows, split_sentences


def test_split_sentences_basic_and_whitespace_collapsed():
    text = "The cat sat on the mat.   It was a very sunny day!\nWhy not go outside today? Because it rained later."
    assert split_sentences(text) == [
        "The cat sat on the mat.", "It was a very sunny day!", "Why not go outside today?", "Because it rained later."]


def test_split_sentences_glues_short_fragments_to_a_neighbour():
    out = split_sentences("See Fig. 3 for the full results of this experiment. Done here.")
    assert all(len(s) >= 20 for s in out)
    assert " ".join(out) == "See Fig. 3 for the full results of this experiment. Done here."


def test_split_sentences_never_loses_text():
    text = "Hi. Ok. A much longer sentence follows here for sure. Bye."
    assert " ".join(split_sentences(text)) == text
    assert split_sentences("short") == ["short"]


def test_bleu_identical_is_100_and_disjoint_is_near_zero():
    refs = ["the quick brown fox jumps over the lazy dog today", "machine written text looks very likely"]
    assert bleu(refs, refs) == pytest.approx(100.0)
    other = ["completely different words appear in this sentence now", "nothing shared with the reference at all here"]
    assert bleu(refs, other) < 5
    assert bleu(refs, ["", ""]) == 0.0


def test_bleu_partial_overlap_is_in_between():
    ref = ["the model was trained on a large corpus of text from the web"]
    hyp = ["the model was trained on a huge collection of text from the internet"]
    assert 5 < bleu(ref, hyp) < 95


def _orig():
    return pd.DataFrame({
        "id": ["arxiv_chatGPT_0001_m", "wikihow_cohere_0002_m"], "text": ["a  b", "c d"], "label": [1, 1],
        "domain": ["arxiv", "wikihow"], "generator": ["chatGPT", "cohere"], "group": ["g1", "g2"]})


def test_make_rows_schema_ids_and_normalised_text():
    rows = make_rows(_orig(), ["new   text one", "new text two"])
    assert list(rows.columns) == ["id", "text", "label", "domain", "generator", "group", "orig_id"]
    assert rows["id"].tolist() == ["arxiv_chatGPT_0001_m_para", "wikihow_cohere_0002_m_para"]
    assert rows["orig_id"].tolist() == ["arxiv_chatGPT_0001_m", "wikihow_cohere_0002_m"]
    assert rows["text"].tolist() == ["new text one", "new text two"]       # whitespace collapsed (§0.1)
    assert rows["group"].tolist() == ["g1", "g2"] and rows["label"].tolist() == [1, 1]


def test_build_splits_paraphrase_swaps_machines_keeps_val_and_humans():
    orig = {"train": [], "val": ["v1", "v2"], "test": ["m1", "m2", "h1", "h2"]}
    para, human = build_splits(orig, ["m1", "m2"], ["h1", "h2"], [])
    assert human is None
    assert para["val"] == ["v1", "v2"] and para["train"] == []
    assert para["test"] == sorted(["h1", "h2", "m1_para", "m2_para"])


def test_build_splits_human_control_keeps_machine_originals_and_swaps_some_humans():
    orig = {"train": [], "val": ["v1"], "test": ["m1", "m2", "h1", "h2", "h3"]}
    para, human = build_splits(orig, ["m1", "m2"], ["h1", "h2", "h3"], ["h2"])
    assert human["test"] == sorted(["m1", "m2", "h1", "h3", "h2_para"])
    assert len(human["test"]) == len(orig["test"])                          # same size as paraphrase_orig
    assert human["val"] == ["v1"]


def test_chunk_long_splits_only_long_sentences_and_keeps_every_word():
    long = " ".join(f"w{i}" for i in range(130))
    out = chunk_long(["short one.", long], max_words=60)
    assert out[0] == "short one." and [len(c.split()) for c in out[1:]] == [60, 60, 10]
    assert " ".join(out) == "short one. " + long
