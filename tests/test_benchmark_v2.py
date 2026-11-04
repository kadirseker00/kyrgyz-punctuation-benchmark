import csv
import json

import pytest

from kg_punctuation import benchmark_v2 as bv


def _doc(doc_id, words, labels):
    return {"document_id": doc_id, "tokens": words, "labels": labels}


def _sentence(prefix, n=6, end="P"):
    return [f"{prefix}{i}" for i in range(n)], ["S"] * (n - 1) + [end]


def test_project_core_maps_question_and_exclamation_to_period():
    assert bv.project(["S", "C", "P", "Q", "E"], "core") == ["S", "C", "P", "P", "P"]
    assert bv.project(["Q", "E"], "extended") == ["Q", "E"]


def test_deduplicate_drops_exact_copies_and_groups_near_duplicates():
    a_words, a_labels = _sentence("a")
    b_words, b_labels = _sentence("b")
    c_words, c_labels = _sentence("c")
    docs = [
        _doc("d1", a_words + b_words, a_labels + b_labels),
        _doc("d0", a_words + b_words, a_labels + b_labels),  # exact copy, smaller id is kept
        _doc("d2", a_words + c_words, a_labels + c_labels),  # shares half of its tokens with d0
        _doc("d3", c_words, c_labels),                        # fully contained in d2
        _doc("d4", ["x"] * 6 + b_words[:3], ["S"] * 9),       # no complete shared sentence
    ]
    kept, dropped, groups = bv.deduplicate(docs)
    assert dropped == ["d1"]
    assert sorted(d["document_id"] for d in kept) == ["d0", "d2", "d3", "d4"]
    assert sorted(sorted(g) for g in groups.values()) == [["d0", "d2", "d3"], ["d4"]]


def test_assign_splits_never_splits_a_group(monkeypatch):
    monkeypatch.setattr(bv, "SPLIT_SIZES", {"test": 3, "validation": 2})
    groups = {f"g{i}": [f"g{i}_{j}" for j in range(1 + i % 3)] for i in range(10)}
    splits = bv.assign_splits(groups, seed=7)
    seen = {}
    for name, ids in splits.items():
        for doc_id in ids:
            seen.setdefault(doc_id.split("_")[0], set()).add(name)
    assert all(len(names) == 1 for names in seen.values())
    assert sum(map(len, splits.values())) == sum(map(len, groups.values()))


@pytest.mark.parametrize("counts,budget", [([1] * 50, 8), ([3, 1, 4, 1, 5, 2, 2, 6], 7), ([2], 2), ([1, 1, 1], 10)])
def test_windows_cover_every_word_within_budget(counts, budget):
    windows = bv.word_windows(counts, budget)
    assert windows[0][0] == 0 and windows[-1][1] == len(counts)
    assert all(sum(counts[s:e]) <= budget for s, e in windows)
    owner = bv.owner_windows(windows, len(counts))
    assert all(windows[w][0] <= i < windows[w][1] for i, w in enumerate(owner))


def test_window_rejects_word_longer_than_budget():
    with pytest.raises(ValueError):
        bv.word_windows([1, 9, 1], 8)


class _FakeEncoding(dict):
    def __init__(self, input_ids, word_ids):
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


class _FakeTokenizer:
    """Word w -> len(w) sub-tokens; ids encode (word, piece) so the fake model can label them."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, words, add_special_tokens=True, is_split_into_words=False, **kwargs):
        import torch

        if not is_split_into_words:
            return {"input_ids": [[0] * len(w) for w in words]}
        if words and isinstance(words[0], str):  # single sequence, no tensors (training path)
            encoding = self([words], is_split_into_words=True)
            ids = [i for i, w in zip(encoding["input_ids"][0].tolist(), encoding.word_ids(0))]
            single = _FakeEncoding(ids, [encoding.word_ids(0)])
            single["attention_mask"] = [1] * len(ids)
            return single
        batch_ids, batch_wids = [], []
        for seq in words:
            ids, wids = [0], [None]
            for wi, word in enumerate(seq):
                for piece in range(len(word)):
                    ids.append(100 + (1 if piece == len(word) - 1 else 0))
                    wids.append(wi)
            ids.append(0)
            wids.append(None)
            batch_ids.append(ids)
            batch_wids.append(wids)
        width = max(map(len, batch_ids))
        padded = [ids + [0] * (width - len(ids)) for ids in batch_ids]
        wids = [w + [None] * (width - len(w)) for w in batch_wids]
        return _FakeEncoding(torch.tensor(padded), wids)


class _FakeModel:
    """Predicts label 1 on a word's last sub-token and label 0 elsewhere."""

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        import torch

        logits = torch.zeros(*input_ids.shape, 2)
        logits[..., 0] = (input_ids != 101).float()
        logits[..., 1] = (input_ids == 101).float()
        return type("Out", (), {"logits": logits})()


def test_predict_words_covers_all_words_and_honours_subtoken_policy():
    docs = [{"tokens": ["ab", "c", "defg", "hi", "j", "kl", "mno"] * 6}, {"tokens": ["x"]}]
    id2label = {0: "S", 1: "C"}
    last = bv.predict_words(_FakeModel(), _FakeTokenizer(), docs, id2label, subtoken="last", max_length=10, batch_size=3)
    first = bv.predict_words(_FakeModel(), _FakeTokenizer(), docs, id2label, subtoken="first", max_length=10, batch_size=3)
    assert [len(p) for p in last] == [len(d["tokens"]) for d in docs]
    assert all(label == "C" for doc in last for label in doc)
    # the first sub-token is the last one only for one-character words
    assert first[0] == ["C" if len(w) == 1 else "S" for w in docs[0]["tokens"]]


def test_score_asserts_manifest_order(tmp_path, monkeypatch):
    data = tmp_path / "src.jsonl"
    rows = [_doc(f"d{i}", ["w1", "w2", "w3"], ["S", "C", "Q"]) for i in range(3)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(bv, "ROOT", tmp_path)
    manifest = {"sources": {"src": {"data_file": "src.jsonl", "sha256": bv._sha256(data),
                                    "splits": {"test": ["d2", "d0"], "validation": [], "train": ["d1"]}}}}
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bench = bv.Benchmark(manifest_path)
    docs = bench.documents("src", "test", "core")
    assert [d["labels"] for d in docs] == [["S", "C", "P"]] * 2
    good = tmp_path / "good.csv"
    bv.write_predictions(good, docs, [["S", "C", "P"], ["S", "S", "P"]])
    result = bv.score(good, bench, "src")
    assert result["words"] == 6 and result["per_class"]["C"]["recall"] == 0.5
    bad = tmp_path / "bad.csv"
    with good.open() as src, bad.open("w", newline="") as dst:
        lines = list(csv.reader(src))
        lines[1], lines[4] = lines[4], lines[1]
        csv.writer(dst).writerows(lines)
    with pytest.raises(AssertionError):
        bv.score(bad, bench, "src")


def test_training_windows_label_every_word_exactly_once():
    docs = [{"tokens": ["ab", "c", "defg", "hi", "j", "kl", "mno"] * 6, "labels": ["S", "C", "P", "S", "S", "C", "P"] * 6}]
    features = bv.encode_training_windows(_FakeTokenizer(), docs, {"S": 0, "C": 1, "P": 2}, max_length=10)
    assert len(features) > 1 and all(len(f["input_ids"]) <= 10 for f in features)
    labelled = [lab for f in features for lab in f["labels"] if lab != -100]
    assert len(labelled) == len(docs[0]["tokens"])
    assert sorted(labelled) == sorted({"S": 0, "C": 1, "P": 2}[l] for l in docs[0]["labels"])


def test_predictions_carry_group_ids(tmp_path):
    docs = [{"source": "src", "document_id": "d1", "group_id": "src:g7", "tokens": ["a", "b"], "labels": ["S", "P"]}]
    path = tmp_path / "p.csv"
    bv.write_predictions(path, docs, [["S", "P"]])
    rows = list(csv.DictReader(path.open()))
    assert [r["group_id"] for r in rows] == ["src:g7", "src:g7"]


def test_overlong_words_are_cut_to_fit_and_still_predicted():
    long_word = "x" * 20  # 20 sub-tokens with the fake tokenizer, budget is 8
    docs = [{"tokens": ["ab", long_word, "c"], "labels": ["S", "C", "P"]}]
    words, counts = bv.fit_long_words(_FakeTokenizer(), docs[0]["tokens"], 8)
    assert words == ["ab", "x" * 8, "c"] and counts == [2, 8, 1]
    preds = bv.predict_words(_FakeModel(), _FakeTokenizer(), docs, {0: "S", 1: "C"}, subtoken="last", max_length=10)
    assert preds == [["C", "C", "C"]]
    features = bv.encode_training_windows(_FakeTokenizer(), docs, {"S": 0, "C": 1, "P": 2}, max_length=10)
    assert sum(lab != -100 for f in features for lab in f["labels"]) == 3
