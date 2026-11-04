from collections import Counter
import csv

import pytest
from sklearn.metrics import f1_score

from kg_punctuation.overlap_sensitivity import build_mask, metrics, rescore


def document():
    return {"source": "tiny", "document_id": "d", "group_id": "g",
            "tokens": [str(i) for i in range(15)],
            "labels": ["S"] * 5 + ["Q"] + ["S"] * 3 + ["P"] + ["S"] * 5}


def test_mask_excludes_complete_sentences_but_keeps_short_and_unfinished_parts():
    doc = document()
    repeated = {tuple(doc["tokens"][:6]), tuple(doc["tokens"][6:10]), tuple(doc["tokens"][10:])}
    mask = build_mask([doc], repeated, set())
    assert [r["word_index"] for r in mask if r["keep"]] == [str(i) for i in range(6, 15)]
    assert mask[5]["gold"] == "P"
    assert not any(r["own_source_overlap"] for r in mask)


def test_macro_f1_uses_pooled_counts_and_retains_absent_inventory_classes():
    gold = ["S", "S", "C", "C", "S"]
    predicted = ["S", "C", "C", "S", "S"]
    got = metrics(Counter(zip(gold, predicted)))
    assert got["macro_f1"] == pytest.approx(f1_score(gold, predicted, labels=["S", "C", "P"], average="macro", zero_division=0))
    assert got["per_class"]["P"]["f1"] == 0


@pytest.mark.parametrize("change", ["reorder", "wrong_group", "missing_tail", "wrong_gold"])
def test_rescore_rejects_identity_and_coverage_errors(tmp_path, change):
    mask = build_mask([document()], set(), set())
    rows = [{k: r[k] for k in ("source", "document_id", "group_id", "word_index", "gold")} for r in mask]
    for r in rows:
        r["prediction"] = r["gold"]
    if change == "reorder": rows[0], rows[1] = rows[1], rows[0]
    elif change == "wrong_group": rows[0]["group_id"] = "other"
    elif change == "missing_tail": rows.pop()
    else: rows[0]["gold"] = "C"
    path = tmp_path / "pred.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="canonical mask"):
        rescore(path, mask)
