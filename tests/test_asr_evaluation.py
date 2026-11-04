import pytest
from kg_punctuation.asr_evaluation import align_words, event_counts, event_metrics


def test_alignment_covers_reference_and_insertions_without_dropping_errors():
    ref, hyp = ["a", "b", "c"], ["a", "x", "b", "d"]
    pairs, distance = align_words(ref, hyp)
    assert [i for i, _ in pairs if i is not None] == [0, 1, 2]
    assert [j for _, j in pairs if j is not None] == [0, 1, 2, 3]
    assert distance == 2


def test_deleted_period_is_false_negative_and_inserted_comma_is_false_positive():
    deleted = event_counts(["a", "b"], ["S", "P"], ["a"], ["S"])
    inserted = event_counts(["a"], ["S"], ["a", "b"], ["S", "C"])
    assert deleted["P_fn"] == 1
    assert inserted["C_fp"] == 1
    assert event_metrics(deleted)["per_class"]["P"]["support"] == 1


def test_substituted_anchor_policy_is_explicit_and_changes_joint_score():
    args = (["a"], ["P"], ["b"], ["P"])
    aligned = event_counts(*args)
    anchored = event_counts(*args, require_exact_anchor=True)
    assert aligned["P_tp"] == 1
    assert anchored["P_tp"] == 0
    assert anchored["P_fp"] == anchored["P_fn"] == 1
    assert anchored["reference_words"] == 1 and anchored["matched_words"] == 0


def test_punctuation_inventory_does_not_reward_no_mark_gaps():
    counts = event_counts(["a"], ["S"], [], [])
    assert event_metrics(counts)["punctuation_macro_f1"] == 0
    assert event_metrics(counts)["WER"] == 1


def test_missing_word_labels_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        event_counts(["a"], [], ["a"], ["S"])
