import numpy as np
import pytest

from kg_punctuation.rare_calibration import LABELS, class_metrics, predict, select_offsets


def test_validation_offsets_remove_spurious_marks_and_keep_true_rare_marks():
    logits = np.array([[2., 0., 0., 3., 3.], [0., 0., 0., 5., 0.], [0., 0., 0., 0., 5.]])
    gold = np.array([0, 3, 4])
    fitted = select_offsets(logits, gold, grid=(0., 2.))
    assert fitted["selected"]["q_penalty"] == 2.
    assert fitted["selected"]["e_penalty"] == 2.
    assert predict(logits, fitted).tolist() == gold.tolist()
    # Applying the frozen rule to different inputs does not fit it again.
    assert predict(np.array([[2., 0., 0., 10., 0.]]), fitted).tolist() == [3]
    assert fitted["selection_split"] == "validation"


def test_equal_validation_objectives_prefer_no_adjustment():
    fitted = select_offsets(np.array([[10., 0., 0., 0., 0.]]), np.array([0]))
    assert fitted["selected"]["q_penalty"] == fitted["selected"]["e_penalty"] == 0.
    assert class_metrics(np.array([0]), np.array([0]))["macro_f1"] == pytest.approx(.2)


def test_nonfinite_validation_scores_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        select_offsets(np.full((1, 5), np.nan), np.array([0]))
