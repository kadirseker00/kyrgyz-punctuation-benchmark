from kg_punctuation.data.labeling import tokens_and_labels


def test_tokens_and_labels_extracts_space_comma_period() -> None:
    tokens, labels = tokens_and_labels("Бул тест, жакшы иштейт. Экинчи сүйлөм")

    assert tokens == ["бул", "тест", "жакшы", "иштейт", "экинчи", "сүйлөм"]
    assert labels == ["S", "C", "S", "P", "S", "S"]


def test_extended_labels_keep_question_and_exclamation_distinct() -> None:
    tokens, labels = tokens_and_labels(
        "Бул сурообу? Ооба! Бул тест.",
        label_set="extended",
    )

    assert tokens == ["бул", "сурообу", "ооба", "бул", "тест"]
    assert labels == ["S", "Q", "E", "S", "P"]


def test_core_labels_preserve_backward_compatibility_for_question_marks() -> None:
    tokens, labels = tokens_and_labels("Бул сурообу? Ооба!", label_set="core")

    assert tokens == ["бул", "сурообу", "ооба"]
    assert labels == ["S", "S", "S"]
