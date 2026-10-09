import pytest

from evaluation.metrics import (
    abstention_confusion,
    coverage_at_k,
    hit_at_k,
    mean,
    ranked_pages,
    reciprocal_rank,
    rubric_passes,
)


def test_ranked_pages_counts_each_page_once_in_first_seen_order():
    assert ranked_pages([4, 4, 5, 4, 2], k=4) == [4, 5]
    assert ranked_pages([4, 4, 5, 4, 2], k=5) == [4, 5, 2]


def test_hit_at_k_needs_any_relevant_page_within_k():
    chunk_pages = [3, 3, 7, 1]

    assert hit_at_k(chunk_pages, {7}, k=3) is True
    assert hit_at_k(chunk_pages, {7}, k=2) is False
    assert hit_at_k(chunk_pages, {1, 7}, k=2) is False


def test_coverage_is_the_fraction_of_required_pages_found():
    chunk_pages = [4, 4, 4, 5]

    assert coverage_at_k(chunk_pages, {4, 5}, k=3) == 0.5
    assert coverage_at_k(chunk_pages, {4, 5}, k=4) == 1.0
    assert coverage_at_k(chunk_pages, {9}, k=4) == 0.0


def test_duplicate_chunks_from_one_page_do_not_inflate_coverage():
    assert coverage_at_k([4, 4, 4, 4], {4, 5}, k=4) == 0.5


def test_reciprocal_rank_uses_the_first_chunk_on_a_relevant_page():
    assert reciprocal_rank([3, 8, 5, 8], {8}) == 0.5
    assert reciprocal_rank([8, 3], {8, 3}) == 1.0
    assert reciprocal_rank([3, 4], {9}) == 0.0


def test_mean_of_nothing_is_undefined_not_zero():
    assert mean([]) is None
    assert mean([1.0, 0.0, 0.5]) == 0.5


@pytest.mark.parametrize(
    ("answer", "criteria", "expected"),
    [
        ("The firm employed 3,120 people.", [["3,120", "3120"]], True),
        ("About 3120 staff", [["3,120", "3120"]], True),
        ("South Basin, 1.4 million TEU", [["south basin"], ["1.4"]], True),
        ("South Basin, capacity unknown", [["south basin"], ["1.4"]], False),
        ("anything", [], True),
    ],
)
def test_rubric_needs_every_group_to_match_one_alternative_ignoring_case(answer, criteria, expected):
    assert rubric_passes(answer, criteria) is expected


def test_abstention_confusion_treats_abstaining_as_the_positive_class():
    rows = [
        {"answerable": False, "abstained": True},  # true positive
        {"answerable": False, "abstained": True},  # true positive
        {"answerable": False, "abstained": False},  # false negative: invented an answer
        {"answerable": True, "abstained": True},  # false positive: refused a good question
        {"answerable": True, "abstained": False},  # true negative
    ]

    result = abstention_confusion(rows)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)


def test_precision_and_recall_are_none_when_undefined():
    only_answerable = [{"answerable": True, "abstained": False}]

    result = abstention_confusion(only_answerable)

    assert result["precision"] is None  # nothing abstained
    assert result["recall"] is None  # nothing was unanswerable


@pytest.mark.parametrize(
    ("answer", "criteria"),
    [
        ("about 9\u202f% of containers", [["9%"]]),  # narrow no-break space before the sign
        ("about 9 % of containers", [["9%"]]),
        ("the co\u2011founded firm", [["co-founded"]]),  # non-breaking hyphen
        ("South\u00a0Basin", [["south basin"]]),  # no-break space
        ("a \u2013 b", [["a - b"]]),  # en dash
    ],
)
def test_rubric_ignores_typographic_variants_models_commonly_emit(answer, criteria):
    assert rubric_passes(answer, criteria) is True
