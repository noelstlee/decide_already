import json
import random

import pytest

from decide_already.questions import FITS_ANY_QUESTION, GROUP_QUESTION, build_questions
from decide_already.types import Group

GROUPS = [
    Group(id="hw", name="HW", covers="homework specs, starter code", excludes="project files"),
    Group(id="logistics", name="Logistics", covers="syllabus, schedule, deadlines"),
    Group(id="project", name="Project", covers="project specs, examples", excludes="deadlines"),
    Group(id="lecture-notes", name="Lecture slides and notes", covers="lecture slides and notes"),
    Group(id="exams", name="Exams", covers="quizzes, midterms and finals"),
    Group(id="readings", name="Readings", covers="assigned papers and book chapters"),
]
ID_BY_NAME = {group.name: group.id for group in GROUPS}


def test_two_questions_with_fixed_ids_and_types():
    questions, _ = build_questions(GROUPS, random.Random(0))
    assert (GROUP_QUESTION, FITS_ANY_QUESTION) == ("group", "fits_any")
    assert list(questions) == ["group", "fits_any"]
    assert questions["group"]["type"] == "choice"
    assert questions["fits_any"]["type"] == "noul"


def test_the_options_are_exactly_the_groups_described_under_their_names():
    questions, order = build_questions(GROUPS, random.Random(0))
    group = questions["group"]
    assert group["instructions"] == "Which group does this document belong in?"
    assert set(group["criteria"]) == set(ID_BY_NAME)  # so there is no "none of these" option
    assert len(order) == len(GROUPS)
    assert group["criteria"]["HW"] == {
        "covers": "homework specs, starter code",
        "excludes": "project files",
    }


def test_a_group_without_excludes_has_no_excludes_key():
    questions, _ = build_questions(GROUPS, random.Random(0))
    assert questions["group"]["criteria"]["Logistics"] == {
        "covers": "syllabus, schedule, deadlines"
    }
    blank = [Group(id="a", name="A", covers="things", excludes="")]
    questions, _ = build_questions(blank, random.Random(0))
    assert questions["group"]["criteria"]["A"] == {"covers": "things"}


def test_option_order_is_the_order_of_the_sent_criteria():
    questions, order = build_questions(GROUPS, random.Random(3))
    assert [ID_BY_NAME[name] for name in questions["group"]["criteria"]] == order
    assert sorted(order) == sorted(ID_BY_NAME.values())


def test_the_order_survives_the_trip_through_json():
    questions, order = build_questions(GROUPS, random.Random(3))
    sent = json.loads(json.dumps(questions))
    assert sent == questions
    assert [ID_BY_NAME[name] for name in sent["group"]["criteria"]] == order


def test_one_seed_gives_one_order():
    first = build_questions(GROUPS, random.Random(11))
    second = build_questions(GROUPS, random.Random(11))
    assert first == second


def test_different_seeds_give_different_orders():
    _, first = build_questions(GROUPS, random.Random(1))
    _, second = build_questions(GROUPS, random.Random(2))
    assert first != second


def test_every_call_shuffles_again():
    rng = random.Random(5)
    orders = {tuple(build_questions(GROUPS, rng)[1]) for _ in range(10)}
    assert len(orders) > 1


def test_the_callers_list_is_left_alone():
    groups = list(GROUPS)
    build_questions(groups, random.Random(4))
    assert groups == GROUPS


def test_fits_any_carries_the_groups_itself_in_the_same_order():
    questions, _ = build_questions(GROUPS, random.Random(3))
    fits_any = questions["fits_any"]
    assert fits_any["instructions"]["statement"] == (
        "This document clearly belongs in one of these groups."
    )
    assert fits_any["instructions"]["groups"] == questions["group"]["criteria"]
    assert list(fits_any["instructions"]["groups"]) == list(questions["group"]["criteria"])
    assert fits_any["criteria"] == {
        "true": "The document's content matches what one of the groups covers.",
        "false": "The document is about something that none of the groups cover.",
    }


def test_names_only_sends_no_descriptions():
    questions, order = build_questions(GROUPS, random.Random(3), with_descriptions=False)
    criteria = questions["group"]["criteria"]
    assert [ID_BY_NAME[name] for name in criteria] == order
    assert all(value is None for value in criteria.values())
    assert questions["fits_any"]["instructions"]["groups"] == list(criteria)
    sent = json.dumps(questions)
    assert "covers" not in json.dumps(questions["group"])
    assert all(group.covers not in sent for group in GROUPS)


def test_names_only_uses_the_same_shuffle_as_the_full_questions():
    _, described = build_questions(GROUPS, random.Random(8))
    _, names_only = build_questions(GROUPS, random.Random(8), with_descriptions=False)
    assert described == names_only


def test_one_group_is_enough():
    only = [GROUPS[0]]
    questions, order = build_questions(only, random.Random(0))
    assert order == ["hw"]
    assert list(questions["group"]["criteria"]) == ["HW"]


def test_no_groups_is_an_error():
    with pytest.raises(ValueError, match="no groups"):
        build_questions([], random.Random(0))


def test_two_groups_with_one_name_is_an_error():
    twins = [Group(id="a", name="Same", covers="x"), Group(id="b", name="Same", covers="y")]
    with pytest.raises(ValueError, match="share a name"):
        build_questions(twins, random.Random(0))
