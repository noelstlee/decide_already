"""The card: its fields, its size limit, the five variants, and the card note body."""

import hashlib
import json
from itertools import pairwise

import pymupdf
import pytest
from pptx import Presentation

from decide_already.card import VARIANTS, build_card, estimate_tokens, hash_file, render_card
from decide_already.config import Config

BUDGET = Config().card_token_budget


def words(count):
    """A text of numbered words, so that any passage cut from it can be found again."""
    return " ".join(f"word{number}" for number in range(count))


def write(path, content):
    """Write a text fixture and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


def state_tokens(state):
    """The size estimate of one state: 4 characters per token over the JSON that is sent."""
    return estimate_tokens(json.dumps(state, ensure_ascii=False))


def spans(text, state):
    """Where the opening and each excerpt of a card state sit in the text, as (start, end)."""
    passages = [state["opening"], *state.get("excerpts", {}).values()]
    found = [(text.index(passage), text.index(passage) + len(passage)) for passage in passages]
    assert all(text.count(passage) == 1 for passage in passages)
    return found


# The two small helpers


def test_estimate_tokens_is_four_characters_per_token_rounded_up():
    assert [estimate_tokens("x" * size) for size in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]


def test_hash_file_is_stable_and_changes_with_content(tmp_path):
    path = write(tmp_path / "a.txt", "first version")
    first = hash_file(path)
    assert first == hash_file(path)
    assert first == hashlib.sha256(b"first version").hexdigest()
    write(path, "second version")
    assert hash_file(path) != first
    assert hash_file(write(tmp_path / "b.txt", "first version")) == first


# The default card


def test_card_has_named_fields_in_order(tmp_path):
    source = "# Homework 3: Kernel Methods\n\n## Problem 1\n\nDerive the dual.\n"
    card = build_card(write(tmp_path / "hw3.md", source), "hw3/hw3.md", Config())
    assert card.title == "Homework 3: Kernel Methods"
    assert card.states == [
        {
            "file": {"name": "hw3.md", "path": "hw3/hw3.md", "type": "Markdown"},
            "title": "Homework 3: Kernel Methods",
            "outline": ["Homework 3: Kernel Methods", "Problem 1"],
            "opening": source.strip(),
        }
    ]
    assert list(card.states[0]) == ["file", "title", "outline", "opening"]


def test_empty_fields_are_left_out_of_the_state(tmp_path):
    card = build_card(write(tmp_path / "n.txt", "42 17\n3.14\n"), "n.txt", Config())
    assert card.title is None
    assert card.states == [
        {"file": {"name": "n.txt", "path": "n.txt", "type": "Text"}, "opening": "42 17\n3.14"}
    ]


def test_card_of_a_notebook_a_pdf_and_a_deck(tmp_path):
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": "# Copyright Information"},
        {"cell_type": "markdown", "metadata": {}, "source": "# Lab 2: Computer Vision"},
        {"cell_type": "code", "metadata": {}, "source": "def train():\n    pass", "outputs": []},
    ]
    notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    write(tmp_path / "lab2.ipynb", json.dumps(notebook))
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Problem Set 1\nDue Thursday at noon.")
    document.set_metadata({"title": "Problem Set 1"})
    document.set_toc([[1, "Problem 1", 1]])
    document.save(tmp_path / "hw1.pdf")
    document.close()
    deck = Presentation()
    for title in ("Lecture 9: Kernels", "The kernel trick"):
        deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = title
    deck.save(str(tmp_path / "lec9.pptx"))
    expected = {  # file name: type, title, outline, opening
        "lab2.ipynb": (
            "Jupyter notebook, 3 cells",
            "Lab 2: Computer Vision",  # not the license heading that comes first
            ["Copyright Information", "Lab 2: Computer Vision", "train()"],
            "# Copyright Information\n\n# Lab 2: Computer Vision\n\ndef train():\n    pass",
        ),
        "hw1.pdf": (
            "PDF, 1 page",
            "Problem Set 1",
            ["Problem 1"],
            "Problem Set 1\nDue Thursday at noon.",
        ),
        "lec9.pptx": (
            "Slides, 2",
            "Lecture 9: Kernels",
            ["Lecture 9: Kernels", "The kernel trick"],
            "Lecture 9: Kernels\n\nThe kernel trick",
        ),
    }
    for name, (file_type, title, outline, opening) in expected.items():
        card = build_card(tmp_path / name, f"week1/{name}", Config())
        assert card.title == title
        assert card.file.file_type == file_type
        assert card.states == [
            {
                "file": {"name": name, "path": f"week1/{name}", "type": file_type},
                "title": title,
                "outline": outline,
                "opening": opening,
            }
        ]


def test_file_record_carries_id_hash_type_and_size(tmp_path):
    path = write(tmp_path / "notes.txt", "Office hours are on Monday.\n")
    card = build_card(path, "week1/notes.txt", Config())
    assert card.file.id == "week1/notes.txt"
    assert card.file.content_hash == hash_file(path)
    assert card.file.file_type == "Text"
    assert card.file.card_tokens == state_tokens(card.states[0])


def test_opening_is_the_first_6000_characters_cut_between_words(tmp_path):
    text = words(5_000)
    state = build_card(write(tmp_path / "long.txt", text), "long.txt", Config()).states[0]
    opening = state["opening"]
    assert 5_990 <= len(opening) <= 6_000
    assert text.startswith(opening + " ")


def test_short_text_has_no_excerpts(tmp_path):
    text = words(600)
    state = build_card(write(tmp_path / "short.txt", text), "short.txt", Config()).states[0]
    assert state["opening"] == text
    assert "excerpts" not in state


def test_long_text_has_three_excerpts_that_do_not_overlap(tmp_path):
    text = words(10_000)
    state = build_card(write(tmp_path / "long.txt", text), "long.txt", Config()).states[0]
    assert list(state["excerpts"]) == ["at_40_percent", "at_70_percent", "at_95_percent"]
    assert all(0 < len(passage) <= 1_200 for passage in state["excerpts"].values())
    found = spans(text, state)
    assert found[0][0] == 0
    for (_, end), (start, _) in pairwise(found):
        assert end <= start
    for (start, _), mark in zip(found[1:], (40, 70, 95)):
        assert start == pytest.approx(len(text) * mark / 100, abs=10)


def test_an_excerpt_that_would_start_inside_the_opening_is_left_out(tmp_path):
    text = words(1_300)  # about 10,000 characters: the 40 percent mark falls inside the opening
    state = build_card(write(tmp_path / "mid.txt", text), "mid.txt", Config()).states[0]
    assert list(state["excerpts"]) == ["at_70_percent", "at_95_percent"]
    found = spans(text, state)
    for (_, end), (start, _) in pairwise(found):
        assert end <= start


def test_excerpts_stop_where_the_next_one_begins(tmp_path):
    text = words(5_000)
    config = Config(card_token_budget=2_000)  # the opening gets shorter; the excerpts stay apart
    state = build_card(write(tmp_path / "tight.txt", text), "tight.txt", config).states[0]
    found = spans(text, state)
    for (_, end), (start, _) in pairwise(found):
        assert end <= start
    assert state_tokens(state) <= 2_000


# The size limit


def test_huge_file_with_a_long_outline_stays_under_budget(tmp_path):
    sections = [
        f"## Section {number} " + "heading " * 20 + f"\n\n{words(1_500)}" for number in range(60)
    ]
    source = "# A very large file\n\n" + "\n\n".join(sections)
    assert len(source) > 700_000
    state = build_card(write(tmp_path / "huge.md", source), "huge.md", Config()).states[0]
    assert state_tokens(state) <= BUDGET
    assert set(state) == {"file", "title", "outline", "opening", "excerpts"}


def test_outline_is_shortened_before_the_opening(tmp_path):
    headings = "\n\n".join(f"## Section {number} " + "heading " * 20 for number in range(60))
    source = f"{headings}\n\n{words(10_000)}"
    state = build_card(write(tmp_path / "many.md", source), "many.md", Config()).states[0]
    assert state_tokens(state) <= BUDGET
    assert 0 < len(state["outline"]) < 40
    assert state["outline"][0].startswith("Section 0 ")
    assert len(state["opening"]) > 5_900
    assert len(state["excerpts"]) == 3


def test_opening_is_shortened_when_the_text_grows_in_json(tmp_path):
    source = 'say "yes"\n' * 4_000  # quotes and newlines take two characters each in JSON
    state = build_card(write(tmp_path / "quotes.txt", source), "quotes.txt", Config()).states[0]
    assert state_tokens(state) <= BUDGET
    assert 1_000 < len(state["opening"]) < 6_000
    assert source.startswith(state["opening"])
    assert len(state["excerpts"]) == 3


@pytest.mark.parametrize("budget", [2_000, 1_000, 400, 100])
def test_a_smaller_budget_is_kept_too(tmp_path, budget):
    source = "# Title\n\n" + "\n\n".join(f"## Part {number}" for number in range(30)) + words(9_000)
    config = Config(card_token_budget=budget)
    state = build_card(write(tmp_path / "small.md", source), "small.md", config).states[0]
    assert state_tokens(state) <= budget
    assert state["file"] == {"name": "small.md", "path": "small.md", "type": "Markdown"}
    assert all(state.values())


def test_every_kind_of_real_text_stays_under_budget(tmp_path):
    sources = {
        "code.py": "def function_%d(argument):\n    return argument * %d\n\n\n",
        "page.html": "<h2>Heading %d</h2><p>Paragraph %d with some words in it.</p>\n",
        "data.csv": "%d,name,%d,value,more,columns\n",
        "paper.tex": "\\section{Part %d}\nSome text about item %d and $x^2$.\n\n",
    }
    for name, line in sources.items():
        source = "".join(line % (number, number) for number in range(3_000))
        card = build_card(write(tmp_path / name, source), name, Config())
        assert state_tokens(card.states[0]) <= BUDGET, name
        assert card.file.card_tokens <= BUDGET, name


# The variants


def test_variants_are_the_five_in_the_contract():
    assert VARIANTS == ("card", "card_no_name", "full_text", "window_vote", "names_only")


def test_card_no_name_drops_name_and_path_but_keeps_type(tmp_path):
    path = write(tmp_path / "hw3.md", f"# Homework 3\n\n{words(10_000)}")
    card = build_card(path, "hw3/hw3.md", Config(), "card").states[0]
    no_name = build_card(path, "hw3/hw3.md", Config(), "card_no_name").states[0]
    assert no_name["file"] == {"type": "Markdown"}
    assert {**no_name, "file": card["file"]} == card
    assert "hw3" not in json.dumps(no_name["file"])


def test_names_only_is_the_same_state_as_the_card(tmp_path):
    path = write(tmp_path / "hw3.md", f"# Homework 3\n\n{words(10_000)}")
    card = build_card(path, "hw3.md", Config(), "card")
    assert build_card(path, "hw3.md", Config(), "names_only") == card


def test_full_text_is_one_state_cut_at_30000_tokens(tmp_path):
    text = words(30_000)  # about 290,000 characters
    card = build_card(write(tmp_path / "book.txt", text), "book.txt", Config(), "full_text")
    assert len(card.states) == 1
    state = card.states[0]
    assert list(state) == ["file", "text"]
    assert state["file"] == {"name": "book.txt", "path": "book.txt", "type": "Text"}
    assert 29_990 <= estimate_tokens(state["text"]) <= 30_000
    assert text.startswith(state["text"] + " ")


def test_full_text_of_a_short_file_is_the_whole_text(tmp_path):
    card = build_card(write(tmp_path / "a.txt", "Short note.\n"), "a.txt", Config(), "full_text")
    assert card.states == [
        {"file": {"name": "a.txt", "path": "a.txt", "type": "Text"}, "text": "Short note."}
    ]


def test_window_vote_keeps_at_most_8_windows_with_the_first_and_the_last(tmp_path):
    text = words(30_000)  # about 37 windows of 2,000 tokens
    card = build_card(write(tmp_path / "book.txt", text), "book.txt", Config(), "window_vote")
    windows = [state["text"] for state in card.states]
    assert len(windows) == 8
    assert all(list(state) == ["file", "text"] for state in card.states)
    assert all(state["file"] == card.states[0]["file"] for state in card.states)
    assert all(estimate_tokens(window) <= 2_000 for window in windows)
    assert text.startswith(windows[0])
    assert text.endswith(windows[-1])
    starts = [text.index(window) for window in windows]
    assert starts == sorted(starts)
    gaps = [later - earlier for earlier, later in pairwise(starts)]
    assert max(gaps) - min(gaps) <= 8_000  # evenly spread over the text


def test_window_vote_of_a_mid_sized_text_covers_all_of_it(tmp_path):
    text = words(2_500)  # about 21,000 characters: three windows
    card = build_card(write(tmp_path / "mid.txt", text), "mid.txt", Config(), "window_vote")
    windows = [state["text"] for state in card.states]
    assert len(windows) == 3
    assert "".join(windows).replace(" ", "") == text.replace(" ", "")
    assert max(map(len, windows)) - min(map(len, windows)) <= 2  # equal shares, no short leftover


def test_window_vote_of_a_short_text_is_one_window(tmp_path):
    card = build_card(write(tmp_path / "a.txt", "Short note.\n"), "a.txt", Config(), "window_vote")
    assert [state["text"] for state in card.states] == ["Short note."]


def test_card_tokens_is_the_estimate_over_all_states(tmp_path):
    path = write(tmp_path / "book.txt", words(30_000))
    for variant in VARIANTS:
        card = build_card(path, "book.txt", Config(), variant)
        assert card.file.card_tokens == sum(state_tokens(state) for state in card.states)
    assert build_card(path, "book.txt", Config(), "window_vote").file.card_tokens > 8 * 1_900


@pytest.mark.parametrize("variant", VARIANTS)
def test_file_with_no_text_has_no_states(tmp_path, variant):
    document = pymupdf.open()
    document.new_page()
    document.set_metadata({"title": "Scanned lecture notes"})
    document.save(tmp_path / "scan.pdf")
    document.close()
    card = build_card(tmp_path / "scan.pdf", "scan.pdf", Config(), variant)
    assert card.states == []
    assert card.title == "Scanned lecture notes"
    assert card.file.card_tokens == 0
    assert card.file.file_type == "PDF, 1 page"
    assert card.file.content_hash == hash_file(tmp_path / "scan.pdf")


@pytest.mark.parametrize("variant", VARIANTS)
def test_binary_file_has_no_states(tmp_path, variant):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 30)
    card = build_card(tmp_path / "photo.png", "photo.png", Config(), variant)
    assert card.states == []
    assert card.title is None


@pytest.mark.parametrize("variant", VARIANTS)
def test_card_of_a_notebook_with_a_broken_emoji_escape_can_be_written_as_utf8(tmp_path, variant):
    content = (  # "\ud83d" alone is half a character: valid JSON that UTF-8 cannot hold
        '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": "# Lab \\ud83d one"}], '
        '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    card = build_card(write(tmp_path / "emoji.ipynb", content), "emoji.ipynb", Config(), variant)
    assert len(card.states) == 1
    assert card.title == "Lab \ufffd one"
    json.dumps(card.states, ensure_ascii=False).encode("utf-8")  # what is sent and logged
    render_card(card).encode("utf-8")  # what is written to the card note


def test_variant_defaults_to_the_one_in_the_config(tmp_path):
    path = write(tmp_path / "a.txt", "Short note.\n")
    assert "opening" in build_card(path, "a.txt", Config()).states[0]
    assert "text" in build_card(path, "a.txt", Config(variant="full_text")).states[0]
    assert "opening" in build_card(path, "a.txt", Config(variant="full_text"), "card").states[0]


def test_unknown_variant_is_an_error(tmp_path):
    path = write(tmp_path / "a.txt", "Short note.\n")
    with pytest.raises(ValueError, match="unknown variant 'summary'"):
        build_card(path, "a.txt", Config(), "summary")


def test_building_a_card_never_changes_the_file(tmp_path):
    sources = {"a.md": "# Title\n\ntext\r\n", "b.py": "def f():\n    pass\n", "c.html": "<p>hi</p>"}
    for name, source in sources.items():
        path = tmp_path / name
        path.write_bytes(source.encode())
        for variant in VARIANTS:
            build_card(path, name, Config(), variant)
        assert path.read_bytes() == source.encode()
    assert sorted(item.name for item in tmp_path.iterdir()) == sorted(sources)


# The card note body


def test_render_card_shows_name_type_title_outline_and_text(tmp_path):
    source = "# Homework 3: Kernel Methods\n\n## Problem 1\n\nDerive the dual.\n"
    card = build_card(write(tmp_path / "hw3.md", source), "hw3/hw3.md", Config())
    assert render_card(card) == (
        "# `hw3.md`\n"
        "\n"
        "- Type: Markdown\n"
        "- Title: `Homework 3: Kernel Methods`\n"
        "\n"
        "## Outline\n"
        "\n"
        "```text\n"
        "Homework 3: Kernel Methods\n"
        "Problem 1\n"
        "```\n"
        "\n"
        "## Opening\n"
        "\n"
        "```text\n"
        "# Homework 3: Kernel Methods\n"
        "\n"
        "## Problem 1\n"
        "\n"
        "Derive the dual.\n"
        "```\n"
    )


def test_render_card_shows_every_excerpt(tmp_path):
    card = build_card(write(tmp_path / "long.txt", words(10_000)), "long.txt", Config())
    body = render_card(card)
    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Opening",
        "## Excerpt at 40 percent",
        "## Excerpt at 70 percent",
        "## Excerpt at 95 percent",
    ]
    for passage in card.states[0]["excerpts"].values():
        assert passage in body


def test_render_card_keeps_file_text_inside_its_fence(tmp_path):
    source = (
        "Setup notes\n\n````md\n```python\nprint('hi')\n```\n````\n\n"
        "![x](https://example.com/x.png)\n"
    )
    card = build_card(write(tmp_path / "__init__.txt", source), "__init__.txt", Config())
    body = render_card(card)
    assert body.startswith("# `__init__.txt`\n")
    assert body.endswith("\n`````\n")
    assert body.count("`````") == 2  # the fence is longer than any run of backticks in the text
    assert source.strip() in body


def test_render_card_of_other_variants(tmp_path):
    path = write(tmp_path / "mid.txt", words(2_500))
    full = render_card(build_card(path, "mid.txt", Config(), "full_text"))
    assert [line for line in full.splitlines() if line.startswith("## ")] == ["## Text"]
    vote = render_card(build_card(path, "mid.txt", Config(), "window_vote"))
    assert [line for line in vote.splitlines() if line.startswith("## ")] == [
        "## Window 1 of 3",
        "## Window 2 of 3",
        "## Window 3 of 3",
    ]
    no_name = render_card(build_card(path, "mid.txt", Config(), "card_no_name"))
    assert no_name.startswith("# `mid.txt`\n\n- Type: Text\n")


def test_render_card_of_a_file_with_no_text(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 30)
    card = build_card(tmp_path / "photo.png", "figures/photo.png", Config())
    assert render_card(card) == (
        "# `photo.png`\n\n- Type: Binary file\n\nNo text could be extracted from this file.\n"
    )
