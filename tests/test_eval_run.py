"""eval/run.py and eval/report.py: the cache, the stops, the dry run and the report.

Every backend here is a stand-in. Nothing is asked of a real model and nothing uses the network.
Every expected number was worked out by hand from the answers in JEV below.
"""

import hashlib
import json
import threading
from collections import Counter

import httpx2
import pytest

from decide_already import backends as backends_module
from decide_already import vault
from decide_already.backends import jev as jev_module
from decide_already.backends.base import BackendError, BudgetExceeded, LogFailed
from decide_already.backends.jev import JevBackend
from decide_already.card import VARIANTS, build_card
from decide_already.config import JEV_LOG_FILE, Config, decide_path
from decide_already.eval import run
from decide_already.eval.baselines import llm
from decide_already.eval.baselines.embed import EmbeddingBackend
from decide_already.eval.baselines.rules import RulesBackend
from decide_already.eval.dataset import GoldLabel
from decide_already.eval.experiments import Edit
from decide_already.eval.metrics import Metrics, Prediction
from decide_already.eval.report import Answers, Inputs, build_metrics, render_report
from decide_already.log import append_jsonl, read_jsonl
from decide_already.types import Group
from fakes import FakeBackend

GROUP_IDS = ("exams", "homework", "lecture-notes")
NO_TEXT = "data/hw3_data.zip"

# (file id, content, gold group, split). The zip is all NUL bytes, so it has no text.
FILES = [
    (NO_TEXT, b"PK\x03\x04" + b"\x00" * 60, "homework", "test"),
    ("final.txt", "Final exam. Closed book, two hours.\n", "exams", "test"),
    ("hw1.txt", "Problem set 1. Derive the gradient of the loss.\n", "homework", "test"),
    ("hw2_soln.txt", "Solutions to problem set 2.\n", "homework", "test"),
    (
        "labs/lab1.py",
        '"""Lab 1 starter code."""\n\n\ndef train():\n    return 1\n',
        "homework",
        "test",
    ),
    ("lec1.md", "# Lecture 1\n\nLinear regression and least squares.\n", "lecture-notes", "test"),
    ("lec2.md", "# Lecture 2\n\nKernels and feature maps.\n", "lecture-notes", "test"),
    ("midterm.txt", "Midterm exam. One page of notes allowed.\n", "exams", "test"),
    ("hw0.txt", "Problem set 0. A warm-up.\n", "homework", "dev"),
    ("lec0.md", "# Lecture 0\n\nCourse overview.\n", "lecture-notes", "dev"),
    ("quiz0.txt", "Quiz 0. Ten short questions.\n", "exams", "dev"),
]
ALL_IDS = sorted(file_id for file_id, *_ in FILES)
TEXT_IDS = [file_id for file_id in ALL_IDS if file_id != NO_TEXT]
GOLD = {file_id: gold for file_id, _, gold, _ in FILES}


def spread(top, prob, fits_any=0.9):
    """An answer with `prob` on one group and the rest split between the other two."""
    rest = (1 - prob) / 2
    return {group_id: prob if group_id == top else rest for group_id in GROUP_IDS}, fits_any


# What the stand-in for Jev says about each file with text, under the default cutoffs
# (Unsorted under 0.5 fits_any or under 0.6 on top).
JEV = {
    "final.txt": spread("exams", 0.9),  # right, placed
    "hw1.txt": spread("homework", 0.8),  # right, placed
    "hw2_soln.txt": spread("exams", 0.7),  # wrong, placed
    "labs/lab1.py": ({"homework": 0.55, "lecture-notes": 0.3, "exams": 0.15}, 0.9),  # right, weak
    "lec1.md": spread("lecture-notes", 0.9, fits_any=0.3),  # right top, but no group fits
    "lec2.md": spread("lecture-notes", 0.85),  # right, placed
    "midterm.txt": spread("exams", 0.95),  # right, placed
    "hw0.txt": spread("homework", 0.9),  # dev: right, placed
    "lec0.md": spread("lecture-notes", 0.9),  # dev: right, placed
    "quiz0.txt": ({"exams": 0.55, "homework": 0.45, "lecture-notes": 0.0}, 0.9),  # dev: weak
}


class CappedBackend(FakeBackend):
    """A fake with a spend cap: it answers a set number of requests and refuses the rest."""

    def __init__(self, allowed, **answers):
        """Answer `allowed` requests as FakeBackend would, then raise BudgetExceeded."""
        super().__init__(**answers)
        self.allowed = allowed

    def sort(self, request):
        """Answer while under the cap. A refused request is not sent, so it is not recorded."""
        if len(self.calls) >= self.allowed:
            raise BudgetExceeded("spend cap reached: $5.0000 spent, the cap is $5")
        return super().sort(request)


class UnloggedBackend(FakeBackend):
    """A fake whose call log breaks: it answers a set number of requests, then cannot log."""

    def __init__(self, allowed, **answers):
        """Answer `allowed` requests as FakeBackend would, then raise LogFailed."""
        super().__init__(**answers)
        self.allowed = allowed

    def sort(self, request):
        """Answer while the log can be written. After that every request stops the run."""
        if len(self.calls) >= self.allowed:
            raise LogFailed("the call log cannot be written: No space left on device")
        return super().sort(request)


class MeteredBackend(FakeBackend):
    """A fake that reports tokens and a latency with every answer, as a billed backend does."""

    def __init__(self, input_tokens, output_tokens=None, **answers):
        """Report these token counts, and 250 ms, for every request."""
        super().__init__(**answers)
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": 250.0,
        }

    def sort(self, request):
        """Answer as FakeBackend would, with the usage added."""
        return super().sort(request).model_copy(update=self.usage)


class WindowBackend(FakeBackend):
    """A fake that reads the window: Exams when it says exam more often than lecture."""

    def sort(self, request):
        """Answer from the text of the one window in the request."""
        text = request.state.get("text", "")
        top = "exams" if text.count("exam") > text.count("lecture") else "lecture-notes"
        probs, fits_any = spread(top, 0.9)
        return super().sort(request).model_copy(update={"probs": probs, "fits_any": fits_any})


class StandIns:
    """The stand-in backends of one test, by the name a configuration starts with."""

    def __init__(self):
        """Jev answers from JEV, the rules are the real ones, the rest give one fixed answer."""
        self.made = {
            "jev": FakeBackend(answers=dict(JEV)),
            "rules": RulesBackend(),
            "embed": FakeBackend(default=spread("homework", 0.6, fits_any=1.0)),
            "embed-chunks": FakeBackend(default=spread("lecture-notes", 0.7, fits_any=1.0)),
            "llm": FakeBackend(answers={file_id: spread(GOLD[file_id], 1.0) for file_id in JEV}),
        }
        self.built = []

    def build(self, key, root, config):
        """Stand in for run.build_backend. An exception in the table is raised, not returned."""
        self.built.append(key)
        if isinstance(self.made[key], Exception):
            raise self.made[key]
        return self.made[key]

    @property
    def jev(self):
        """The stand-in for Jev."""
        return self.made["jev"]


@pytest.fixture
def backends(monkeypatch):
    """Replace run.build_backend with stand-ins, so no test builds a real backend."""
    stand_ins = StandIns()
    monkeypatch.setattr(run, "build_backend", stand_ins.build)
    return stand_ins


def add_file(root, file_id, content, gold, split="test"):
    """Put a file in the dump folder and add its gold label to gold.jsonl."""
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    label = GoldLabel(
        file_id=file_id,
        gold_group=gold,
        source_url=f"https://example.org/{file_id}",
        source_type="made up",
        license="CC0",
        sha256=hashlib.sha256(data).hexdigest(),
        split=split,
    )
    with (root / "gold.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.model_dump()) + "\n")


@pytest.fixture
def root(tmp_path):
    """A small evaluation vault: three groups (set version 3), the files above, gold labels."""
    root = tmp_path / "eval"
    vault.init_vault(root)
    vault.add_group(root, "Exams", "quizzes, midterm and final exams")
    vault.add_group(root, "Homework", "problem sets, labs and their solutions")
    vault.add_group(root, "Lecture notes", "lecture notes and slides")
    for file_id, content, gold, split in FILES:
        add_file(root, file_id, content, gold, split)
    return root


def evaluate(root, *args):
    """Run the command line on a vault and return its exit status."""
    return run.main(["--root", str(root), *args])


def metrics_of(root):
    """The metrics.json a run wrote."""
    return json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))


def report_of(root):
    """The report.md a run wrote."""
    return (root / "results" / "report.md").read_text(encoding="utf-8")


def snapshot(root):
    """Every file under a folder with its bytes, to check that nothing was written."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


# Names, the selection of files, and the builders


def test_a_configuration_name_splits_into_backend_and_variant():
    """The part before the colon is the backend; an experiment suffix is dropped."""
    assert run.split_name("jev:card") == ("jev", "card")
    assert run.split_name("embed-chunks:card") == ("embed-chunks", "card")
    assert run.split_name("jev:names_only@without-exams") == ("jev", "names_only")


def test_the_nine_configurations_are_the_five_variants_and_the_baselines():
    """Jev runs every variant; each baseline runs on the card."""
    assert run.CONFIG_NAMES[:5] == tuple(f"jev:{variant}" for variant in VARIANTS)
    assert run.CONFIG_NAMES[5:] == ("rules:card", "embed:card", "embed-chunks:card", "llm:card")


def test_configs_are_parsed_in_the_order_given():
    """Names are split on commas; spaces and repeats are dropped."""
    assert run.parse_configs("rules:card, jev:card,rules:card") == ["rules:card", "jev:card"]


@pytest.mark.parametrize("text", ["jev:cards", "jev:card,nope", "", " , "])
def test_a_name_that_is_no_configuration_is_rejected(text):
    """An unknown name, or no name at all, is a ValueError that lists the choices."""
    with pytest.raises(ValueError, match="pick from: jev:card"):
        run.parse_configs(text)


def test_select_takes_the_first_files_of_each_split():
    """With a limit, each split gives its first files by file id; the result is sorted."""
    labels = [
        GoldLabel(
            file_id=file_id,
            gold_group="homework",
            source_url="https://example.org",
            source_type="made up",
            license="CC0",
            sha256="0" * 64,
            split=split,
        )
        for file_id, split in [("d", "test"), ("b", "dev"), ("a", "test"), ("c", "test")]
    ]
    assert [label.file_id for label in run.select(labels, None)] == ["a", "b", "c", "d"]
    assert [label.file_id for label in run.select(labels, 2)] == ["a", "b", "c"]
    assert [label.file_id for label in run.select(labels, 1)] == ["a", "b"]


def test_the_fingerprint_follows_the_groups():
    """Any change to a group's id, name or description gives another fingerprint."""
    groups = [
        Group(id="hw", name="HW", covers="homework"),
        Group(id="ex", name="Exams", covers="e"),
    ]
    same = [group.model_copy() for group in groups]
    edited = [groups[0].model_copy(update={"covers": "homework and labs"}), groups[1]]
    assert run.groups_hash(groups) == run.groups_hash(same)
    assert run.groups_hash(groups) != run.groups_hash(edited)
    assert run.groups_hash(groups) != run.groups_hash(groups[:1])
    assert len(run.groups_hash(groups)) == 12


def test_the_cache_file_is_named_after_the_configuration(tmp_path):
    """One file per configuration under results/predictions, with the colon written as _."""
    folder = tmp_path / "results" / "predictions"
    assert run.cache_path(tmp_path, "jev:card") == folder / "jev_card.jsonl"
    assert run.cache_path(tmp_path, "embed-chunks:card") == folder / "embed-chunks_card.jsonl"
    assert (
        run.cache_path(tmp_path, "jev:card@without-exams")
        == folder / "jev_card@without-exams.jsonl"
    )


def test_the_baselines_are_built_as_the_contract_says(monkeypatch, tmp_path):
    """Rules as they are; both embedding rows keep the model in <root>/../models."""
    root = tmp_path / "eval"
    assert isinstance(run.build_backend("rules", root, Config()), RulesBackend)
    assert isinstance(run.build_backend("embed", root, Config()), EmbeddingBackend)
    made = []

    def embedding(**settings):
        """Stand in for EmbeddingBackend, and remember the settings it was built with."""
        made.append(settings)
        return "an embedding backend"

    monkeypatch.setattr(run, "EmbeddingBackend", embedding)
    assert run.build_backend("embed", root, Config()) == "an embedding backend"
    assert run.build_backend("embed-chunks", root, Config()) == "an embedding backend"
    assert made == [
        {"cache_dir": tmp_path / "models"},
        {"cache_dir": tmp_path / "models", "chunk_chars": 1600},
    ]
    with pytest.raises(ValueError, match="no backend is called 'nope'"):
        run.build_backend("nope", root, Config())


def test_jev_is_built_by_make_backend_for_the_vault(monkeypatch, tmp_path):
    """make_backend logs to the vault's own Jev log with the vault's key, so the cap covers it."""
    seen = []

    def make(root, config):
        """Stand in for backends.make_backend, and remember what it was given."""
        seen.append((root, config))
        return "the vault's Jev backend"

    monkeypatch.setattr(run, "make_backend", make)
    config = Config()
    assert run.build_backend("jev", tmp_path / "eval", config) == "the vault's Jev backend"
    assert seen == [(tmp_path / "eval", config)]


def test_a_backend_without_its_key_cannot_be_built(monkeypatch, tmp_path):
    """Jev and the LLM raise BackendError without a key. That is what makes them pending."""
    monkeypatch.setattr(backends_module, "env_value", lambda name, start=None: None)
    monkeypatch.setattr(llm, "env_value", lambda name, start=None: None)
    with pytest.raises(BackendError, match="TYPESAFE_API_KEY"):
        run.build_backend("jev", tmp_path, Config())
    with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
        run.build_backend("llm", tmp_path, Config())


# The cache


def test_a_run_caches_one_line_per_file_for_every_configuration(root, backends):
    """Each configuration gets one JSONL file with one line per gold file, sorted by id."""
    assert evaluate(root, "--no-experiments") == 0
    folder = root / "results" / "predictions"
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        run.cache_path(root, name).name for name in run.CONFIG_NAMES
    )
    for name in run.CONFIG_NAMES:
        lines = run.cache_path(root, name).read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["file_id"] for line in lines] == ALL_IDS


def test_a_cached_line_holds_the_answer_and_what_it_was_computed_under(root, backends):
    """A line has the file, its gold label and split, the answer, and the set it ran under."""
    backends.made["jev"] = MeteredBackend(1200, answers=dict(JEV))
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    lines = run.cache_path(root, "jev:card").read_text(encoding="utf-8").splitlines()
    line = lines[ALL_IDS.index("hw1.txt")]
    probs, fits_any = JEV["hw1.txt"]
    assert json.loads(line) == {
        "file_id": "hw1.txt",
        "gold": "homework",
        "split": "test",
        "no_text": False,
        "probs": probs,
        "fits_any": fits_any,
        "input_tokens": 1200,
        "output_tokens": None,
        "latency_ms": 250.0,
        "model": "fake-0",
        "set_version": 3,
        "groups_hash": run.groups_hash(vault.read_groups(root)),
        "error": None,
        "window_tops": None,
    }


def test_a_second_run_asks_nothing(root, backends, capsys):
    """Every answer is in the cache, so a rerun builds no backend and sends no request."""
    evaluate(root, "--no-experiments")
    assert len(backends.jev.calls) == 5 * len(TEXT_IDS)
    before = snapshot(root / "results" / "predictions")
    built = list(backends.built)
    capsys.readouterr()
    assert evaluate(root, "--no-experiments") == 0
    assert len(backends.jev.calls) == 5 * len(TEXT_IDS)
    assert backends.built == built
    assert snapshot(root / "results" / "predictions") == before
    assert "jev:card: asked 0, failed 0, no text 0, already cached 11" in capsys.readouterr().out


def test_a_failed_request_is_cached_with_its_error_and_retried(root, backends):
    """A line with an error is asked again on the next run; the other lines are left alone."""
    backends.jev.fail = {"hw1.txt"}
    assert evaluate(root, "--configs", "jev:card", "--no-experiments") == 0
    row = run.read_rows(root, "jev:card")["hw1.txt"]
    assert "told to fail for hw1.txt" in row.error
    assert (row.probs, row.fits_any, row.model) == ({}, None, None)
    assert metrics_of(root)["errors"] == {"jev:card": {"hw1.txt": row.error}}

    backends.jev.fail = set()
    backends.jev.calls.clear()
    assert evaluate(root, "--configs", "jev:card", "--no-experiments") == 0
    assert [call.file_id for call in backends.jev.calls] == ["hw1.txt"]
    rows = run.read_rows(root, "jev:card")
    assert rows["hw1.txt"].error is None
    assert rows["hw1.txt"].probs == JEV["hw1.txt"][0]
    lines = run.cache_path(root, "jev:card").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ALL_IDS)
    assert metrics_of(root)["errors"] == {}


def test_answers_under_an_older_set_version_are_asked_again(root, backends):
    """Editing a group gives a new set version, and no older answer counts under it."""
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    vault.edit_group(root, "Homework", covers="problem sets and labs")
    backends.jev.calls.clear()
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    assert sorted(call.file_id for call in backends.jev.calls) == TEXT_IDS
    assert {call.set_version for call in backends.jev.calls} == {4}
    assert {row.set_version for row in run.read_rows(root, "jev:card").values()} == {4}


def test_answers_of_another_set_version_are_not_reported(root, backends):
    """The report reads only rows of the current set version, even when nothing is asked."""
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    vault.edit_group(root, "Homework", covers="problem sets and labs")
    evaluate(root, "--configs", "rules:card", "--no-experiments")
    metrics = metrics_of(root)
    assert metrics["set_version"] == 4
    assert metrics["headline"]["jev:card"] is None
    assert metrics["configurations"]["jev:card"]["status"] == "not run"


def test_gold_labels_are_read_fresh_and_not_from_the_cache(root, backends):
    """A changed gold label changes the score without a new request."""
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    assert metrics_of(root)["headline"]["jev:card"]["top1_accuracy"] == pytest.approx(6 / 7)
    gold = root / "gold.jsonl"
    old = '"file_id": "hw2_soln.txt", "gold_group": "homework"'
    assert old in gold.read_text(encoding="utf-8")
    gold.write_text(
        gold.read_text(encoding="utf-8").replace(old, old.replace("homework", "exams")),
        encoding="utf-8",
    )
    backends.jev.calls.clear()
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    assert backends.jev.calls == []
    assert metrics_of(root)["headline"]["jev:card"]["top1_accuracy"] == pytest.approx(7 / 7)


def test_no_temporary_file_is_left_behind(root, backends):
    """The cache is written through a temporary file that then takes its place."""
    evaluate(root, "--no-experiments")
    assert not list((root / "results").rglob("*.tmp"))


# Files with no text


def test_a_file_with_no_text_is_never_sent_to_a_content_backend(root, backends):
    """It is recorded with no_text true and no answer, and no model is asked about it."""
    evaluate(root, "--no-experiments")
    for key in ("jev", "embed", "embed-chunks", "llm"):
        asked = {call.file_id for call in backends.made[key].calls}
        assert asked == set(TEXT_IDS)
        name = "jev:card" if key == "jev" else f"{key}:card"
        row = run.read_rows(root, name)[NO_TEXT]
        assert row.no_text is True
        assert (row.probs, row.fits_any, row.model, row.error) == ({}, None, None, None)
    assert run.read_rows(root, "jev:card")["hw1.txt"].no_text is False


def test_the_rules_get_a_state_with_only_the_name_for_a_file_with_no_text(root, backends):
    """The rules read names, so they are asked about every file, with or without text."""
    rules = FakeBackend()
    backends.made["rules"] = rules
    evaluate(root, "--configs", "rules:card")
    states = {call.file_id: call.state for call in rules.calls}
    assert sorted(states) == ALL_IDS
    assert states[NO_TEXT] == {
        "file": {"name": "hw3_data.zip", "path": NO_TEXT, "type": "Binary file"}
    }
    assert set(states["hw1.txt"]) == {"file", "title", "opening"}
    row = run.read_rows(root, "rules:card")[NO_TEXT]
    assert row.no_text is True
    assert row.fits_any == 0.9
    assert row.model == "fake-0"


def test_the_real_rules_sort_a_file_with_no_text_by_its_name(root, backends):
    """With the real rules, the zip goes to Homework for the "hw" in its name."""
    evaluate(root, "--configs", "rules:card")
    row = run.read_rows(root, "rules:card")[NO_TEXT]
    assert row.probs == {"exams": 0.0, "homework": 1.0, "lecture-notes": 0.0}
    assert row.model == "rules"


def test_a_configuration_with_nothing_to_ask_needs_no_backend(root, backends):
    """With only the file that has no text in the run, Jev is recorded without being built."""
    assert evaluate(root, "--configs", "jev:card", "--limit", "1", "--no-experiments") == 0
    assert backends.built == ["jev"]  # for hw0.txt, the first dev file
    backends.built.clear()
    gold = root / "gold.jsonl"
    lines = gold.read_text(encoding="utf-8").splitlines()
    gold.write_text(lines[0] + "\n", encoding="utf-8")  # only the zip is left
    assert evaluate(root, "--configs", "jev:full_text", "--no-experiments") == 0
    assert backends.built == []
    assert run.read_rows(root, "jev:full_text")[NO_TEXT].no_text is True


# What is asked, and through which backend


def test_one_jev_backend_serves_every_jev_configuration_and_every_pass(root, backends):
    """Jev is built once for the five variants and the experiments, so they share a cap."""
    assert evaluate(root) == 0
    assert Counter(backends.built) == {
        "jev": 1,
        "rules": 1,
        "embed": 1,
        "embed-chunks": 1,
        "llm": 1,
    }
    passes = [call for call in backends.jev.calls if call.set_version == 0]
    assert len(passes) == 2 * len(TEXT_IDS)  # the card and the names-only pass without Exams


def test_each_jev_configuration_asks_with_its_own_variant(root, backends):
    """Every variant asks once per file with text, under the vault's set version and groups."""
    evaluate(root, "--no-experiments")
    calls = backends.jev.calls
    assert Counter(call.variant for call in calls) == dict.fromkeys(VARIANTS, len(TEXT_IDS))
    assert {call.set_version for call in calls} == {3}
    assert {tuple(group.id for group in call.groups) for call in calls} == {GROUP_IDS}
    for call in calls:
        assert call.with_descriptions is (call.variant != "names_only")
        assert ("name" in call.state["file"]) is (call.variant != "card_no_name")
        assert ("text" in call.state) is (call.variant in ("full_text", "window_vote"))


def test_cards_are_built_once_per_file_and_variant_in_the_main_thread(root, backends, monkeypatch):
    """PyMuPDF is not safe across threads, so no worker thread ever builds a card."""
    built = []
    real_build_card = run.build_card

    def recording(path, file_id, config, variant=None):
        """Build the card as usual, and note what was built and in which thread."""
        built.append((file_id, variant, threading.current_thread() is threading.main_thread()))
        return real_build_card(path, file_id, config, variant)

    monkeypatch.setattr(run, "build_card", recording)
    assert evaluate(root) == 0
    assert all(in_main_thread for _, _, in_main_thread in built)
    # Five variants of eleven files. The experiments' passes reuse the cards.
    assert len(built) == len(set(built)) == len(VARIANTS) * len(ALL_IDS)


def test_jev_gets_the_worker_threads_and_the_other_backends_run_one_at_a_time(
    root, backends, monkeypatch
):
    """--workers sets the size of the pool for Jev; every other backend gets one thread."""
    sizes = []
    real_pool = run.ThreadPoolExecutor

    def pool(max_workers):
        """Make the real pool, and note its size."""
        sizes.append(max_workers)
        return real_pool(max_workers=max_workers)

    monkeypatch.setattr(run, "ThreadPoolExecutor", pool)
    configs = "jev:card,rules:card,embed:card,llm:card,jev:names_only"
    assert evaluate(root, "--configs", configs, "--workers", "3", "--no-experiments") == 0
    assert sizes == [3, 1, 1, 1, 3]


def test_a_bug_in_a_backend_is_not_swallowed(root, backends):
    """Only BackendError is a failed request. Anything else ends the run with its traceback."""

    class BrokenBackend(FakeBackend):
        """A fake with a bug that shows on one file."""

        def sort(self, request):
            """Raise something that is no BackendError for hw1.txt."""
            if request.file_id == "hw1.txt":
                raise RuntimeError("a bug, not a failed request")
            return super().sort(request)

    backends.made["jev"] = BrokenBackend(answers=dict(JEV))
    with pytest.raises(RuntimeError, match="a bug, not a failed request"):
        evaluate(root, "--configs", "jev:card", "--no-experiments")
    # The cache is still readable, and holds no line for the file that hit the bug.
    rows = run.read_rows(root, "jev:card")
    assert rows[NO_TEXT].no_text is True
    assert "hw1.txt" not in rows
    assert not (root / "results" / "report.md").exists()


def test_a_window_vote_line_holds_the_top_group_of_each_window(root, backends):
    """The answer is the average over the windows; the line also keeps each window's pick."""
    add_file(root, "lec3_long.md", "lecture " * 1500 + "exam " * 1600, "lecture-notes")
    backends.made["jev"] = WindowBackend()
    evaluate(root, "--configs", "jev:window_vote,jev:card", "--no-experiments")
    rows = run.read_rows(root, "jev:window_vote")
    long = rows["lec3_long.md"]
    assert long.window_tops == ["lecture-notes", "lecture-notes", "exams"]
    assert long.probs == pytest.approx(
        {
            "exams": (0.05 + 0.05 + 0.9) / 3,
            "homework": 0.05,
            "lecture-notes": (0.9 + 0.9 + 0.05) / 3,
        }
    )
    assert rows["hw1.txt"].window_tops == ["lecture-notes"]
    assert rows[NO_TEXT].window_tops is None
    assert run.read_rows(root, "jev:card")["lec3_long.md"].window_tops is None
    # Eight test files with text now; one has several windows, and they disagree.
    assert metrics_of(root)["window_vote"] == {
        "n": 8,
        "several_windows": 1,
        "disagree": 1,
        "disagree_share": 1 / 8,
    }
    assert (
        "Test files jev:window_vote answered: 8. With more than one window: 1. "
        "With windows that disagree on the top group: 1, which is 12.5% of the 8."
    ) in report_of(root)


def test_a_backend_that_cannot_be_built_makes_its_configuration_pending(root, backends, capsys):
    """Pending is a status with a reason, not an error: the run goes on and exits with 0."""
    backends.made["llm"] = BackendError("the LLM baseline needs an Anthropic API key")
    assert evaluate(root, "--no-experiments") == 0
    assert "llm:card: pending (the LLM baseline needs an Anthropic API key)" in (
        capsys.readouterr().out
    )
    assert not run.cache_path(root, "llm:card").exists()
    metrics = metrics_of(root)
    assert metrics["configurations"]["llm:card"] == {
        "status": "pending",
        "pending": "the LLM baseline needs an Anthropic API key",
        "expected": 10,
        "answered": 0,
        "errors": 0,
        "not_asked": 10,
    }
    assert metrics["headline"]["llm:card"] is None
    assert metrics["errors"] == {}
    assert "llm:card is pending: the LLM baseline needs an Anthropic API key" in metrics["caveats"]
    assert "No row for: llm:card (pending). See the caveats." in report_of(root)
    assert metrics["configurations"]["jev:card"]["status"] == "complete"


def test_jev_without_a_key_leaves_every_jev_configuration_pending(root, backends):
    """The five variants and the passes all wait; the baselines still run."""
    backends.made["jev"] = BackendError("no TypeSafe API key found")
    assert evaluate(root) == 0
    metrics = metrics_of(root)
    for variant in VARIANTS:
        assert metrics["configurations"][f"jev:{variant}"]["status"] == "pending"
    assert metrics["passes"]["jev:card@without-exams"]["status"] == "pending"
    assert metrics["configurations"]["rules:card"]["status"] == "complete"
    assert [row["met"] for row in metrics["go_no_go"]] == [None, None, None]


# The spend cap


def test_the_spend_cap_stops_the_run_and_keeps_what_is_cached(root, backends, capsys):
    """Asking stops, the answers so far stay, the report is still written, and the exit is 2."""
    backends.made["jev"] = CappedBackend(4, answers=dict(JEV))
    code = evaluate(root, "--configs", "jev:card,jev:full_text,rules:card", "--workers", "1")
    assert code == 2
    rows = run.read_rows(root, "jev:card")
    assert sorted(file_id for file_id, row in rows.items() if row.fits_any is not None) == [
        "final.txt",
        "hw0.txt",
        "hw1.txt",
        "hw2_soln.txt",
    ]
    assert sorted(rows) == sorted([NO_TEXT, "final.txt", "hw0.txt", "hw1.txt", "hw2_soln.txt"])
    # Nothing after the stop was started: not the next configuration, not the experiments.
    assert not run.cache_path(root, "jev:full_text").exists()
    assert not run.cache_path(root, "rules:card").exists()
    assert not run.cache_path(root, "jev:card@without-exams").exists()

    streams = capsys.readouterr()
    assert "stopped: spend cap reached: $5.0000 spent, the cap is $5" in streams.err
    assert (
        "jev:card: asked 4, failed 0, no text 1, already cached 0, "
        "stopped (spend cap reached: $5.0000 spent, the cap is $5)"
    ) in streams.out
    metrics = metrics_of(root)
    assert metrics["stopped"] == "spend cap reached: $5.0000 spent, the cap is $5"
    assert metrics["configurations"]["jev:card"] == {
        "status": "partial",
        "pending": None,
        "expected": 10,
        "answered": 4,
        "errors": 0,
        "not_asked": 6,
    }
    assert metrics["headline"]["jev:card"]["n"] == 3  # final, hw1 and hw2_soln are test files
    assert (
        "The run was stopped before it finished (spend cap reached: $5.0000 spent, the cap is "
        "$5). Whatever had not been asked by then is missing."
    ) in metrics["caveats"]
    assert any("jev:card answered 4 of 10 files" in caveat for caveat in metrics["caveats"])
    assert "## Headline" in report_of(root)

    # With room under the cap again, a rerun asks only about what is missing.
    later = FakeBackend(answers=dict(JEV))
    backends.made["jev"] = later
    assert evaluate(root, "--configs", "jev:card", "--no-experiments") == 0
    assert sorted(call.file_id for call in later.calls) == [
        "labs/lab1.py",
        "lec0.md",
        "lec1.md",
        "lec2.md",
        "midterm.txt",
        "quiz0.txt",
    ]
    assert metrics_of(root)["stopped"] is None


def test_a_cap_reached_before_the_first_answer_still_gives_a_report(root, backends):
    """With nothing cached at all, the report is written and says what is missing."""
    backends.made["jev"] = CappedBackend(0)
    assert evaluate(root, "--configs", "jev:card") == 2
    metrics = metrics_of(root)
    assert metrics["headline"]["jev:card"] is None
    assert metrics["configurations"]["jev:card"]["status"] == "not run"
    assert "No row for: jev:card (not run)" in report_of(root)


def test_a_call_log_that_cannot_be_written_stops_the_run_like_the_spend_cap(root, backends, capsys):
    """LogFailed is a RunStopped too: no further request, the report from the cache, exit 2."""
    backends.made["jev"] = UnloggedBackend(2, answers=dict(JEV))
    code = evaluate(root, "--configs", "jev:card,jev:full_text,rules:card", "--workers", "1")
    assert code == 2
    rows = run.read_rows(root, "jev:card")
    # The two answers that were logged stay. The request that broke is not cached as a failed
    # file: the failure is about the run, so a later run asks it again.
    assert sorted(file_id for file_id, row in rows.items() if row.fits_any is not None) == [
        "final.txt",
        "hw0.txt",
    ]
    assert all(row.error is None for row in rows.values())
    assert len(backends.jev.calls) == 2
    assert not run.cache_path(root, "jev:full_text").exists()
    assert not run.cache_path(root, "rules:card").exists()

    why = "the call log cannot be written: No space left on device"
    streams = capsys.readouterr()
    assert f"stopped: {why}" in streams.err
    assert "Fix the cause (for the spend cap, raise max_spend_usd) and run again" in streams.err
    assert f"jev:card: asked 2, failed 0, no text 1, already cached 0, stopped ({why})" in (
        streams.out
    )
    metrics = metrics_of(root)
    assert metrics["stopped"] == why
    assert metrics["errors"] == {}
    assert (
        f"The run was stopped before it finished ({why}). "
        "Whatever had not been asked by then is missing."
    ) in metrics["caveats"]
    assert "## Headline" in report_of(root)


def test_a_backend_that_stops_the_run_when_it_is_built_is_not_pending(root, backends, capsys):
    """A RunStopped from building the backend ends the run with status 2, as one from a call."""
    backends.made["jev"] = LogFailed("the call log cannot be read: Permission denied")
    assert evaluate(root, "--configs", "jev:card,rules:card") == 2
    streams = capsys.readouterr()
    assert "jev:card: stopped (the call log cannot be read: Permission denied)" in streams.out
    assert "stopped: the call log cannot be read: Permission denied" in streams.err
    assert not run.cache_path(root, "rules:card").exists()
    metrics = metrics_of(root)
    assert metrics["stopped"] == "the call log cannot be read: Permission denied"
    assert metrics["configurations"]["jev:card"]["pending"] is None


# The real Jev backend, over a fake wire

GROUP_NAMES = {"exams": "Exams", "homework": "Homework", "lecture-notes": "Lecture notes"}
CALL_TOKENS = 100_000  # what the fake endpoint bills per request: $0.0042 at the default price


def typesafe(request):
    """A fake TypeSafe endpoint: it answers one request from JEV, by the file path in the card."""
    body = json.loads(request.content)
    probs, fits_any = JEV[body["state"]["file"]["path"]]
    offered = body["questions"]["group"]["criteria"]
    by_name = {GROUP_NAMES[group_id]: prob for group_id, prob in probs.items()}
    by_name = {name: prob for name, prob in by_name.items() if name in offered}
    top = max(by_name, key=by_name.get)
    answers = {
        "group": {
            "type": "choice",
            "choice": top,
            "confidence": by_name[top],
            "probabilities": by_name,
        },
        "fits_any": {"type": "noul", "noul": fits_any},
    }
    return httpx2.Response(
        200,
        json={
            "model": "jev-1.14.0",
            "answers": answers,
            "usage": {"input_tokens": CALL_TOKENS, "output_tokens": 30},
        },
        headers={"x-typesafe-request-id": "req_test"},
    )


@pytest.fixture
def real_jev(monkeypatch, root):
    """Build real JevBackends that log to the vault and send to the fake endpoint above."""
    real_client = jev_module.TypeSafeClient

    def client(**kwargs):
        """The real SDK client, with bodies left uncompressed so the endpoint can read them."""
        return real_client(compression=False, **kwargs)

    def build():
        """A JevBackend on the vault's config and log, as run.build_backend would make it."""
        return JevBackend(
            Config.load(root),
            decide_path(root, JEV_LOG_FILE),
            api_key="test-key-not-a-real-one",
            transport=httpx2.MockTransport(typesafe),
        )

    monkeypatch.setattr(jev_module, "TypeSafeClient", client)
    return build


def test_a_run_through_the_real_jev_backend_logs_every_call_in_the_vault(
    root, backends, real_jev, capsys
):
    """Four threads share the backend: every call lands in the vault's log and in the cache."""
    backends.made["jev"] = real_jev()
    assert evaluate(root, "--configs", "jev:card", "--workers", "4") == 0
    log = read_jsonl(decide_path(root, JEV_LOG_FILE))
    # jev:card under the vault's set, and its pass without Exams under set version 0.
    assert Counter(line["set_version"] for line in log) == {3: len(TEXT_IDS), 0: len(TEXT_IDS)}
    assert {(line["status"], line["variant"], line["model_answered"]) for line in log} == {
        ("ok", "card", "jev-1.14.0")
    }
    rows = run.read_rows(root, "jev:card")
    assert all(rows[file_id].error is None for file_id in ALL_IDS)
    row = rows["hw1.txt"]
    assert row.probs == pytest.approx(JEV["hw1.txt"][0])
    assert (row.fits_any, row.input_tokens, row.model) == (0.9, CALL_TOKENS, "jev-1.14.0")
    assert row.latency_ms > 0
    headline = metrics_of(root)["headline"]["jev:card"]
    assert headline["top1_accuracy"] == pytest.approx(6 / 7)
    assert headline["usd_per_file"] == pytest.approx(CALL_TOKENS / 1e6 * 0.042)
    # Twenty calls of 100,000 tokens are in the log, and the dry run reads its spend from it.
    capsys.readouterr()
    evaluate(root, "--dry-run")
    assert "Spent so far: $0.0840 of the $5.00 cap." in capsys.readouterr().out


def test_the_vaults_own_spend_cap_stops_the_run(root, backends, real_jev, capsys):
    """Three calls cost $0.0126, which reaches a cap of $0.0125: the fourth is never sent."""
    Config.load(root).model_copy(update={"max_spend_usd": 0.0125}).save(root)
    backends.made["jev"] = real_jev()
    assert evaluate(root, "--configs", "jev:card", "--workers", "1") == 2
    log_path = decide_path(root, JEV_LOG_FILE)
    assert [line["file_id"] for line in read_jsonl(log_path)] == ["final.txt", "hw0.txt", "hw1.txt"]
    rows = run.read_rows(root, "jev:card")
    assert sorted(file_id for file_id, row in rows.items() if row.fits_any is not None) == [
        "final.txt",
        "hw0.txt",
        "hw1.txt",
    ]
    assert (
        "stopped: spend cap reached: $0.0126 spent, the cap is $0.0125" in capsys.readouterr().err
    )
    assert metrics_of(root)["configurations"]["jev:card"]["answered"] == 3

    # A new backend reads the spend from the log, so a second run sends nothing at all.
    backends.made["jev"] = real_jev()
    assert evaluate(root, "--configs", "jev:card", "--workers", "4") == 2
    assert len(read_jsonl(log_path)) == 3


# The dry run


def test_a_dry_run_asks_and_writes_nothing(root, backends, capsys):
    """It builds no backend, and leaves every file of the vault as it was."""
    # A hand-made group note without an id: a real run would write the id and a new set.
    (root / "_groups" / "Projects.md").write_text("---\ncovers: course projects\n---\n")
    before = snapshot(root)
    assert evaluate(root, "--dry-run") == 0
    assert snapshot(root) == before
    assert backends.built == []
    out = capsys.readouterr().out
    assert "Set version 3, model jev-1.14.0. 11 files (dev 3, test 8), 1 with no text." in out
    assert "Spent so far: $0.0000 of the $5.00 cap." in out
    assert out.rstrip().endswith("Nothing was built, sent or written.")


def test_a_dry_run_estimates_what_each_configuration_would_ask(root, backends, capsys):
    """Per configuration: files, requests, and tokens as card size plus 700 per request."""
    append_jsonl(decide_path(root, JEV_LOG_FILE), {"file_id": "x", "input_tokens": 1_000_000})
    assert evaluate(root, "--dry-run") == 0
    lines = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines() if line}
    config = Config.load(root)
    paths = dict(vault.dump_files(root, config))
    tokens = {
        variant: sum(
            build_card(paths[file_id], file_id, config, variant).file.card_tokens
            for file_id in TEXT_IDS
        )
        + 700 * len(TEXT_IDS)
        for variant in ("card", "full_text")
    }
    for name, variant in [("jev:card", "card"), ("jev:full_text", "full_text")]:
        usd = tokens[variant] / 1e6 * 0.042
        assert lines[name] == [name, "10", "10", f"{tokens[variant]:,}", f"{usd:.4f}"]
    assert lines["rules:card"] == ["rules:card", "11", "11", "n/a", "free"]
    assert lines["embed:card"] == ["embed:card", "10", "10", "n/a", "free"]
    llm_usd = tokens["card"] / 1e6 * llm.USD_PER_MILLION_INPUT_TOKENS
    assert lines["llm:card"][:5] == [
        "llm:card",
        "10",
        "10",
        f"{tokens['card']:,}",
        f"{llm_usd:.4f}",
    ]
    assert lines["jev:card@without-exams"][:3] == ["jev:card@without-exams", "10", "10"]
    assert lines["Jev:"][-6:] == ["far:", "$0.0420", "of", "the", "$5.00", "cap."]


def test_a_dry_run_after_a_run_has_nothing_left_to_ask(root, backends, capsys):
    """What is cached is not asked again, so the dry run counts none of it."""
    evaluate(root)
    capsys.readouterr()
    assert evaluate(root, "--dry-run") == 0
    lines = [line.split() for line in capsys.readouterr().out.splitlines() if ":" in line[:32]]
    asked = {line[0]: line[1] for line in lines if line[0] in run.CONFIG_NAMES}
    assert asked == dict.fromkeys(run.CONFIG_NAMES, "0")


def test_a_dry_run_says_when_the_cap_would_be_reached(root, backends, capsys):
    """Spend so far plus the estimate at or over the cap gets its own line."""
    config = Config.load(root)
    config.model_copy(update={"max_spend_usd": 0.0001}).save(root)
    assert evaluate(root, "--dry-run") == 0
    assert "That reaches the cap: the run would stop part way." in capsys.readouterr().out


# The limit and the command line


def test_limit_takes_the_first_files_of_each_split(root, backends):
    """Only those files are asked about, cached and reported."""
    assert evaluate(root, "--limit", "2", "--configs", "jev:card", "--no-experiments") == 0
    # Dev: hw0.txt and lec0.md. Test: the zip (no text) and final.txt.
    assert sorted(call.file_id for call in backends.jev.calls) == [
        "final.txt",
        "hw0.txt",
        "lec0.md",
    ]
    assert sorted(run.read_rows(root, "jev:card")) == [NO_TEXT, "final.txt", "hw0.txt", "lec0.md"]
    metrics = metrics_of(root)
    assert metrics["limit"] == 2
    assert (metrics["files"]["dev"], metrics["files"]["test"]) == (2, 2)
    assert metrics["headline"]["jev:card"]["n"] == 1
    assert metrics["caveats"][0] == (
        "This run took only the first files of each split (--limit 2): "
        "every number rests on 4 files in all."
    )


def test_a_full_run_after_a_limited_one_asks_only_the_rest(root, backends):
    """The limited run's answers stay in the cache and are not paid for again."""
    evaluate(root, "--limit", "2", "--configs", "jev:card", "--no-experiments")
    backends.jev.calls.clear()
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    asked = sorted(call.file_id for call in backends.jev.calls)
    assert asked == [x for x in TEXT_IDS if x not in ("final.txt", "hw0.txt", "lec0.md")]
    assert sorted(run.read_rows(root, "jev:card")) == ALL_IDS


def test_only_the_named_configurations_run(root, backends):
    """--configs picks what is asked; the report still lists the others as not run."""
    assert evaluate(root, "--configs", "rules:card,embed:card") == 0
    assert backends.built == ["rules", "embed"]
    metrics = metrics_of(root)
    assert metrics["configurations"]["embed:card"]["status"] == "complete"
    assert metrics["configurations"]["jev:card"]["status"] == "not run"
    assert "jev:card was not run: it has no cached answers for these files." in metrics["caveats"]


def test_a_folder_that_is_no_vault_is_an_error(tmp_path, backends, capsys):
    """A known problem prints one error line and exits with 1."""
    assert evaluate(tmp_path) == 1
    assert capsys.readouterr().err == (
        f"error: {tmp_path.resolve()} is not a vault: it has no .decide/config.yaml\n"
    )


def test_a_vault_without_gold_labels_is_an_error(root, backends, capsys):
    """Without gold.jsonl there is nothing to score against."""
    (root / "gold.jsonl").unlink()
    assert evaluate(root) == 1
    assert "has no gold.jsonl; build the dataset first" in capsys.readouterr().err


def test_a_gold_label_for_a_file_that_is_gone_is_an_error(root, backends, capsys):
    """Every gold label must name a file in the dump folder."""
    (root / "_dump" / "hw1.txt").unlink()
    assert evaluate(root) == 1
    assert "names files that are not in the dump folder (1 in all, the first is hw1.txt)" in (
        capsys.readouterr().err
    )


def test_an_unknown_configuration_is_an_error(root, backends, capsys):
    """The message lists the names to pick from, and nothing is built."""
    assert evaluate(root, "--configs", "jev:cards") == 1
    assert "error: unknown configuration jev:cards; pick from: jev:card," in capsys.readouterr().err
    assert backends.built == []


@pytest.mark.parametrize("args", [["--workers", "0"], ["--limit", "0"]])
def test_workers_and_limit_must_be_at_least_one(root, backends, capsys, args):
    """Zero or less is a mistake, not a request for nothing."""
    assert evaluate(root, *args) == 1
    assert "must be at least 1" in capsys.readouterr().err


def test_a_usage_error_exits_with_1_because_2_means_the_spend_cap(root, capsys):
    """argparse would exit with 2 for a bad flag; here that status is taken."""
    with pytest.raises(SystemExit) as stop:
        evaluate(root, "--nope")
    assert stop.value.code == 1
    assert capsys.readouterr().err.startswith("error: unrecognized arguments: --nope")


# The report, end to end


def test_metrics_json_has_every_section(root, backends):
    """The file is one JSON object with a fixed set of sections."""
    assert evaluate(root) == 0
    metrics = metrics_of(root)
    assert list(metrics) == [
        "model",
        "set_version",
        "limit",
        "stopped",
        "cutoffs",
        "groups",
        "files",
        "configurations",
        "passes",
        "headline",
        "majority",
        "all_files",
        "by_construction",
        "sensitivity",
        "tuned",
        "per_group",
        "variant_differences",
        "ceiling",
        "window_vote",
        "go_no_go",
        "experiments",
        "label_signal",
        "errors",
        "caveats",
    ]
    assert (metrics["model"], metrics["set_version"], metrics["limit"]) == ("jev-1.14.0", 3, None)
    assert metrics["cutoffs"] == {"fits_any_below": 0.5, "top_below": 0.6, "close_margin": 0.15}
    assert metrics["groups"] == {
        "exams": "Exams",
        "homework": "Homework",
        "lecture-notes": "Lecture notes",
    }
    assert metrics["files"] == {
        "dev": 3,
        "test": 8,
        "dev_with_text": 3,
        "test_with_text": 7,
        "dev_no_text": 0,
        "test_no_text": 1,
        "test_gold": {"exams": 2, "homework": 3, "lecture-notes": 2},
        "no_gold": 0,
    }
    assert list(metrics["headline"]) == list(run.CONFIG_NAMES)
    assert list(metrics["configurations"]) == list(run.CONFIG_NAMES)
    assert list(metrics["all_files"]) == list(run.CONFIG_NAMES)
    for name in run.CONFIG_NAMES:
        assert set(metrics["headline"][name]) == set(Metrics.model_fields)
        assert set(metrics["all_files"][name]) == {"n", "top1_accuracy"}
        assert metrics["configurations"][name]["status"] == "complete"
    assert list(metrics["tuned"]) == [f"jev:{variant}" for variant in VARIANTS]
    assert set(metrics["tuned"]["jev:card"]) == {
        "dev_n",
        "cutoffs",
        "metrics",
        "scored_pairs",
        "tied_pairs",
        "tunable",
        "close_call_can_fire",
    }
    assert set(metrics["majority"]) == {"group", "headline", "all_files"}
    assert set(metrics["by_construction"]) == {
        "close_call_can_fire",
        "probabilities_not_measured",
        "no_pick",
    }
    assert set(metrics["sensitivity"]) == {"n", "rows", "rules_right"}
    assert set(metrics["ceiling"]) == {"right", "n", "margin_points", "at_ceiling"}
    assert set(metrics["label_signal"]) == {"folders", "test_with_text", "rules"}
    assert list(metrics["per_group"]) == list(GROUP_IDS)
    assert set(metrics["per_group"]["exams"]) == {"n", "precision", "recall", "f1"}
    assert list(metrics["variant_differences"]) == [f"jev:{variant}" for variant in VARIANTS[1:]]
    assert [set(row) for row in metrics["go_no_go"]] == [
        {"result", "then", "measured", "comparisons", "verdict", "met"}
    ] * 3
    assert set(metrics["go_no_go"][0]["comparisons"][0]) == {
        "first",
        "second",
        "margin_points",
        "n",
        "first_right",
        "second_right",
        "only_first",
        "only_second",
        "p_value",
        "outcome",
    }
    assert set(metrics["experiments"]) == {
        "held_out_group",
        "has_held_out_group",
        "held_out_named_in",
        "group_added",
        "edits_base",
        "description_edits",
    }
    assert metrics["errors"] == {}
    assert all(isinstance(caveat, str) for caveat in metrics["caveats"])


def test_the_headline_numbers_of_the_constructed_case(root, backends):
    """Test split, files with text, default cutoffs: checked against the answers in JEV."""
    evaluate(root, "--no-experiments")
    metrics = metrics_of(root)
    card = metrics["headline"]["jev:card"]
    assert card["n"] == 7
    assert card["top1_accuracy"] == pytest.approx(6 / 7)  # all but hw2_soln.txt
    assert card["placed_share"] == pytest.approx(5 / 7)  # lab1 and lec1 go to Unsorted
    assert card["placed_accuracy"] == pytest.approx(4 / 5)
    assert card["unsorted_share"] == pytest.approx(2 / 7)
    assert card["unsorted_top1_right_share"] == 1.0
    assert card["close_call_share"] == 0.0
    assert card["per_group"]["homework"] == {
        "n": 3,
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(0.8),
    }
    # The stand-in answers by file id, so every Jev variant gives the same numbers.
    assert metrics["headline"]["jev:full_text"] == card
    assert metrics["headline"]["rules:card"]["top1_accuracy"] == 1.0
    assert metrics["headline"]["embed:card"]["top1_accuracy"] == pytest.approx(3 / 7)
    assert metrics["headline"]["embed-chunks:card"]["top1_accuracy"] == pytest.approx(2 / 7)
    assert metrics["headline"]["llm:card"]["top1_accuracy"] == 1.0


def test_every_configuration_is_scored_on_the_same_files(root, backends):
    """The rules could sort the zip by name, but the headline leaves it out for them too."""
    evaluate(root, "--no-experiments")
    metrics = metrics_of(root)
    assert {row["n"] for row in metrics["headline"].values()} == {7}
    assert metrics["majority"]["headline"]["n"] == 7


def test_the_all_files_accuracy_counts_a_file_with_no_text_as_wrong(root, backends):
    """Wrong for every content-based configuration; the rules and the majority are scored on it."""
    evaluate(root, "--no-experiments")
    metrics = metrics_of(root)
    assert metrics["all_files"]["jev:card"] == {"n": 8, "top1_accuracy": pytest.approx(6 / 8)}
    assert metrics["all_files"]["llm:card"] == {"n": 8, "top1_accuracy": pytest.approx(7 / 8)}
    assert metrics["all_files"]["embed:card"] == {"n": 8, "top1_accuracy": pytest.approx(3 / 8)}
    assert metrics["all_files"]["rules:card"] == {"n": 8, "top1_accuracy": 1.0}
    assert "Test files with no extractable text: 1 of 8." in report_of(root)
    assert "| jev:card | 7 | 85.7% | 8 | 75.0% |" in report_of(root)


def test_the_majority_class_row_always_picks_the_largest_gold_group(root, backends):
    """Homework has four of the eight test files, three of the seven with text."""
    evaluate(root, "--configs", "rules:card")
    majority = metrics_of(root)["majority"]
    assert majority["group"] == "homework"
    assert majority["headline"]["top1_accuracy"] == pytest.approx(3 / 7)
    assert majority["headline"]["placed_share"] == 1.0
    assert majority["all_files"] == {"group": "homework", "n": 8, "top1_accuracy": 0.5}
    assert "| majority class (Homework) | 7 | 42.9% |" in report_of(root)
    assert "| majority class (Homework) | 7 | 42.9% | 8 | 50.0% |" in report_of(root)


def test_tuned_cutoffs_are_shown_next_to_the_defaults(root, backends):
    """Tuning on the three dev files lowers the top cutoff to 0.55, which places lab1.py."""
    evaluate(root, "--no-experiments")
    metrics = metrics_of(root)
    tuned = metrics["tuned"]["jev:card"]
    assert tuned["dev_n"] == 3
    assert tuned["cutoffs"] == {"fits_any_below": 0.5, "top_below": 0.55, "close_margin": 0.15}
    assert tuned["metrics"]["placed_share"] == pytest.approx(6 / 7)
    assert tuned["metrics"]["placed_accuracy"] == pytest.approx(5 / 6)
    assert tuned["metrics"]["top1_accuracy"] == pytest.approx(6 / 7)
    # The defaults stay what the headline reports.
    assert metrics["headline"]["jev:card"]["placed_share"] == pytest.approx(5 / 7)
    # Top 0.55 or under places all three dev files, whatever the fits_any cutoff: 19 x 12 pairs.
    assert (tuned["tied_pairs"], tuned["scored_pairs"], tuned["tunable"]) == (228, 361, True)
    report = report_of(root)
    assert (
        "| jev:card | default | 0.50 | 0.60 | 71.4% | 80.0% | 28.6% | 100.0% | n/a |  |"
    ) in report
    assert (
        "| jev:card | tuned on dev (n = 3) | 0.50 | 0.55 | 85.7% | 83.3% | 14.3% | 100.0% "
        "| 0.0% | 228 of 361 |"
    ) in report
    assert "not tunable on this dev split" not in report
    assert "rules:card | default" not in report


def test_a_dev_split_that_cannot_tell_the_cutoffs_apart_is_not_tunable(root, backends):
    """Every dev file sure and right: all 361 pairs tie, so no tuned row repeats the defaults."""
    backends.jev.answers["quiz0.txt"] = spread("exams", 0.9)
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    metrics = metrics_of(root)
    tuned = metrics["tuned"]["jev:card"]
    assert tuned["cutoffs"] == metrics["cutoffs"]
    assert (tuned["tied_pairs"], tuned["scored_pairs"], tuned["tunable"]) == (361, 361, False)
    report = report_of(root)
    assert (
        "| jev:card | tuned on dev (n = 3) | not tunable on this dev split |  |  |  |  |  |  "
        "| 361 of 361 |"
    ) in report
    assert "| jev:card | tuned on dev (n = 3) | 0.50 | 0.60 |" not in report
    assert (
        '"not tunable on this dev split" means the defaults are among several pairs that tie '
        "for best there."
    ) in report
    assert (
        "Files in the dev split: 3 (3 with text). The tuned cutoffs rest on them. They could "
        "not tune the cutoffs of jev:card: several pairs of cutoffs tie for best on the dev "
        "split, the defaults among them."
    ) in metrics["caveats"]


def test_cost_and_latency_come_from_what_the_backends_report(root, backends):
    """Jev is billed on input at the vault's price, the LLM on both, the rest are free."""
    backends.made["jev"] = MeteredBackend(1000, answers=dict(JEV))
    backends.made["llm"] = MeteredBackend(1000, 100, default=spread("homework", 1.0))
    evaluate(root, "--configs", "jev:card,rules:card,llm:card", "--no-experiments")
    headline = metrics_of(root)["headline"]
    assert headline["jev:card"]["mean_input_tokens"] == 1000
    assert headline["jev:card"]["usd_per_file"] == pytest.approx(1000 / 1e6 * 0.042)
    assert headline["jev:card"]["latency_p50_ms"] == 250.0
    assert headline["llm:card"]["usd_per_file"] == pytest.approx(
        1000 / 1e6 * llm.USD_PER_MILLION_INPUT_TOKENS
        + 100 / 1e6 * llm.USD_PER_MILLION_OUTPUT_TOKENS
    )
    assert headline["rules:card"]["mean_input_tokens"] is None
    assert headline["rules:card"]["usd_per_file"] == 0.0
    assert "| jev:card | 7 | 85.7% |" in report_of(root)
    assert "| 1000 | $0.000042 | 250 | 250 |" in report_of(root)


def test_the_report_holds_the_headline_and_the_go_no_go_tables(root, backends):
    """Plain markdown tables, the three go/no-go rows with their numbers, and no em dash."""
    assert evaluate(root) == 0
    report = report_of(root)
    assert report.startswith("# Phase 0 evaluation report\n")
    assert (
        "## Headline\n\nTest split, files with text, default cutoffs.\n\n| Configuration | n |"
        in (report)
    )
    assert "| jev:card | 7 | 85.7% | 0.867 | 71.4% | 80.0% | 28.6% | 100.0% | n/a | " in report
    assert "| rules:card | 7 | 100.0% | 1.000 |  |  |  |  |  |  | n/a | $0.000000 |" in report
    assert "## Go/no-go" in report
    assert (
        "| Card accuracy of 90% or more, above both non-LLM baselines | Start Phase 1 as designed "
        "| jev:card 85.7% (n 7); rules:card 100.0% (n 7); embed:card 42.9% (n 7); "
        "embed-chunks:card 28.6% (n 7) | no |"
    ) in report
    assert (
        "| Full text or window vote beats the card by 5 points or more | Use that input for D1 "
        "| jev:full_text 85.7% (n 7); jev:window_vote 85.7% (n 7); jev:card 85.7% (n 7) | no |"
    ) in report
    # The card leads both embedding rows, on 4 to 1 and on 4 to 0 files: too few to tell.
    assert (
        "| Jev doesn't beat embedding similarity "
        "| Rethink Jev's role on the write path before Phase 1 "
        "| jev:card 85.7% (n 7); embed:card 42.9% (n 7); embed-chunks:card 28.6% (n 7) "
        "| not separable (jev:card against embed:card: 4 to 1 on 5 discordant files; "
        "jev:card against embed-chunks:card: 4 to 0 on 4 discordant files) |"
    ) in report
    assert (
        "| jev:card against embed:card | above | 7 | 6 to 3 | 5 (4 to 1) | 0.375 | not separable |"
    ) in report
    assert (
        "| jev:full_text against jev:card | above by 5 points | 7 | 6 to 6 | 0 | 1.000 | no |"
    ) in report
    for heading in ("All test files", "Tuned cutoffs", "Per group: jev:card", "Window vote"):
        assert f"\n## {heading}\n" in report
    assert "\n## Experiments\n" in report
    assert "\n## Caveats\n" in report
    assert "It does not make the go/no-go decision." in report
    assert chr(0x2014) not in report  # an em dash
    assert "None" not in report


def test_the_per_group_table_and_the_variant_differences(root, backends):
    """jev:card per group, then each other variant minus the card, in points."""
    backends.made["jev"] = by_variant = FakeBackend(answers=dict(JEV))
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    # The names-only variant gets hw2_soln.txt right as well, and lec2.md wrong.
    by_variant.answers["hw2_soln.txt"] = spread("homework", 0.8)
    by_variant.answers["lec2.md"] = spread("exams", 0.8)
    evaluate(root, "--configs", "jev:names_only", "--no-experiments")
    metrics = metrics_of(root)
    assert metrics["per_group"]["lecture-notes"] == {
        "n": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    difference = metrics["variant_differences"]["jev:names_only"]
    assert difference["top1_accuracy"] == pytest.approx(0.0)  # six of seven both times
    assert difference["recall"] == pytest.approx(
        {"exams": 0.0, "homework": 100 / 3, "lecture-notes": -50.0}
    )
    # hw2_soln.txt is right for the variant only, lec2.md for the card only.
    assert difference["paired"] == {
        "n": 7,
        "first_right": 6,
        "second_right": 6,
        "only_first": 1,
        "only_second": 1,
        "p_value": 1.0,
    }
    report = report_of(root)
    assert "| Homework | 3 | 100.0% | 66.7% | 0.800 |" in report
    assert "| jev:names_only | +0.0 | +0.0 | +33.3 | -50.0 | 2 (1 to 1) |" in report
    assert "the card is at the ceiling" not in report.split("## Go/no-go")[0]


# The report from hand-made inputs

A_OR_B = [Group(id="a", name="A", covers="a things"), Group(id="b", name="B", covers="b things")]


def gold_label(file_id, gold="a", split="test"):
    """A gold label for a made-up file."""
    return GoldLabel(
        file_id=file_id,
        gold_group=gold,
        source_url="https://example.org",
        source_type="made up",
        license="CC0",
        sha256="0" * 64,
        split=split,
    )


def answered(name, right, total=20, **extra):
    """A configuration that answered files f0, f1, and so on, the first `right` of them right."""
    predictions = [
        Prediction(
            file_id=f"f{number}",
            gold="a",
            probs={"a": 0.9, "b": 0.1} if number < right else {"a": 0.1, "b": 0.9},
            fits_any=0.9,
        )
        for number in range(total)
    ]
    return Answers(name=name, backend=name.split(":")[0], predictions=predictions, **extra)


def right_on(name, right, total=20, **extra):
    """A configuration that answered files f0 to f(total - 1), right on the numbers given."""
    right = set(right)
    predictions = [
        Prediction(
            file_id=f"f{number}",
            gold="a",
            probs={"a": 0.9, "b": 0.1} if number in right else {"a": 0.1, "b": 0.9},
            fits_any=0.9,
        )
        for number in range(total)
    ]
    return Answers(name=name, backend=name.split(":")[0], predictions=predictions, **extra)


def inputs_for(*answers, total=20, **extra):
    """Report inputs over `total` test files whose gold group is "a"."""
    labels = extra.pop("labels", None) or [gold_label(f"f{number}") for number in range(total)]
    return Inputs(
        config=Config(),
        set_version=1,
        groups=A_OR_B,
        labels=labels,
        no_text=extra.pop("no_text", []),
        answers=list(answers),
        **extra,
    )


def go_no_go(*answers, total=20):
    """The three go/no-go rows for these answers."""
    return build_metrics(inputs_for(*answers, total=total))["go_no_go"]


def met(*answers, total=20):
    """Whether each of the three go/no-go conditions is met for these answers."""
    return [row["met"] for row in go_no_go(*answers, total=total)]


def verdicts(*answers, total=20):
    """The verdict of each of the three go/no-go rows for these answers."""
    return [row["verdict"] for row in go_no_go(*answers, total=total)]


def test_go_no_go_when_the_card_wins_everywhere():
    """95% on the card, each baseline at least six files behind, full text six files ahead."""
    assert met(
        answered("jev:card", 114, total=120),
        answered("rules:card", 100, total=120),
        answered("embed:card", 60, total=120),
        answered("embed-chunks:card", 70, total=120),
        answered("jev:full_text", 120, total=120),
        answered("jev:window_vote", 114, total=120),
        total=120,
    ) == [True, True, False]


def test_go_no_go_counts_exactly_90_percent_and_exactly_5_points():
    """Both thresholds are "or more": 108 of 120 is 90%, and 114 of 120 is 5 points above it."""
    answers = [
        answered("jev:card", 108, total=120),
        answered("rules:card", 100, total=120),
        answered("embed:card", 100, total=120),
        answered("embed-chunks:card", 100, total=120),
        answered("jev:full_text", 114, total=120),
        answered("jev:window_vote", 108, total=120),
    ]
    assert met(*answers, total=120) == [True, True, False]
    # One file short of either threshold is a plain no.
    answers[0] = answered("jev:card", 107, total=120)
    assert verdicts(*answers, total=120)[0] == "no"
    answers[0] = answered("jev:card", 108, total=120)
    answers[4] = answered("jev:full_text", 113, total=120)
    assert verdicts(*answers, total=120)[1] == "no"


def test_go_no_go_needs_the_card_strictly_above_each_baseline():
    """A tie with the rules is not above them. A tie with an embedding row is "doesn't beat"."""
    baselines = [answered("embed:card", 19), answered("embed-chunks:card", 3)]
    tied = met(answered("jev:card", 19), answered("rules:card", 19), *baselines)
    assert tied == [False, None, True]


def test_go_no_go_below_90_percent_or_under_5_points():
    """85% is not enough, and a 4 point gain is not enough."""
    assert met(
        answered("jev:card", 20, total=25),
        answered("rules:card", 1, total=25),
        answered("embed:card", 1, total=25),
        answered("embed-chunks:card", 1, total=25),
        answered("jev:full_text", 21, total=25),
        answered("jev:window_vote", 19, total=25),
        total=25,
    ) == [False, False, False]


def test_go_no_go_is_undecided_only_while_a_missing_number_could_change_it():
    """A failed check settles "no"; a passed one settles "any of"; else a gap leaves n/a."""
    card = answered("jev:card", 19)
    # Everything at hand passes, but one embedding row is missing.
    assert met(card, answered("rules:card", 1), answered("embed:card", 1)) == [None, None, None]
    assert verdicts(card, answered("rules:card", 1), answered("embed:card", 1)) == ["n/a"] * 3
    # The rules already beat the card, so the first row is settled.
    assert met(card, answered("rules:card", 20), answered("embed:card", 1))[0] is False
    # One embedding row at or above the card settles the third row.
    assert met(card, answered("embed:card", 19))[2] is True
    # One long input that beats the card, six files ahead, settles the second row.
    assert met(answered("jev:card", 10), answered("jev:window_vote", 16))[1] is True
    assert met(answered("jev:card", 10), answered("jev:window_vote", 10))[1] is None
    # Without the card, nothing can be told.
    assert met(answered("rules:card", 10), answered("embed:card", 10)) == [None, None, None]


def test_go_no_go_does_not_call_a_lead_of_one_file_met():
    """73 of 73 against 72 of 73 is one discordant file: the card is not shown to be above."""
    answers = [
        answered("jev:card", 73, total=73),
        answered("rules:card", 72, total=73),
        answered("embed:card", 43, total=73),
        answered("embed-chunks:card", 40, total=73),
    ]
    metrics = build_metrics(inputs_for(*answers, total=73))
    first = metrics["go_no_go"][0]
    assert (first["verdict"], first["met"]) == ("not separable", None)
    assert first["comparisons"][0] == {
        "first": "jev:card",
        "second": "rules:card",
        "margin_points": 0.0,
        "n": 73,
        "first_right": 73,
        "second_right": 72,
        "only_first": 1,
        "only_second": 0,
        "p_value": 1.0,
        "outcome": "not separable",
    }
    assert [item["outcome"] for item in first["comparisons"][1:]] == ["yes", "yes"]
    # Thirty and thirty-three files, all one way, do separate the card from the embeddings.
    assert (metrics["go_no_go"][2]["verdict"], metrics["go_no_go"][2]["met"]) == ("no", False)
    report = render_report(metrics)
    assert (
        "embed-chunks:card 54.8% (n 73) "
        "| not separable (jev:card against rules:card: 1 to 0 on 1 discordant file) |"
    ) in report
    assert "| jev:card against rules:card | above | 73 | 73 to 72 | 1 (1 to 0) | 1.000 |" in report
    assert (
        "| jev:card against embed:card | above | 73 | 73 to 43 | 30 (30 to 0) | <0.001 | yes |"
    ) in report
    assert (
        "Go/no-go comparisons that rest on fewer than 6 discordant files (files only one of the "
        "two got right): jev:card against rules:card, 1 (1 to 0). It takes 6 such files, all "
        "one way, for an exact two-sided sign test to separate two configurations at p under "
        "0.05. Fewer cannot tell them apart, so one file falling the other way could change "
        "what these comparisons show."
    ) in metrics["caveats"]
    # The section says what the verdicts mean, and what the test behind them takes for granted.
    assert (
        "counts as met only when the two are separable: an exact two-sided sign test on the "
        "discordant files (the files only one of the two got right) gives p under 0.05. That "
        "takes at least 6 discordant files, all one way. The test counts every file as "
        "independent"
    ) in report
    assert '"not separable" (the numbers meet it, but the discordant files are too few' in report
    # One file the other way is a tie, a plain no: neither reading is a yes.
    answers[1] = answered("rules:card", 73, total=73)
    assert verdicts(*answers, total=73)[0] == "no"


def test_go_no_go_takes_six_discordant_files_all_one_way():
    """Five to none has p 0.0625 and is not separable; six to none has p 0.031 and is."""
    baselines = [answered("embed:card", 1), answered("embed-chunks:card", 1)]
    five = go_no_go(answered("jev:card", 19), answered("rules:card", 14), *baselines)[0]
    assert (five["verdict"], five["comparisons"][0]["p_value"]) == ("not separable", 0.0625)
    six = go_no_go(answered("jev:card", 19), answered("rules:card", 13), *baselines)[0]
    assert (six["verdict"], six["comparisons"][0]["p_value"]) == ("yes", 0.03125)
    # Seven against one is eight discordant files, but not one-sided enough: p 0.07.
    card = right_on("jev:card", range(1, 20))
    rules = right_on("rules:card", [0, *range(8, 20)])
    mixed = go_no_go(card, rules, *baselines)[0]["comparisons"][0]
    assert (mixed["only_first"], mixed["only_second"], mixed["p_value"]) == (7, 1, 0.0703125)
    assert mixed["outcome"] == "not separable"


def test_go_no_go_compares_two_rows_on_the_files_both_answered():
    """The card failed on the one file the rules got wrong: on the other 19 the two tie."""
    card = answered("jev:card", 19, total=19, errors={"f19": "timeout"})
    rules = right_on("rules:card", range(19))
    baselines = [answered("embed:card", 5), answered("embed-chunks:card", 5)]
    metrics = build_metrics(inputs_for(card, rules, *baselines))
    # The two accuracies alone would say 100% against 95%.
    assert metrics["headline"]["jev:card"]["top1_accuracy"] == 1.0
    assert metrics["headline"]["rules:card"]["top1_accuracy"] == 0.95
    first = metrics["go_no_go"][0]
    compared = first["comparisons"][0]
    assert (compared["n"], compared["first_right"], compared["second_right"]) == (19, 19, 19)
    assert (compared["only_first"], compared["only_second"]) == (0, 0)
    assert (first["verdict"], first["met"]) == ("no", False)
    assert "| jev:card against rules:card | above | 19 | 19 to 19 | 0 | 1.000 | no |" in (
        render_report(metrics)
    )


def test_go_no_go_cannot_tell_a_gain_when_the_card_is_at_the_ceiling():
    """At 73 of 73 no input can gain 5 points, so "no" would be arithmetic, not a result."""
    answers = [
        answered("jev:card", 73, total=73),
        answered("jev:full_text", 73, total=73),
        answered("jev:window_vote", 73, total=73),
    ]
    metrics = build_metrics(inputs_for(*answers, total=73))
    assert metrics["ceiling"] == {"right": 73, "n": 73, "margin_points": 5.0, "at_ceiling": True}
    second = metrics["go_no_go"][1]
    assert second["verdict"] == "cannot be told: the card is at the ceiling"
    assert second["met"] is None
    assert [item["outcome"] for item in second["comparisons"]] == [second["verdict"]] * 2
    report = render_report(metrics)
    assert ("jev:card 100.0% (n 73) | cannot be told: the card is at the ceiling |") in report
    assert (
        "| jev:full_text against jev:card | above by 5 points | 73 | 73 to 73 | 0 | 1.000 "
        "| cannot be told: the card is at the ceiling |"
    ) in report
    assert (
        "jev:card is right on 73 of its 73 test files, which leaves another input less than "
        "the 5 points of the go/no-go margin to gain. Whether full text, window vote or any "
        "other variant does better than the card cannot be told on these files: a difference "
        "of +0.0 is not evidence that more text does not help."
    ) in metrics["caveats"]
    # The variant comparison says the same next to its +0.0 cells.
    assert "| jev:full_text | +0.0 | +0.0 | 0 |" in report
    assert (
        "Whether a variant does better than jev:card cannot be told: the card is at the "
        "ceiling (73 of 73 right)."
    ) in report


def test_the_ceiling_starts_where_the_card_leaves_less_than_the_margin():
    """Three wrong of 73 leaves 4.1 points, four wrong leaves 5.5: enough room to be beaten."""
    perfect = [
        answered("jev:full_text", 73, total=73),
        answered("jev:window_vote", 73, total=73),
    ]
    near = build_metrics(inputs_for(answered("jev:card", 70, total=73), *perfect, total=73))
    assert near["ceiling"]["at_ceiling"] is True
    assert near["go_no_go"][1]["verdict"] == "cannot be told: the card is at the ceiling"
    room = build_metrics(inputs_for(answered("jev:card", 69, total=73), *perfect, total=73))
    assert room["ceiling"]["at_ceiling"] is False
    # Four files ahead is the margin, but four discordant files cannot carry it.
    assert room["go_no_go"][1]["verdict"] == "not separable"
    assert "the card is at the ceiling" not in "".join(room["caveats"])
    # Exactly 5 points of room (19 of 20) is not the ceiling: a perfect input could still win.
    assert build_metrics(inputs_for(answered("jev:card", 19)))["ceiling"]["at_ceiling"] is False


def test_at_the_ceiling_a_clear_loss_is_still_a_no():
    """Both long inputs 13 files behind the card: that they do not beat it can be told."""
    answers = [
        answered("jev:card", 73, total=73),
        answered("jev:full_text", 60, total=73),
        answered("jev:window_vote", 60, total=73),
    ]
    assert verdicts(*answers, total=73)[1] == "no"
    # With one of them level with the card, the row is back to what the ceiling hides.
    answers[2] = answered("jev:window_vote", 73, total=73)
    assert verdicts(*answers, total=73)[1] == "cannot be told: the card is at the ceiling"
    # That holds whatever a missing long input would say. With both missing it is n/a.
    assert verdicts(answers[0], answers[2], total=73)[1] == (
        "cannot be told: the card is at the ceiling"
    )
    assert verdicts(answers[0], answers[1], total=73)[1] == "n/a"
    assert verdicts(answers[0], total=73)[1] == "n/a"


def test_a_configuration_is_scored_on_the_files_it_answered_and_the_caveats_say_so():
    """An error or a file not reached leaves the row short, and that is flagged."""
    card = answered("jev:card", 15, total=17, errors={"f17": "timeout", "f18": "timeout"})
    metrics = build_metrics(inputs_for(card, answered("rules:card", 10)))
    assert metrics["headline"]["jev:card"]["n"] == 17
    assert metrics["headline"]["rules:card"]["n"] == 20
    assert metrics["configurations"]["jev:card"] == {
        "status": "partial",
        "pending": None,
        "expected": 20,
        "answered": 17,
        "errors": 2,
        "not_asked": 1,
    }
    assert (
        "jev:card answered 17 of 20 files, so its numbers cover only those. Failed: 2 (f17, f18); "
        "the messages are in metrics.json under errors. Not asked yet: 1."
    ) in metrics["caveats"]
    assert metrics["errors"] == {"jev:card": {"f17": "timeout", "f18": "timeout"}}


def test_caveats_name_small_groups_the_dev_split_and_files_with_no_text():
    """Each caveat is made from the data: counts per group, the dev size, the files left out."""
    labels = [
        *(gold_label(f"f{number}") for number in range(6)),
        gold_label("b1", gold="b"),
        gold_label("scan.pdf", gold="b"),
        gold_label("d1", split="dev"),
        gold_label("d2", split="dev"),
    ]
    metrics = build_metrics(
        inputs_for(answered("jev:card", 5, total=6), labels=labels, no_text=["scan.pdf"])
    )
    assert metrics["files"]["test_gold"] == {"a": 6, "b": 1}
    assert metrics["caveats"][:3] == [
        (
            "Groups with fewer than 5 gold files among the test files with text: B 1. Their "
            "per-group numbers, and their share of the macro-F1, rest on very few files."
        ),
        (
            # The card answered neither dev file, so every pair of cutoffs ties.
            "Files in the dev split: 2 (2 with text). The tuned cutoffs rest on them. They "
            "could not tune the cutoffs of jev:card: several pairs of cutoffs tie for best on "
            "the dev split, the defaults among them."
        ),
        (
            "Files with no extractable text: 1 (1 in the test split). No content-based "
            "configuration is asked about them. The headline leaves them out; the all-files "
            "table counts them as wrong."
        ),
    ]
    # f0 to f5 answered, b1 not: seven test files with text, plus the one without.
    assert metrics["all_files"]["jev:card"] == {"n": 7, "top1_accuracy": pytest.approx(5 / 7)}


def test_a_report_with_no_answers_at_all_still_renders():
    """Before anything is cached, the report says what is missing and shows no numbers."""
    names = ["jev:card", "rules:card", "embed:card"]
    empty = [Answers(name=name, backend=name.split(":")[0]) for name in names]
    metrics = build_metrics(inputs_for(*empty))
    assert metrics["headline"] == dict.fromkeys(names)
    assert metrics["tuned"] == {}
    assert metrics["per_group"] is None
    assert metrics["window_vote"] is None
    report = render_report(metrics)
    assert "No row for: jev:card (not run), rules:card (not run), embed:card (not run)." in report
    assert "| majority class (A) | 20 | 100.0% |" in report
    assert "No Jev configuration has answers yet." in report
    assert "jev:card has no answers yet." in report
    assert "jev:window_vote has no answers yet." in report
    assert "| jev:card n/a; rules:card n/a; embed:card n/a; embed-chunks:card n/a | n/a |" in report
    assert "None" not in report


def test_a_table_cell_cannot_break_out_of_its_column():
    """A pipe in a group name is escaped."""
    groups = [Group(id="a", name="A | B", covers="both"), A_OR_B[1]]
    inputs = inputs_for(answered("jev:card", 20)).model_copy(update={"groups": groups})
    assert "| A \\| B | 20 | 100.0% | 100.0% | 1.000 |" in render_report(build_metrics(inputs))


# What the report does not print as a measurement


def test_one_hot_and_tempered_backends_get_no_placement_or_calibration_cells(root, backends):
    """Only Jev's probabilities are its own: the other rows keep top-1, cost and latency."""
    assert evaluate(root, "--no-experiments") == 0
    metrics = metrics_of(root)
    fixed = ["rules:card", "embed:card", "embed-chunks:card", "llm:card"]
    assert metrics["by_construction"]["probabilities_not_measured"] == fixed
    # The numbers are still in metrics.json: the stand-in for the embeddings places every file.
    assert metrics["headline"]["embed:card"]["placed_share"] == 1.0
    report = report_of(root)
    assert "| embed:card | 7 | 42.9% | 0.200 |  |  |  |  |  |  | n/a | $0.000000 |" in report
    assert "| llm:card | 7 | 100.0% | 1.000 |  |  |  |  |  |  | n/a |" in report
    assert "| majority class (Homework) | 7 | 42.9% | 0.200 |  |  |  |  |  |  | n/a |" in report
    assert "| jev:card | 7 | 85.7% | 0.867 | 71.4% | 80.0% | 28.6% | 100.0% | n/a | " in report
    assert (
        "The cells from Placed to ECE are left empty for rules:card, embed:card, "
        "embed-chunks:card, llm:card and the majority class."
    ) in report
    assert "is fixed by that construction and not measured." in report
    assert "Files with no pick at all" not in report


def test_a_file_the_rules_have_no_pick_for_is_named_under_the_headline():
    """The empty cells would hide that the rules left a file out, so the footnote counts it."""
    rules = right_on("rules:card", range(19))
    rules.predictions[19] = rules.predictions[19].model_copy(update={"probs": {}, "fits_any": 0.0})
    metrics = build_metrics(inputs_for(answered("jev:card", 20), rules))
    assert metrics["by_construction"]["no_pick"] == {"rules:card": 1}
    assert metrics["headline"]["rules:card"]["top1_accuracy"] == 0.95
    assert "Files with no pick at all, counted as wrong: rules:card for 1 of 20." in (
        render_report(metrics)
    )


def test_close_calls_show_n_a_under_cutoffs_that_cannot_flag_one(root, backends):
    """Under 0.60 and 0.15 the flag has no room; with the top cutoff at 0.55 it is measured."""
    add_file(root, "quiz1.txt", "Quiz 1. Ten short questions.\n", "exams")
    backends.jev.answers["quiz1.txt"] = ({"exams": 0.55, "homework": 0.45, "lecture-notes": 0}, 0.9)
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    metrics = metrics_of(root)
    assert metrics["by_construction"]["close_call_can_fire"] is False
    assert metrics["headline"]["jev:card"]["close_call_share"] == 0.0
    report = report_of(root)
    assert "| jev:card | 8 | 87.5% | 0.886 | 62.5% | 80.0% | 37.5% | 100.0% | n/a |" in report
    assert (
        "Close calls show n/a because the flag cannot fire under these cutoffs. It needs a top "
        "group at or over 0.60 with a second group within 0.15 of it, and two such "
        "probabilities add up to more than 1.05, which probabilities that sum to 1 have no "
        "room for. A 0% there would not be a measurement."
    ) in report

    Config.load(root).model_copy(update={"unsorted_top_below": 0.55}).save(root)
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    metrics = metrics_of(root)
    assert metrics["by_construction"]["close_call_can_fire"] is True
    assert metrics["headline"]["jev:card"]["close_call_share"] == pytest.approx(1 / 8)
    report = report_of(root)
    assert "| jev:card | 8 | 87.5% | 0.886 | 87.5% | 85.7% | 12.5% | 100.0% | 12.5% |" in report
    assert "Close calls show n/a" not in report


def test_an_ece_over_files_that_are_all_right_is_named_for_what_it_is():
    """With nothing wrong, ECE is only how far the confidence stays under 1."""
    metrics = build_metrics(inputs_for(answered("jev:card", 20), answered("jev:full_text", 19)))
    assert metrics["headline"]["jev:card"]["ece"] == pytest.approx(0.1)
    report = render_report(metrics)
    assert (
        "Every file is right for jev:card, so the ECE in those rows is only the mean of 1 "
        "minus the top probability: how far the confidence stays under 1. It does not test "
        "calibration."
    ) in report
    assert "Every file is right" not in render_report(
        build_metrics(inputs_for(answered("jev:card", 19)))
    )


def test_the_majority_class_is_picked_on_the_files_it_is_scored_on():
    """With text: three A and four B. Without: two more A. B wins the headline, A the rest."""
    labels = [
        *(gold_label(f"a{number}") for number in range(5)),
        *(gold_label(f"b{number}", gold="b") for number in range(4)),
    ]
    card = Answers(name="jev:card", backend="jev")
    metrics = build_metrics(inputs_for(card, labels=labels, no_text=["a3", "a4"]))
    majority = metrics["majority"]
    assert majority["group"] == "b"
    assert majority["headline"]["top1_accuracy"] == pytest.approx(4 / 7)
    assert majority["all_files"] == {"group": "a", "n": 9, "top1_accuracy": pytest.approx(5 / 9)}
    report = render_report(metrics)
    assert "| majority class (B) | 7 | 57.1% |" in report
    assert "| majority class (B with text, A over all files) | 7 | 57.1% | 9 | 55.6% |" in report
    assert (
        "The majority class gives every file the largest gold group among the files of its "
        "row, so it is the best a constant answer can do on them."
    ) in report


# The sensitivity line, the label signal and the held-out wording


def test_the_headline_comes_with_its_sensitivity_to_the_wording():
    """The card under the vault's descriptions, names only and each edit, on the same files."""
    edits = [
        Edit(name="first-draft", group_id="a", covers="a things, roughly"),
        Edit(name="not-run", group_id="b", covers="b things, roughly"),
    ]
    card = answered("jev:card", 20)
    names_only = answered("jev:names_only", 17)
    # The edit pass has no answer for f19, so every wording is counted on f0 to f18.
    edited = answered("jev:card@edit-first-draft", 18, total=19)
    rules = answered("rules:card", 16)
    metrics = build_metrics(inputs_for(card, names_only, rules, passes=[edited], edits=edits))
    assert metrics["sensitivity"] == {
        "n": 19,
        "rows": [
            {"wording": "the vault's descriptions", "name": "jev:card", "right": 19},
            {"wording": "group names only", "name": "jev:names_only", "right": 17},
            {"wording": "edit first-draft", "name": "jev:card@edit-first-draft", "right": 18},
        ],
        "rules_right": 16,
    }
    assert (
        "Sensitivity to the wording of the groups: top-1 of jev:card on the same 19 test files, "
        "by wording: the vault's descriptions 100.0% (19 of 19); group names only 89.5% (17 of "
        "19); edit first-draft 94.7% (18 of 19). For scale, rules:card is right on 16 of them. "
        "The headline is the first of these: it holds for the vault's own descriptions, not "
        "for any wording of the same groups."
    ) in render_report(metrics)


def test_no_sensitivity_line_without_another_wording():
    """The card alone has nothing to be compared with, and neither has a missing card."""
    alone = build_metrics(inputs_for(answered("jev:card", 20)))
    assert alone["sensitivity"] is None
    assert "Sensitivity to the wording" not in render_report(alone)
    assert build_metrics(inputs_for(answered("jev:names_only", 17)))["sensitivity"] is None
    # Without the rules on the same files, the line gives no number for scale.
    two = build_metrics(inputs_for(answered("jev:card", 20), answered("jev:names_only", 17)))
    assert two["sensitivity"]["rules_right"] is None
    assert "group names only 85.0% (17 of 20)" in render_report(two)
    assert "For scale" not in render_report(two)


def test_a_folder_that_gives_the_group_away_is_a_caveat():
    """labs/ holds three files, all A: its name tells the group. notes/ and solo/ do not."""
    labels = [
        gold_label("labs/a.py"),
        gold_label("labs/b.zip"),
        gold_label("labs/c.py", split="dev"),
        gold_label("notes/x.md", gold="b"),
        gold_label("notes/y.md"),
        gold_label("solo/one.txt"),
        gold_label("top.txt", gold="b"),
    ]
    test_with_text = ["labs/a.py", "notes/x.md", "notes/y.md", "solo/one.txt", "top.txt"]
    rules = Answers(
        name="rules:card",
        backend="rules",
        predictions=[
            Prediction(file_id=file_id, gold=gold, probs={"a": 1.0, "b": 0.0}, fits_any=1.0)
            for file_id, gold in zip(test_with_text, ["a", "b", "a", "a", "b"])
        ],
    )
    metrics = build_metrics(inputs_for(rules, labels=labels, no_text=["labs/b.zip"]))
    assert metrics["label_signal"] == {
        "folders": {"labs": {"gold_group": "a", "files": 3}},
        "test_with_text": 1,
        "rules": {"n": 5, "right": 3},
    }
    assert (
        "Folders whose files all have the same gold group: labs/ (A, 3 files). Test files with "
        "text under such a folder: 1 of 5. The folder name in their path gives the group away "
        "to every configuration that is shown the path. The file-name rules (rules:card), "
        "which read only names and paths, are right on 3 of the 5 test files with text they "
        "were scored on (60.0%)."
    ) in metrics["caveats"]


def test_no_folder_caveat_when_no_folder_gives_a_group_away():
    """Flat file ids, and a folder with mixed groups, tell nothing."""
    flat = build_metrics(inputs_for(answered("jev:card", 20)))
    assert flat["label_signal"] == {"folders": {}, "test_with_text": 0, "rules": None}
    assert not any("Folders whose files" in caveat for caveat in flat["caveats"])


def held_out_inputs(groups):
    """Report inputs with a cached pass without Exams, under the given full set of groups."""
    labels = [gold_label("e1", gold="exams"), gold_label("h1", gold="homework")]
    reduced = [
        Prediction(file_id="e1", gold="exams", probs={"homework": 0.9}, fits_any=0.2),
        Prediction(file_id="h1", gold="homework", probs={"homework": 0.9}, fits_any=0.9),
    ]
    full = [
        Prediction(file_id="e1", gold="exams", probs={"exams": 0.9}, fits_any=0.9),
        Prediction(file_id="h1", gold="homework", probs={"homework": 0.9}, fits_any=0.9),
    ]
    return Inputs(
        config=Config(),
        set_version=1,
        groups=groups,
        labels=labels,
        no_text=[],
        answers=[Answers(name="jev:card", backend="jev", predictions=full)],
        passes=[Answers(name="jev:card@without-exams", backend="jev", predictions=reduced)],
    )


def test_a_reduced_set_that_still_names_the_held_out_group_is_flagged():
    """A description that says "exams" tells the model those files are excluded."""
    exams = Group(id="exams", name="Exams", covers="quizzes and exams")
    homework = Group(id="homework", name="Homework", covers="problem sets", excludes="exams")
    projects = Group(id="projects", name="Projects", covers="project work")
    clean = build_metrics(held_out_inputs([exams, homework, projects]))
    # Homework's own excludes text is replaced in the reduced set, so nothing names Exams.
    assert clean["experiments"]["held_out_named_in"] == []
    assert "The reduced set's descriptions do not name the held-out group" in render_report(clean)
    assert not any("still name that group" in caveat for caveat in clean["caveats"])

    naming = projects.model_copy(update={"excludes": "anything for an exam"})
    flagged = build_metrics(held_out_inputs([exams, homework, naming]))
    assert flagged["experiments"]["held_out_named_in"] == ["Projects"]
    report = render_report(flagged)
    assert (
        "Careful: the reduced set's descriptions still name the held-out group (in Projects), "
        "which tells the model that its files are excluded."
    ) in report
    assert "do not name the held-out group" not in report
    assert (
        "In the passes without Exams, the descriptions of Projects still name that group. "
        "That tells the model its files are excluded, so the column with descriptions may be "
        "helped by it."
    ) in flagged["caveats"]
