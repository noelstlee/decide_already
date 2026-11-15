"""The evaluation dataset builder, tested offline with a tiny zip shaped like the OCW download."""

import hashlib
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from decide_already import vault
from decide_already.config import Config
from decide_already.eval import dataset
from decide_already.eval.groups import EVAL_GROUPS

GROUP_IDS = {group.id for group in EVAL_GROUPS}
ZIP_URL = "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/6.867-fall-2006.zip"
RAW_URL = "https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/"
BLOB_URL = "https://github.com/MITDeepLearning/introtodeeplearning/blob/"


def resource(number: int, file_name: str, types: list[str], parent=None, kind="Document") -> dict:
    """One resource's metadata, with the fields a real resources/<name>/data.json carries."""
    return {
        "content_type": "resource",
        "file": f"/courses/6-867-machine-learning-fall-2006/{number:032x}_{file_name}",
        "learning_resource_types": types,
        "license": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "ocw_type": "OCWFile",
        "parent_uid": parent,
        "resourcetype": kind,
        "title": file_name,
        "uid": f"resource-{number}",
    }


def page(name: str, title: str, types: list[str], content: str) -> dict:
    """One page's metadata, with the fields a real pages/<name>/data.json carries."""
    return {
        "content": content,
        "content_type": "page",
        "learning_resource_types": types,
        "license": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "ocw_type": "CourseSection",
        "title": title,
        "uid": f"page-{name}",
    }


# The tiny course. Three resources share the file name hw1.pdf (one in capitals), two carry
# types that fit no single group, one has no type, one is an image, one has no file in the zip.
RESOURCES = {
    "lec1": resource(1, "lec1.pdf", ["Lecture Notes"], parent="page-lecture-notes"),
    "lagrange": resource(2, "lagrange.pdf", ["Readings"]),
    "hw1": resource(3, "hw1.pdf", ["Problem Sets"], parent="page-assignments"),
    "hw1-2": resource(4, "hw1.pdf", ["Problem Set Solutions"], parent="page-assignments"),
    "hw1-upper": resource(5, "HW1.pdf", ["Exams"]),
    "p1": resource(6, "p1.zip", ["Problem Sets"], parent="page-assignments", kind="Other"),
    "plot": resource(7, "plot.m", ["Tools"], kind="Other"),
    "final_f01": resource(8, "final_f01.pdf", ["Exams"]),
    "final_f01soln": resource(9, "final_f01soln.pdf", ["Exam Solutions"]),
    "rec1": resource(10, "rec1.pdf", ["Recitation Notes"]),
    "mixed": resource(11, "practice.pdf", ["Exams", "Problem Sets"]),
    "untyped": resource(12, "notes.txt", [], kind="Other"),
    "photo": resource(13, "photo.jpg", [], kind="Image"),
    "lecture-video": resource(14, "lec1.mp4", ["Lecture Videos"], kind="Video"),
}
NOT_IN_ZIP = {"lecture-video"}

# Two pages have resources filed under them (index pages), one is blank.
PAGES = {
    "syllabus": page("syllabus", "Syllabus", [], "<h3>Grading</h3><p>Midterm 15%, final 25%.</p>"),
    "projects": page("projects", "Projects", ["Projects"], "<p>A four page write-up is due.</p>"),
    "lecture-notes": page(
        "lecture-notes", "Lecture Notes", ["Lecture Notes"], "<p>Lec 1 (PDF)</p>"
    ),
    "assignments": page(
        "assignments", "Assignments", ["Assignments"], "<p>Problem set 1 (PDF)</p>"
    ),
    "instructor-insights": page("instructor-insights", "Insights", [], "<p>How it was taught.</p>"),
    "blank": page("blank", "Blank", [], "  "),
}

CHROME = [
    "index.html",
    "data.json",
    "content_map.json",
    "sitemap.xml",
    "static_shared/js/main.js",
    "static_shared/css/main.css",
    "static_shared/images/logo.svg",
    "static_resources/index.html",
    "static_resources/data.json",
    "static_resources/00000000000000000000000000000063_orphan.pdf",
    "pages/index.html",
    "pages/data.json",
    "resources/lecture-notes/index.html",
    "resources/lecture-notes/index.xml",
    "download/index.html",
]

# What the tiny course should give, in the order select_ocw returns it.
OCW_NAMES = [
    "final_f01.pdf",
    "final_f01soln.pdf",
    "hw1.pdf",
    "hw1.pdf",
    "HW1.pdf",
    "lagrange.pdf",
    "lec1.pdf",
    "practice.pdf",
    "p1.zip",
    "plot.m",
    "rec1.pdf",
    "notes.txt",
    "instructor-insights.html",
    "projects.html",
    "syllabus.html",
]
OCW_FLAT_NAMES = [*OCW_NAMES[:3], "hw1-2.pdf", "HW1-3.pdf", *OCW_NAMES[5:]]


def make_course_zip(path: Path) -> Path:
    """Write a tiny course zip with the layout of the real OCW download."""
    with zipfile.ZipFile(path, "w") as course:
        for folder in ("resources/", "pages/", "static_resources/", "static_shared/"):
            course.writestr(folder, "")
        for member in CHROME:
            course.writestr(member, '{"content": ""}' if member.endswith(".json") else "chrome")
        for name, meta in RESOURCES.items():
            course.writestr(f"resources/{name}/data.json", json.dumps(meta))
            course.writestr(f"resources/{name}/index.html", "<html><nav>site chrome</nav></html>")
            if name not in NOT_IN_ZIP:
                stored = meta["file"].rsplit("/", 1)[-1]
                course.writestr(f"static_resources/{stored}", f"bytes of {name}")
        for name, meta in PAGES.items():
            course.writestr(f"pages/{name}/data.json", json.dumps(meta))
            course.writestr(f"pages/{name}/index.html", "<html><nav>site chrome</nav></html>")
    return path


class FakeFetch:
    """Stands in for dataset.download: serves the tiny course zip and made-up lab files."""

    def __init__(self, course_zip: Path):
        """Remember which zip to serve and start an empty list of fetched URLs."""
        self.course_zip = course_zip
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        """Write the answer for one URL into dest, as download would."""
        self.urls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url == ZIP_URL:
            dest.write_bytes(self.course_zip.read_bytes())
        else:
            prefix = RAW_URL + dataset.LABS_COMMIT + "/"
            assert url.startswith(prefix), f"unexpected download: {url}"
            dest.write_text(f"content of {url.removeprefix(prefix)}\n")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to go online. Local file: URLs stay allowed."""
    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        """Open local file: URLs only."""
        assert str(url).startswith("file:"), f"tests must not use the network: {url}"
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture
def course_zip(tmp_path):
    """The tiny course zip, as a file."""
    return make_course_zip(tmp_path / "course.zip")


@pytest.fixture
def ocw_items(course_zip):
    """What select_ocw picks out of the tiny course."""
    with zipfile.ZipFile(course_zip) as course:
        return dataset.select_ocw(course)


@pytest.fixture
def fetch(course_zip):
    """A network stand-in that records what was asked for."""
    return FakeFetch(course_zip)


@pytest.fixture
def root(tmp_path):
    """An empty sandbox folder."""
    return tmp_path / "sandbox"


@pytest.fixture
def built(root, fetch):
    """A sandbox in which the dataset was built once."""
    dataset.build_dataset(root, fetch)
    return root


def dump_ids(root: Path) -> list[str]:
    """Ids of every file in the evaluation vault's dump, sorted."""
    dump = root / "eval" / "_dump"
    return sorted(path.relative_to(dump).as_posix() for path in dump.rglob("*") if path.is_file())


def gold_group_sizes(sizes: dict[str | None, int]) -> dict[str, str | None]:
    """Made-up file ids with gold groups: `sizes[group]` files in each group."""
    return {f"{group}/file{n:02}.pdf": group for group, size in sizes.items() for n in range(size)}


# Selection


def test_select_takes_resource_files_and_content_pages_and_nothing_else(ocw_items):
    # Left out: the image, the video that is not in the zip, the orphan file without metadata,
    # the two index pages, the blank page and all site chrome.
    assert [item.name for item in ocw_items] == OCW_NAMES


def test_a_page_is_an_index_page_only_when_resources_are_filed_under_it(tmp_path):
    path = tmp_path / "course.zip"
    with zipfile.ZipFile(path, "w") as course:
        readings = page("readings", "Readings", ["Readings"], "<p>Chapter 1 for lecture 1.</p>")
        exams = page("exams", "Exams", ["Exams"], "<p>Midterm (PDF)</p>")
        midterm = resource(1, "midterm.pdf", ["Exams"], parent=exams["uid"])
        course.writestr("pages/readings/data.json", json.dumps(readings))
        course.writestr("pages/exams/data.json", json.dumps(exams))
        course.writestr("resources/midterm/data.json", json.dumps(midterm))
        course.writestr(f"static_resources/{1:032x}_midterm.pdf", "bytes")
    with zipfile.ZipFile(path) as course:
        items = dataset.select_ocw(course)
    assert [(item.name, item.gold_group) for item in items] == [
        ("midterm.pdf", "exams"),
        ("readings.html", "lecture-notes"),
    ]


def test_selected_files_keep_their_bytes_and_lose_the_hash_prefix(ocw_items):
    by_url = {item.source_url: item for item in ocw_items}
    lec1 = by_url[dataset.OCW_COURSE_URL + "resources/lec1/"]
    assert (lec1.name, lec1.data) == ("lec1.pdf", b"bytes of lec1")
    capitals = by_url[dataset.OCW_COURSE_URL + "resources/hw1-upper/"]
    assert (capitals.name, capitals.data) == ("HW1.pdf", b"bytes of hw1-upper")


def test_a_page_becomes_a_small_html_file_without_site_chrome(ocw_items):
    syllabus = next(item for item in ocw_items if item.name == "syllabus.html")
    text = syllabus.data.decode("utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<title>Syllabus</title>" in text
    assert "<h1>Syllabus</h1>\n<h3>Grading</h3><p>Midterm 15%, final 25%.</p>" in text
    assert "site chrome" not in text


def test_page_html_escapes_the_title_and_keeps_the_content():
    text = dataset.page_html("Q & A", "<p>Ask <b>early</b>.</p>")
    assert "<title>Q &amp; A</title>" in text
    assert "<h1>Q &amp; A</h1>" in text
    assert "<p>Ask <b>early</b>.</p>" in text


# The mapping from source structure to gold group


def test_gold_group_and_source_type_come_from_the_metadata(ocw_items):
    found = {
        item.source_url.removeprefix(dataset.OCW_COURSE_URL): (item.gold_group, item.source_type)
        for item in ocw_items
    }
    assert found == {
        "resources/lec1/": ("lecture-notes", "Lecture Notes"),
        "resources/lagrange/": ("lecture-notes", "Readings"),
        "resources/hw1/": ("homework", "Problem Sets"),
        "resources/hw1-2/": ("homework", "Problem Set Solutions"),
        "resources/hw1-upper/": ("exams", "Exams"),
        "resources/p1/": ("homework", "Problem Sets"),
        "resources/plot/": ("homework", "Tools"),
        "resources/final_f01/": ("exams", "Exams"),
        "resources/final_f01soln/": ("exams", "Exam Solutions"),
        "resources/rec1/": (None, "Recitation Notes"),
        "resources/mixed/": (None, "Exams, Problem Sets"),
        "resources/untyped/": (None, "none"),
        "pages/syllabus/": ("logistics", "page: syllabus"),
        "pages/projects/": ("projects", "Projects"),
        "pages/instructor-insights/": (None, "page: instructor-insights"),
    }
    assert {item.license for item in ocw_items} == {"CC BY-NC-SA 4.0"}


@pytest.mark.parametrize(
    ("types", "group"),
    [
        (["Lecture Notes"], "lecture-notes"),
        (["Readings"], "lecture-notes"),
        (["Problem Sets"], "homework"),
        (["Problem Set Solutions"], "homework"),
        (["Assignments"], "homework"),
        (["Tools"], "homework"),
        (["Exams"], "exams"),
        (["Exam Solutions"], "exams"),
        (["Projects"], "projects"),
        (["Problem Sets", "Problem Set Solutions"], "homework"),  # two types, one group
        (["Lecture Notes", "Lecture Videos"], "lecture-notes"),  # an unknown type is ignored
        (["Exams", "Problem Sets"], None),  # two groups: no single answer
        (["Lecture Videos"], None),
        ([], None),
    ],
)
def test_ocw_types_map_to_groups(types, group):
    assert dataset.ocw_group(types) == group


@pytest.mark.parametrize(
    ("types", "page_name", "group"),
    [
        ([], "syllabus", "logistics"),
        ([], "calendar", "logistics"),
        ([], "instructor-insights", None),
        (["Projects"], "projects", "projects"),
        (["Exams"], "syllabus", "exams"),  # a page that has a type goes by its type
    ],
)
def test_a_page_without_a_type_is_mapped_by_its_name(types, page_name, group):
    assert dataset.ocw_group(types, page=page_name) == group


def test_the_mapping_only_names_evaluation_groups_and_reaches_all_of_them():
    mapped = {
        *dataset.OCW_TYPE_GROUPS.values(),
        *dataset.OCW_PAGE_GROUPS.values(),
        dataset.LABS_GROUP,
    }
    assert mapped == GROUP_IDS


def test_an_unusual_license_is_reported_as_the_source_states_it(tmp_path):
    path = tmp_path / "course.zip"
    with zipfile.ZipFile(path, "w") as course:
        meta = resource(1, "lec1.pdf", ["Lecture Notes"]) | {"license": "https://example.org/x"}
        course.writestr("resources/lec1/data.json", json.dumps(meta))
        course.writestr(f"static_resources/{1:032x}_lec1.pdf", "bytes")
    with zipfile.ZipFile(path) as course:
        assert [item.license for item in dataset.select_ocw(course)] == ["https://example.org/x"]


# Flat naming


def test_strip_hash_drops_only_a_leading_32_character_hash():
    assert dataset.strip_hash("be6c46f4bb8a2e709c04533b7979d51f_hw2_soln.pdf") == "hw2_soln.pdf"
    assert dataset.strip_hash("hw2_soln.pdf") == "hw2_soln.pdf"
    assert dataset.strip_hash("6-867f06_notes.pdf") == "6-867f06_notes.pdf"
    bare = "be6c46f4bb8a2e709c04533b7979d51f_"
    assert dataset.strip_hash(bare) == bare


def test_flat_names_leaves_unique_names_alone():
    assert dataset.flat_names(["hw1.pdf", "lec1.pdf", "plot.m"]) == [
        "hw1.pdf",
        "lec1.pdf",
        "plot.m",
    ]


def test_flat_names_numbers_repeats_in_order_and_ignores_case():
    assert dataset.flat_names(["a.pdf", "A.pdf", "a.pdf"]) == ["a.pdf", "A-2.pdf", "a-3.pdf"]
    assert dataset.flat_names(OCW_NAMES) == OCW_FLAT_NAMES


def test_flat_names_never_reuses_a_name_it_made_up():
    names = ["hw1.pdf", "hw1.pdf", "hw1-2.pdf", "README", "README", "data.tar.gz", "data.tar.gz"]
    assert dataset.flat_names(names) == [
        "hw1.pdf",
        "hw1-2.pdf",
        "hw1-2-2.pdf",
        "README",
        "README-2",
        "data.tar.gz",
        "data.tar-2.gz",
    ]


# The split


def test_split_takes_a_fifth_of_each_gold_group_for_dev():
    gold = gold_group_sizes(
        {"homework": 52, "lecture-notes": 24, "exams": 17, "projects": 1, None: 4}
    )
    splits = dataset.assign_splits(gold)
    assert set(splits) == set(gold)
    assert set(splits.values()) == {"dev", "test"}
    dev = Counter(gold[file_id] for file_id, split in splits.items() if split == "dev")
    # round(0.2 * size) per group: a group of one or two files is all test.
    assert dev == {"homework": 10, "lecture-notes": 5, "exams": 3, None: 1}


def test_split_is_the_same_on_every_run_and_in_any_input_order():
    gold = gold_group_sizes({"homework": 52, "exams": 17})
    first = dataset.assign_splits(gold)
    shuffled = list(gold.items())
    random.Random(1).shuffle(shuffled)
    assert dataset.assign_splits(dict(shuffled)) == first
    assert dataset.assign_splits(gold, seed=6999) == first
    assert dataset.assign_splits(gold, seed=7000) != first


def test_split_is_pinned_to_seed_6999():
    assert (dataset.SEED, dataset.DEV_SHARE) == (6999, 0.2)
    gold = {f"hw{n:02}.pdf": "homework" for n in range(1, 11)}
    splits = dataset.assign_splits(gold)
    assert sorted(file_id for file_id in gold if splits[file_id] == "dev") == [
        "hw04.pdf",
        "hw09.pdf",
    ]


def test_one_group_splits_the_same_whatever_else_is_in_the_dataset():
    alone = dataset.assign_splits(gold_group_sizes({"exams": 17}))
    together = dataset.assign_splits(gold_group_sizes({"exams": 17, "homework": 52, None: 3}))
    assert {file_id: together[file_id] for file_id in alone} == alone


# The build: dump layout, vault, gold.jsonl, DATASET.md


def test_ocw_files_land_flat_with_repeated_names_numbered(built):
    folder = built / "eval" / "_dump" / "6.867"
    assert all(path.is_file() for path in folder.iterdir())
    assert sorted(path.name for path in folder.iterdir()) == sorted(OCW_FLAT_NAMES)
    assert (folder / "hw1.pdf").read_bytes() == b"bytes of hw1"
    assert (folder / "hw1-2.pdf").read_bytes() == b"bytes of hw1-2"
    assert (folder / "HW1-3.pdf").read_bytes() == b"bytes of hw1-upper"


def test_lab_files_keep_their_repository_paths(built):
    folder = built / "eval" / "_dump" / "6.S191"
    found = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert found == sorted(dataset.LAB_FILES)
    solution = "lab1/solutions/PT_Part1_Intro_Solution.ipynb"
    assert (folder / solution).read_text() == f"content of {solution}\n"


def test_the_lab_files_are_the_notebooks_readmes_and_helper_modules():
    allowed = re.compile(
        r"lab[123]/([^/]+\.ipynb|README\.md)"
        r"|lab[123]/solutions/[^/]+\.ipynb"
        r"|mitdeeplearning/[^/]+\.py"
    )
    assert len(dataset.LAB_FILES) == len(set(dataset.LAB_FILES)) == 26
    assert all(allowed.fullmatch(path) for path in dataset.LAB_FILES)


def test_only_the_two_sources_are_fetched_and_only_once(built, fetch):
    lab_urls = [f"{RAW_URL}{dataset.LABS_COMMIT}/{path}" for path in dataset.LAB_FILES]
    assert fetch.urls == [ZIP_URL, *lab_urls]
    dataset.build_dataset(built, fetch)
    assert len(fetch.urls) == 1 + len(dataset.LAB_FILES)


def test_gold_jsonl_has_one_line_per_dump_file_with_the_agreed_fields(built):
    lines = (built / "eval" / "gold.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["file_id"] for row in rows] == dump_ids(built)
    assert len(rows) == len(OCW_FLAT_NAMES) + len(dataset.LAB_FILES)
    fields = ["file_id", "gold_group", "source_url", "source_type", "license", "sha256", "split"]
    for row in rows:
        assert list(row) == fields
        assert row["gold_group"] is None or row["gold_group"] in GROUP_IDS
        assert row["source_url"].startswith("https://")
        assert row["source_type"]
        assert row["license"] in ("CC BY-NC-SA 4.0", "MIT")
        data = (built / "eval" / "_dump" / row["file_id"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["split"] in ("dev", "test")


def test_gold_labels_follow_each_file_through_the_renaming(built):
    gold = {label.file_id: label for label in dataset.read_gold(built / "eval")}
    numbered = gold["6.867/hw1-2.pdf"]
    assert (numbered.gold_group, numbered.source_type) == ("homework", "Problem Set Solutions")
    assert numbered.source_url == dataset.OCW_COURSE_URL + "resources/hw1-2/"
    assert gold["6.867/HW1-3.pdf"].gold_group == "exams"
    assert gold["6.867/syllabus.html"].gold_group == "logistics"
    assert gold["6.867/projects.html"].gold_group == "projects"
    assert gold["6.867/practice.pdf"].gold_group is None


def test_lab_gold_labels_are_homework_with_the_folder_as_source_type(built):
    gold = {label.file_id: label for label in dataset.read_gold(built / "eval")}
    labs = {file_id: label for file_id, label in gold.items() if file_id.startswith("6.S191/")}
    assert len(labs) == 26
    assert {label.gold_group for label in labs.values()} == {"homework"}
    assert {label.license for label in labs.values()} == {"MIT"}
    assert {label.source_type for label in labs.values()} == {
        "lab1",
        "lab1/solutions",
        "lab2",
        "lab2/solutions",
        "lab3",
        "lab3/solutions",
        "mitdeeplearning",
    }
    path = "lab2/solutions/TF_Part1_MNIST_Solution.ipynb"
    assert labs[f"6.S191/{path}"].source_url == f"{BLOB_URL}{dataset.LABS_COMMIT}/{path}"


def test_gold_split_is_stratified_over_the_built_dataset(built):
    labels = dataset.read_gold(built / "eval")
    sizes = Counter(label.gold_group for label in labels)
    dev = Counter(label.gold_group for label in labels if label.split == "dev")
    assert sizes == {
        "homework": 30,
        "exams": 3,
        "lecture-notes": 2,
        "projects": 1,
        "logistics": 1,
        None: 4,
    }
    assert dev == {"homework": 6, "exams": 1, None: 1}


def test_build_returns_what_it_wrote(root, fetch):
    assert dataset.build_dataset(root, fetch) == dataset.read_gold(root / "eval")


def test_build_sets_up_the_vault_with_the_evaluation_groups(built):
    eval_root = built / "eval"
    assert Config.load(eval_root).dump_dir == "_dump"
    assert vault.read_groups(eval_root) == sorted(EVAL_GROUPS, key=lambda group: group.id)
    assert vault.latest_set(eval_root).version == len(EVAL_GROUPS)
    for group in EVAL_GROUPS:
        assert (eval_root / group.name).is_dir()


def test_groups_that_already_exist_are_left_alone(root, fetch):
    eval_root = root / "eval"
    vault.init_vault(eval_root)
    vault.add_group(eval_root, "Exams", "my own wording")
    dataset.build_dataset(root, fetch)
    groups = {group.id: group for group in vault.read_groups(eval_root)}
    assert set(groups) == GROUP_IDS
    assert groups["exams"].covers == "my own wording"


def test_a_rebuild_changes_nothing(built, fetch):
    eval_root = built / "eval"
    lecture = eval_root / "_dump" / "6.867" / "lec1.pdf"
    before = (
        (eval_root / "gold.jsonl").read_bytes(),
        (eval_root / "DATASET.md").read_bytes(),
        vault.latest_set(eval_root).version,
        dump_ids(built),
        lecture.stat().st_mtime_ns,
    )
    dataset.build_dataset(built, fetch)
    assert before == (
        (eval_root / "gold.jsonl").read_bytes(),
        (eval_root / "DATASET.md").read_bytes(),
        vault.latest_set(eval_root).version,
        dump_ids(built),
        lecture.stat().st_mtime_ns,
    )


def test_a_dump_file_that_differs_from_its_source_is_written_again(built, fetch):
    lecture = built / "eval" / "_dump" / "6.867" / "lec1.pdf"
    lecture.write_text("edited by hand")
    dataset.build_dataset(built, fetch)
    assert lecture.read_bytes() == b"bytes of lec1"


def test_a_stray_file_in_the_dump_is_an_error(built, fetch):
    (built / "eval" / "_dump" / "6.867" / "extra.pdf").write_text("not from the course")
    with pytest.raises(dataset.DatasetError, match=re.escape("6.867/extra.pdf")):
        dataset.build_dataset(built, fetch)


def test_hidden_files_in_the_dump_are_ignored(built, fetch):
    (built / "eval" / "_dump" / ".DS_Store").write_text("finder")
    (built / "eval" / "_dump" / "6.867" / ".DS_Store").write_text("finder")
    dataset.build_dataset(built, fetch)
    assert len(dataset.read_gold(built / "eval")) == len(OCW_FLAT_NAMES) + len(dataset.LAB_FILES)


def test_dataset_notes_list_sources_licenses_mapping_and_counts(built, course_zip):
    notes = (built / "eval" / "DATASET.md").read_text()
    expected = [
        ZIP_URL,
        hashlib.sha256(course_zip.read_bytes()).hexdigest(),
        "| 15 in `6.867/` | CC BY-NC-SA 4.0 |",
        "https://github.com/MITDeepLearning/introtodeeplearning at commit",
        dataset.LABS_COMMIT,
        "| 26 in `6.S191/` | MIT |",
        "| OCW 6.867 | learning resource type | Problem Set Solutions | homework | 1 |",
        "| OCW 6.867 | learning resource type | Exams | exams | 2 |",
        "| OCW 6.867 | learning resource type | Assignments | homework | 0 |",
        "| OCW 6.867 | page with no learning resource type | syllabus | logistics | 1 |",
        "| 6.S191 labs | repository folder | every folder taken | homework | 26 |",
        "| Gold group | dev | test | all |",
        "| homework | 6 | 24 | 30 |",
        "| logistics | 0 | 1 | 1 |",
        "| none | 1 | 3 | 4 |",
        "| all | 8 | 33 | 41 |",
        "Files without a gold group: 4.",
        "seed 6999",
        ".ipynb 18",
    ]
    for text in expected:
        assert text in notes, text
    assert chr(8212) not in notes  # 8212 is the em dash, which the project never writes


# The network step and the command line


def test_download_writes_the_file_and_leaves_no_partial_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    dest = tmp_path / "cache" / "copy.bin"
    dataset.download(source.as_uri(), dest)
    assert dest.read_bytes() == b"payload"
    assert [path.name for path in dest.parent.iterdir()] == ["copy.bin"]


def test_a_failed_download_names_the_url_and_writes_no_file(tmp_path):
    dest = tmp_path / "cache" / "copy.bin"
    with pytest.raises(dataset.DatasetError, match=re.escape("missing.bin")):
        dataset.download((tmp_path / "missing.bin").as_uri(), dest)
    assert not dest.exists()


def test_a_download_that_was_cut_short_is_an_error(tmp_path, monkeypatch):
    response = io.BytesIO(b"1234")  # four bytes, where the header below promises ten
    response.headers = {"Content-Length": "10"}
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: response)
    dest = tmp_path / "cache" / "copy.bin"
    with pytest.raises(dataset.DatasetError, match="cut short: 4 of 10 bytes"):
        dataset.download("https://example.invalid/file.bin", dest)
    assert not dest.exists()


def test_main_builds_under_the_given_root_and_prints_the_counts(built, monkeypatch, capsys):
    # Everything is cached by the first build, so the real download is never needed.
    monkeypatch.setattr(sys, "argv", ["dataset", str(built)])
    dataset.main()
    out = capsys.readouterr().out
    assert "41 files" in out
    assert "homework 30" in out
    assert "none 4" in out
    assert "dev 8, test 33" in out


def test_main_reports_a_known_failure_in_one_line(built, monkeypatch):
    (built / "eval" / "_dump" / "stray.txt").write_text("not part of the dataset")
    monkeypatch.setattr(sys, "argv", ["dataset", str(built)])
    with pytest.raises(SystemExit) as failure:
        dataset.main()
    message = str(failure.value)
    assert message.startswith("error: ")
    assert message.endswith("Remove them and run again: stray.txt")
