"""The public evaluation dataset and its gold labels (build step 6).

Run as `python -m decide_already.eval.dataset [root]`. The root defaults to `sandbox`, which is
gitignored. Running it again is safe: downloads are cached and unchanged files are left alone.

    <root>/datasets/    the downloaded sources (the cache)
    <root>/eval/        the evaluation vault: _dump/, the group notes, gold.jsonl, DATASET.md

Two sources fill the dump:

- MIT OCW 6.867 Machine Learning, Fall 2006 (CC BY-NC-SA 4.0), from the course's zip download.
  Its files land flat in _dump/6.867/, so a folder never gives the group away.
- The MIT 6.S191 labs (MIT license): notebooks, solutions and helper modules, fetched one by one
  from GitHub at a pinned commit. They keep the repository's own paths under _dump/6.S191/.

Gold labels come only from the sources' own structure (OCW's metadata, the repository's folders),
never from a model and never from reading a file's content. They are written before any sort.
"""

import hashlib
import html
import json
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from .. import vault
from ..types import StrictModel
from .groups import EVAL_GROUPS

SEED = 6999
DEV_SHARE = 0.2
GOLD_FILE = "gold.jsonl"
NOTES_FILE = "DATASET.md"

OCW_COURSE_URL = "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/"
OCW_ZIP_NAME = "6.867-fall-2006.zip"
OCW_FOLDER = "6.867"
OCW_CITATION = (
    "Tommi Jaakkola. 6.867 Machine Learning. Fall 2006. Massachusetts Institute of Technology: "
    "MIT OpenCourseWare, https://ocw.mit.edu. License: Creative Commons BY-NC-SA."
)
CC_BY_NC_SA_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"

# OCW learning resource type to group id. A type that is not listed gives no gold group.
OCW_TYPE_GROUPS = {
    "Lecture Notes": "lecture-notes",
    "Readings": "lecture-notes",
    "Problem Sets": "homework",
    "Problem Set Solutions": "homework",
    "Assignments": "homework",
    "Tools": "homework",
    "Exams": "exams",
    "Exam Solutions": "exams",
    "Projects": "projects",
}
# OCW pages that carry no learning resource type, by page name.
OCW_PAGE_GROUPS = {"syllabus": "logistics", "calendar": "logistics"}

LABS_REPO = "MITDeepLearning/introtodeeplearning"
LABS_COMMIT = "e44754eac7a61d489fb07a68245bfbb290823228"  # head of master on 2026-11-15
LABS_RAW_URL = f"https://raw.githubusercontent.com/{LABS_REPO}/{LABS_COMMIT}/"
LABS_CREDIT = "© MIT Introduction to Deep Learning, http://introtodeeplearning.com"
LABS_FOLDER = "6.S191"
LABS_LICENSE = "MIT"
LABS_GROUP = "homework"  # every folder taken from the repository holds assignment material
LAB_FILES = (
    "lab1/PT_Part1_Intro.ipynb",
    "lab1/PT_Part2_Music_Generation.ipynb",
    "lab1/README.md",
    "lab1/TF_Part1_Intro.ipynb",
    "lab1/TF_Part2_Music_Generation.ipynb",
    "lab1/solutions/PT_Part1_Intro_Solution.ipynb",
    "lab1/solutions/PT_Part2_Music_Generation_Solution.ipynb",
    "lab1/solutions/TF_Part1_Intro_Solution.ipynb",
    "lab1/solutions/TF_Part2_Music_Generation_Solution.ipynb",
    "lab2/PT_Part1_MNIST.ipynb",
    "lab2/PT_Part2_Debiasing.ipynb",
    "lab2/TF_Part1_MNIST.ipynb",
    "lab2/TF_Part2_Debiasing.ipynb",
    "lab2/solutions/PT_Part1_MNIST_Solution.ipynb",
    "lab2/solutions/PT_Part2_Debiasing_Solution.ipynb",
    "lab2/solutions/TF_Part1_MNIST_Solution.ipynb",
    "lab2/solutions/TF_Part2_Debiasing_Solution.ipynb",
    "lab3/LLM_Finetuning.ipynb",
    "lab3/README.md",
    "lab3/solutions/LLM_Finetuning_Solution.ipynb",
    "mitdeeplearning/__init__.py",
    "mitdeeplearning/lab1.py",
    "mitdeeplearning/lab2.py",
    "mitdeeplearning/lab3.py",
    "mitdeeplearning/lab3_old.py",
    "mitdeeplearning/util.py",
)

_HASH_PREFIX = re.compile(r"^[0-9a-f]{32}_(?=.)")
_RESOURCE_META = re.compile(r"resources/([^/]+)/data\.json")
_PAGE_META = re.compile(r"pages/([^/]+)/data\.json")


class DatasetError(Exception):
    """The dataset could not be built: a download failed, or the dump holds a file it should not."""


class Item(StrictModel):
    """One file chosen for the dataset, before it is written into the dump."""

    name: str  # OCW: the original file name. Labs: the path inside the repository.
    data: bytes
    gold_group: str | None
    source_url: str
    source_type: str
    license: str


class GoldLabel(StrictModel):
    """One line of gold.jsonl: a dump file and the group its source says it belongs in."""

    file_id: str  # path relative to the dump folder, with forward slashes
    gold_group: str | None  # a group id from eval/groups.py; None when the file fits no group
    source_url: str
    source_type: str  # the OCW type or the repository folder the label came from
    license: str
    sha256: str
    split: Literal["dev", "test"]


def download(url: str, dest: Path) -> None:
    """Fetch one URL into a file. This is the only function here that touches the network.

    The file gets its final name only once it has arrived whole, so the cache never holds half
    a download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
            expected = response.headers.get("Content-Length")
    except OSError as error:
        raise DatasetError(f"could not download {url}: {error}") from error
    # Python's HTTP client stays silent when a connection drops early, so compare the sizes.
    size = partial.stat().st_size
    if expected is not None and size != int(expected):
        raise DatasetError(f"download of {url} was cut short: {size} of {expected} bytes")
    partial.replace(dest)


def fetch_sources(datasets: Path, fetch: Callable[[str, Path], None]) -> tuple[Path, Path]:
    """Make sure the course zip and the lab files are in the cache, fetching only what is missing.

    Returns the path of the zip and the folder that holds the lab files.
    """
    course_zip = datasets / OCW_ZIP_NAME
    if not course_zip.exists():
        fetch(OCW_COURSE_URL + OCW_ZIP_NAME, course_zip)
    # The commit is part of the folder name, so a new pin never reuses old files.
    labs = datasets / f"introtodeeplearning-{LABS_COMMIT[:7]}"
    for path in LAB_FILES:
        if not (labs / path).exists():
            fetch(LABS_RAW_URL + path, labs / path)
    return course_zip, labs


def strip_hash(name: str) -> str:
    """Drop the 32-character hash OCW puts before a file name: "<hash>_hw1.pdf" gives "hw1.pdf"."""
    return _HASH_PREFIX.sub("", name)


def flat_names(names: list[str]) -> list[str]:
    """Make file names unique for one flat folder: the first keeps its name, repeats get a number.

    ["a.pdf", "A.pdf", "a.pdf"] gives ["a.pdf", "A-2.pdf", "a-3.pdf"]. Names are compared
    without case, because the default macOS file system does not tell "a.pdf" from "A.pdf".
    The result depends on the order of the input, so callers pass names in a fixed order.
    """
    used: set[str] = set()
    unique: list[str] = []
    for name in names:
        candidate, number = name, 1
        while candidate.lower() in used:
            number += 1
            candidate = f"{PurePosixPath(name).stem}-{number}{PurePosixPath(name).suffix}"
        used.add(candidate.lower())
        unique.append(candidate)
    return unique


def ocw_group(types: list[str], page: str | None = None) -> str | None:
    """Map one OCW item to a group id, using only its learning resource types.

    Types that point at two different groups give None, and so do types that point at no group.
    A page with no type at all (the syllabus has none) is looked up by its name instead.
    """
    if not types and page is not None:
        return OCW_PAGE_GROUPS.get(page)
    groups = {OCW_TYPE_GROUPS[t] for t in types if t in OCW_TYPE_GROUPS}
    return groups.pop() if len(groups) == 1 else None


def assign_splits(
    gold: dict[str, str | None], seed: int = SEED, dev_share: float = DEV_SHARE
) -> dict[str, str]:
    """Split file ids into dev and test, stratified by gold group.

    In each group the ids are sorted, shuffled with the seed, and the first
    round(dev_share * size) become dev. A group shuffles on its own, so the result depends only
    on the ids, their groups and the seed, never on the order they arrive in.
    """
    by_group: dict[str | None, list[str]] = {}
    for file_id, group in gold.items():
        by_group.setdefault(group, []).append(file_id)
    splits: dict[str, str] = {}
    for ids in by_group.values():
        ids.sort()
        random.Random(seed).shuffle(ids)
        dev_count = round(len(ids) * dev_share)
        for position, file_id in enumerate(ids):
            splits[file_id] = "dev" if position < dev_count else "test"
    return splits


def page_html(title: str, content: str) -> str:
    """Wrap a page's own content in a small HTML document, without the site's navigation."""
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        f'<meta charset="utf-8">\n<title>{html.escape(title)}</title>\n'
        f"</head>\n<body>\n<h1>{html.escape(title)}</h1>\n{content.strip()}\n</body>\n</html>\n"
    )


def _ocw_license(meta: dict) -> str:
    """The license an OCW item's metadata names, in short form when it is the usual one."""
    url = meta.get("license") or "unknown"
    return "CC BY-NC-SA 4.0" if url == CC_BY_NC_SA_URL else url


def select_ocw(course: zipfile.ZipFile) -> list[Item]:
    """Pick the course's content out of an OCW course zip: resource files, then content pages.

    The zip holds one resources/<name>/data.json per resource, with its learning resource types
    and the stored file's path. The file itself sits in static_resources/<hash>_<file name>.
    Images, and resources whose file is not in the zip, are left out.

    Each page has a pages/<name>/data.json with its title and its HTML in "content". A page
    that resources are filed under (their parent_uid points at it) only lists them: that is an
    index page and is left out. The other pages are written as <name>.html.

    Everything else in the zip is site chrome: scripts, styles, fonts and index.html shells.
    Items come back in a fixed order: resources by name, then pages by name.
    """
    members = set(course.namelist())
    resources: dict[str, dict] = {}
    pages: dict[str, dict] = {}
    for member in members:
        if match := _RESOURCE_META.fullmatch(member):
            resources[match.group(1)] = json.loads(course.read(member))
        elif match := _PAGE_META.fullmatch(member):
            pages[match.group(1)] = json.loads(course.read(member))

    items: list[Item] = []
    for name, meta in sorted(resources.items()):
        if not meta.get("file") or meta.get("resourcetype") == "Image":
            continue
        stored = "static_resources/" + PurePosixPath(meta["file"]).name
        if stored not in members:
            continue
        types = meta.get("learning_resource_types") or []
        items.append(
            Item(
                name=strip_hash(PurePosixPath(stored).name),
                data=course.read(stored),
                gold_group=ocw_group(types),
                source_url=f"{OCW_COURSE_URL}resources/{name}/",
                source_type=", ".join(types) or "none",
                license=_ocw_license(meta),
            )
        )

    parents = {meta["parent_uid"] for meta in resources.values() if meta.get("parent_uid")}
    for name, meta in sorted(pages.items()):
        content = (meta.get("content") or "").strip()
        if not content or meta.get("uid") in parents:
            continue
        types = meta.get("learning_resource_types") or []
        items.append(
            Item(
                name=f"{name}.html",
                data=page_html(meta.get("title") or name, content).encode("utf-8"),
                gold_group=ocw_group(types, page=name),
                source_url=f"{OCW_COURSE_URL}pages/{name}/",
                source_type=", ".join(types) or f"page: {name}",
                license=_ocw_license(meta),
            )
        )
    return items


def lab_items(labs: Path) -> list[Item]:
    """Turn the cached lab files into items. The repository folder a file sits in is its type."""
    return [
        Item(
            name=path,
            data=(labs / path).read_bytes(),
            gold_group=LABS_GROUP,
            source_url=f"https://github.com/{LABS_REPO}/blob/{LABS_COMMIT}/{path}",
            source_type=PurePosixPath(path).parent.as_posix(),
            license=LABS_LICENSE,
        )
        for path in LAB_FILES
    ]


def collect_items(course_zip: Path, labs: Path) -> dict[str, Item]:
    """Every dataset file by its file id: OCW files flat in 6.867/, lab files by path in 6.S191/."""
    with zipfile.ZipFile(course_zip) as course:
        ocw = select_ocw(course)
    names = flat_names([item.name for item in ocw])
    items = {f"{OCW_FOLDER}/{name}": item for name, item in zip(names, ocw, strict=True)}
    items.update({f"{LABS_FOLDER}/{item.name}": item for item in lab_items(labs)})
    return items


def gold_labels(items: dict[str, Item]) -> list[GoldLabel]:
    """One gold label per item, sorted by file id, with the file's hash and its split."""
    splits = assign_splits({file_id: item.gold_group for file_id, item in items.items()})
    return [
        GoldLabel(
            file_id=file_id,
            gold_group=item.gold_group,
            source_url=item.source_url,
            source_type=item.source_type,
            license=item.license,
            sha256=hashlib.sha256(item.data).hexdigest(),
            split=splits[file_id],
        )
        for file_id, item in sorted(items.items())
    ]


def read_gold(eval_root: Path) -> list[GoldLabel]:
    """Read and validate an evaluation vault's gold.jsonl."""
    lines = (eval_root / GOLD_FILE).read_text(encoding="utf-8").splitlines()
    return [GoldLabel.model_validate_json(line) for line in lines if line.strip()]


def write_dump(dump: Path, items: dict[str, Item]) -> None:
    """Write the items into the dump folder. A file that is already there and equal is left alone.

    Any other file in the dump would be sorted without a gold label, so it is an error.
    Hidden files and folders are skipped, as the sorter skips them.
    """
    for file_id, item in items.items():
        dest = dump / file_id
        if not dest.exists() or dest.read_bytes() != item.data:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(item.data)
    strangers = []
    for path in sorted(dump.rglob("*")):
        relative = path.relative_to(dump)
        hidden = any(part.startswith(".") for part in relative.parts)
        if path.is_file() and not hidden and relative.as_posix() not in items:
            strangers.append(relative.as_posix())
    if strangers:
        raise DatasetError(
            f"{dump} holds files that are not part of the dataset, so they would have no gold "
            f"label. Remove them and run again: {', '.join(strangers)}"
        )


def _table(header: list[str], rows: list[list[object]]) -> str:
    """Render a markdown table."""
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(str(cell) for cell in line) + " |" for line in lines)


def dataset_notes(labels: list[GoldLabel], zip_sha256: str) -> str:
    """The text of DATASET.md: sources, licenses, the mapping behind the labels, and the counts."""
    ocw = [label for label in labels if label.file_id.startswith(f"{OCW_FOLDER}/")]
    labs = [label for label in labels if label.file_id.startswith(f"{LABS_FOLDER}/")]
    by_type = Counter(label.source_type for label in ocw)
    by_group = Counter(label.gold_group for label in labels)
    by_cell = Counter((label.gold_group, label.split) for label in labels)
    by_split = Counter(label.split for label in labels)
    by_kind = Counter(PurePosixPath(label.file_id).suffix or "no extension" for label in labels)

    sources = _table(
        ["Source", "Files", "License", "From"],
        [
            [
                "MIT OpenCourseWare 6.867 Machine Learning, Fall 2006",
                f"{len(ocw)} in `{OCW_FOLDER}/`",
                ", ".join(sorted({label.license for label in ocw})),
                f"{OCW_COURSE_URL}{OCW_ZIP_NAME} (sha256 `{zip_sha256}`)",
            ],
            [
                "MIT 6.S191 Introduction to Deep Learning, the labs",
                f"{len(labs)} in `{LABS_FOLDER}/`",
                ", ".join(sorted({label.license for label in labs})),
                f"https://github.com/{LABS_REPO} at commit `{LABS_COMMIT}`",
            ],
        ],
    )
    type_rows = [
        ["OCW 6.867", "learning resource type", name, group, by_type[name]]
        for name, group in OCW_TYPE_GROUPS.items()
    ]
    page_rows = [
        ["OCW 6.867", "page with no learning resource type", name, group, by_type[f"page: {name}"]]
        for name, group in OCW_PAGE_GROUPS.items()
    ]
    labs_row = ["6.S191 labs", "repository folder", "every folder taken", LABS_GROUP, len(labs)]
    mapping = _table(
        ["Source", "The label comes from", "Value", "Gold group", "Files"],
        [*type_rows, *page_rows, labs_row],
    )
    groups: list[str | None] = [group.id for group in EVAL_GROUPS]
    if by_group[None]:
        groups.append(None)
    count_rows = [
        [group or "none", by_cell[(group, "dev")], by_cell[(group, "test")], by_group[group]]
        for group in groups
    ]
    counts = _table(
        ["Gold group", "dev", "test", "all"],
        [*count_rows, ["all", by_split["dev"], by_split["test"], len(labels)]],
    )
    kinds = ", ".join(f"{kind} {count}" for kind, count in sorted(by_kind.items()))
    return f"""# Evaluation dataset

Written by `python -m decide_already.eval.dataset`. A rebuild overwrites this file.

{len(labels)} files sit in the dump folder, and each has one line in `{GOLD_FILE}`.

## Sources

{sources}

- OCW attribution: {OCW_CITATION}
- Labs attribution: {LABS_CREDIT}
- From the course zip: every resource file except images, flat and under its original name.
  The hash OCW puts before a stored name is dropped, and a repeated name gets a number.
  Archives inside the course (zip files of data and code) stay single files.
- Course pages: a page's own title and content, taken from its metadata, are written as
  `<page>.html` without the site's navigation. A page that resources are filed under only
  lists them, so it counts as an index page and is left out.
- From the labs repository: the notebooks and READMEs in lab1, lab2 and lab3, the notebooks in
  their solutions folders, and the Python modules in mitdeeplearning. Paths are kept.

## Gold labels

Labels come from the sources' own structure. No model was asked and no file's content was read.

{mapping}

A file that matches no row, or rows that name two different groups, has no gold group
(`null` in `{GOLD_FILE}`). Files without a gold group: {by_group[None]}.

## Counts

{counts}

The split is stratified by gold group. In each group the file ids are sorted and shuffled
with seed {SEED}, and the first {DEV_SHARE:.0%} (rounded) are dev. The rest are test.

File kinds: {kinds}.
"""


def build_dataset(root: Path, fetch: Callable[[str, Path], None] = download) -> list[GoldLabel]:
    """Build the evaluation vault under `root`: cached sources, groups, dump files, gold labels.

    `fetch(url, dest)` is the network step. Tests pass their own, so they never go online.
    """
    course_zip, labs = fetch_sources(root / "datasets", fetch)
    items = collect_items(course_zip, labs)

    eval_root = root / "eval"
    config = vault.init_vault(eval_root)
    existing = {group.id for group in vault.read_groups(eval_root)}
    for group in EVAL_GROUPS:
        if group.id not in existing:
            vault.add_group(eval_root, group.name, group.covers, group.excludes)
    write_dump(eval_root / config.dump_dir, items)

    labels = gold_labels(items)
    gold_lines = "".join(json.dumps(label.model_dump()) + "\n" for label in labels)
    (eval_root / GOLD_FILE).write_text(gold_lines, encoding="utf-8")
    zip_sha256 = hashlib.sha256(course_zip.read_bytes()).hexdigest()
    (eval_root / NOTES_FILE).write_text(dataset_notes(labels, zip_sha256), encoding="utf-8")
    return labels


def main() -> None:
    """Build the dataset under the folder named on the command line (default: sandbox)."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sandbox")
    try:
        labels = build_dataset(root)
    except DatasetError as error:
        raise SystemExit(f"error: {error}") from None
    groups = Counter(label.gold_group or "none" for label in labels)
    splits = Counter(label.split for label in labels)
    print(f"{len(labels)} files in {root / 'eval'}, gold labels in {GOLD_FILE}")
    print("by group: " + ", ".join(f"{group} {count}" for group, count in sorted(groups.items())))
    print(f"by split: dev {splits['dev']}, test {splits['test']}")


if __name__ == "__main__":
    main()
