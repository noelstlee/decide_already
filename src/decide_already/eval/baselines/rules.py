"""File-name rules baseline: keywords and extensions, no content and no model.

The default rules were written blind, from general knowledge of how course files are named,
before looking at any evaluation file. The baseline reads only a file's name and path.
"""

import re
import time
from pathlib import PurePosixPath

from pydantic import Field

from ...backends.base import SortRequest, SortResult
from ...types import StrictModel

MODEL = "rules"


class Rule(StrictModel):
    """One rule: a path with one of these words or extensions sends the file to the group."""

    group_id: str
    # Whole lowercase words: "lab" matches lab2 and Lab_3, but not syllabus.
    keywords: list[str] = Field(default_factory=list)
    # Lowercase, with the dot: ".py".
    extensions: list[str] = Field(default_factory=list)

    def matches(self, words: set[str], extension: str) -> bool:
        """True when the path has one of the keywords as a word, or has one of the extensions."""
        return not words.isdisjoint(self.keywords) or extension in self.extensions


def _rule(group_id: str, keywords: str = "", extensions: str = "") -> Rule:
    """Build a rule from space-separated keywords and extensions (keeps the table below short)."""
    return Rule(group_id=group_id, keywords=keywords.split(), extensions=extensions.split())


# Rules for the five evaluation groups (eval/groups.py), checked top to bottom: first match wins.
# The order goes from the most telling words to the least: words that name a kind of document,
# then weaker hints, then the file extension alone.
DEFAULT_RULES: list[Rule] = [
    # First, so "exam_schedule" and "grading_policy" are logistics whatever else the name says.
    _rule("logistics", "syllabus calendar schedule logistics policy policies grading"),
    # Before exams, so "final_project" is a project and not a final exam.
    _rule("projects", "project projects proj proposal proposals milestone milestones"),
    # Before homework, so "midterm_solutions" is an exam and not a homework solution.
    _rule("exams", "exam exams midterm midterms final finals quiz quizzes"),
    _rule(
        "homework",
        "hw homework homeworks pset psets ps problemset assignment assignments lab labs",
    ),
    _rule("lecture-notes", "lecture lectures lec lect slides slide recitation recitations"),
    # Weaker hints: these words show up in more than one kind of file.
    _rule(
        "homework",
        "solution solutions soln solns sol problem problems exercise exercises"
        " starter skeleton notebook notebooks",
    ),
    _rule("lecture-notes", "notes reading readings handout handouts chapter chapters"),
    _rule("logistics", "info overview welcome faq staff"),
    # No telling word at all: code and notebooks are most often assignment material,
    # and slide decks are lectures.
    _rule("homework", extensions=".py .ipynb .m .r .jl .java .c .cpp .h .js .ts .sh .sql"),
    _rule("lecture-notes", extensions=".ppt .pptx .key"),
]


def path_words(path: str) -> set[str]:
    """The lowercase words in a path, its extension left out.

    Words are runs of letters, so digits, punctuation and camelCase all split them:
    "hw3/ProblemSet2_soln.pdf" gives hw, problem, set and soln.
    """
    suffix = PurePosixPath(path).suffix
    stem = path[: len(path) - len(suffix)] if suffix else path
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return set(re.findall(r"[a-z]+", spaced.lower()))


class RulesBackend:
    """Sorts by file name and path alone. It never reads the content and asks no model."""

    name = "rules"

    def __init__(self, rules: list[Rule] | None = None):
        """Use the given rules, in order, or DEFAULT_RULES when none are given."""
        self.rules = DEFAULT_RULES if rules is None else rules

    def sort(self, request: SortRequest) -> SortResult:
        """Give the whole probability to the group of the first rule that matches.

        Rules for groups that are not in the request are skipped. When no rule matches,
        fits_any is 0.0 and probs is empty: the baseline has no pick, so the file ends up
        Unsorted and counts as wrong on top-1 for any file that has a gold group.
        """
        started = time.perf_counter()
        offered = [group.id for group in request.groups]
        file = request.state.get("file") or {}
        paths = [path for path in (file.get("path"), file.get("name"), request.file_id) if path]
        words = set().union(*(path_words(path) for path in paths))
        extension = PurePosixPath(file.get("name") or request.file_id).suffix.lower()
        match = next(
            (
                rule.group_id
                for rule in self.rules
                if rule.group_id in offered and rule.matches(words, extension)
            ),
            None,
        )
        if match is None:
            probs = {}
        else:
            probs = {group_id: 1.0 if group_id == match else 0.0 for group_id in offered}
        return SortResult(
            file_id=request.file_id,
            probs=probs,
            fits_any=0.0 if match is None else 1.0,
            model=MODEL,
            option_order=offered,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
