"""The group set for the public-course evaluation (MIT OCW 6.867 plus the MIT 6.S191 labs)."""

from ..types import Group

EVAL_GROUPS: list[Group] = [
    Group(
        id="lecture-notes",
        name="Lecture notes",
        covers="lecture notes, lecture slides and assigned readings",
        excludes="homework and exams",
    ),
    Group(
        id="homework",
        name="Homework",
        covers="problem sets, lab notebooks, starter and helper code for assignments, and their solutions",
        excludes="exams and course projects",
    ),
    Group(
        id="exams",
        name="Exams",
        covers="quizzes, midterm and final exams, practice exams, and their solutions",
        excludes="homework problem sets",
    ),
    Group(
        id="projects",
        name="Projects",
        covers="course project descriptions, guidelines, proposals and example projects",
        excludes="homework problem sets",
    ),
    Group(
        id="logistics",
        name="Logistics",
        covers="syllabus, calendar, schedule, grading rules and course policies",
    ),
]

# The group held out, then added back, in the "group added mid-run" experiment.
HELD_OUT_GROUP_ID = "exams"
