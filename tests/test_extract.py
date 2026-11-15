"""Text extraction: what each kind of file gives as type, title, outline and text."""

import json

import pymupdf
import pytest
from pptx import Presentation
from pptx.util import Inches

from decide_already.extract import Extracted, extract


def write(path, content):
    """Write a text fixture and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


def make_pdf(path, pages, title=None, toc=None):
    """Write a PDF with one page per text in `pages`; a page given as None holds only a picture."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text is None:
            picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
            picture.clear_with(180)
            page.insert_image(pymupdf.Rect(72, 72, 300, 300), pixmap=picture)
        else:
            page.insert_text((72, 72), text)
    if title is not None:
        document.set_metadata({"title": title})
    if toc is not None:
        document.set_toc(toc)
    document.save(path)
    document.close()
    return path


def make_deck(path):
    """Write a four-slide deck: a title slide, a bullet slide, an untitled slide, a table slide."""
    deck = Presentation()
    first = deck.slides.add_slide(deck.slide_layouts[0])
    first.shapes.title.text = "Lecture 5: Support Vector Machines"
    first.placeholders[1].text = "CS 4641, Fall 2025"
    bullets = deck.slides.add_slide(deck.slide_layouts[1])
    bullets.shapes.title.text = "Margins"
    body = bullets.placeholders[1].text_frame
    body.text = "Maximize the margin"
    body.paragraphs[0].add_line_break()
    body.paragraphs[0].add_run().text = "between the classes"
    bullets.notes_slide.notes_text_frame.text = "speaker notes stay out"
    untitled = deck.slides.add_slide(deck.slide_layouts[6])
    box = untitled.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    box.text_frame.text = "A slide without a title"
    last = deck.slides.add_slide(deck.slide_layouts[5])
    last.shapes.title.text = "Kernels"
    table = last.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    for row, cells in enumerate([("Kernel", "Formula"), ("RBF", "exp(-g d^2)")]):
        for column, text in enumerate(cells):
            table.cell(row, column).text = text
    group = last.shapes.add_group_shape()
    inner = group.shapes.add_textbox(Inches(1), Inches(4), Inches(2), Inches(1))
    inner.text_frame.text = "inside a group"
    deck.save(str(path))
    return path


def make_notebook(path, cells):
    """Write a notebook from (cell type, source) pairs. Code cells get an output to be dropped."""
    built = []
    for kind, source in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["SECRET OUT"]}]
            cell["execution_count"] = 1
        built.append(cell)
    notebook = {"cells": built, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    return write(path, json.dumps(notebook))


# Markdown


def test_markdown_title_outline_and_text(tmp_path):
    source = (
        "# Homework 3: Kernel Methods\n\nDue Friday.\n\n## Problem 1\n\n"
        "```python\n# a comment, not a heading\nprint('hi')\n```\n\n"
        "### Part (a) ###\n\n#hashtag is not a heading\n"
    )
    result = extract(write(tmp_path / "hw3.md", source))
    assert result.file_type == "Markdown"
    assert result.title == "Homework 3: Kernel Methods"
    assert result.outline == ["Homework 3: Kernel Methods", "Problem 1", "Part (a)"]
    assert result.text == source.strip()


def test_markdown_without_headings_takes_its_first_line_as_title(tmp_path):
    path = write(tmp_path / "notes.markdown", "---\n\nGrading policy for the course\n\nMore.\n")
    result = extract(path)
    assert result.file_type == "Markdown"
    assert result.title == "Grading policy for the course"
    assert result.outline == []


def test_markdown_headings_inside_a_longer_fence_are_skipped(tmp_path):
    source = "````\n```\n# still code\n```\n# still code too\n````\n# Real heading\n"
    assert extract(write(tmp_path / "a.md", source)).outline == ["Real heading"]


def test_markdown_title_skips_leading_license_and_copyright_headings(tmp_path):
    source = (
        "# License\n\nMIT License.\n\n## COPYRIGHT NOTICE\n\nThe course staff, 2025.\n\n"
        "# Project guidelines\n\nTeams of two.\n\n## Licensed datasets\n\nAsk a TA.\n"
    )
    result = extract(write(tmp_path / "project.md", source))
    assert result.title == "Project guidelines"
    assert result.outline == [  # the outline stays complete
        "License",
        "COPYRIGHT NOTICE",
        "Project guidelines",
        "Licensed datasets",
    ]


@pytest.mark.parametrize(
    ("source", "title"),
    [
        ("# License\n\nPermission is hereby granted.\n", "License"),
        (
            "## copyright 2025, the staff\n\n# LICENSE\n\nAll rights reserved.\n",
            "copyright 2025, the staff",
        ),
    ],
)
def test_markdown_with_only_license_headings_takes_the_first_one_as_title(tmp_path, source, title):
    assert extract(write(tmp_path / "LICENSE.md", source)).title == title


@pytest.mark.parametrize(
    ("source", "title"),
    [
        ("# Homework 3\n\n## License\n\nMIT.\n", "Homework 3"),
        ("# About the license\n\n## Terms\n", "About the license"),
        ("# Licensing\n\n## Terms\n", "Licensing"),
    ],
)
def test_markdown_title_is_the_first_heading_unless_it_starts_as_boilerplate(
    tmp_path, source, title
):
    assert extract(write(tmp_path / "notes.md", source)).title == title


# Text


def test_text_title_is_its_first_line_and_it_has_no_outline(tmp_path):
    path = write(tmp_path / "hw2errata.txt", "\n\nProblem #1\nThe sign in part (b) is wrong.\n")
    assert extract(path) == Extracted(
        file_type="Text", title="Problem #1", text="Problem #1\nThe sign in part (b) is wrong."
    )


@pytest.mark.parametrize("name", ["notes.rst", "grades.csv", "README", "data.unknown"])
def test_any_other_readable_file_is_text(tmp_path, name):
    result = extract(write(tmp_path / name, "=====\nOffice hours\n=====\nMonday at 3\n"))
    assert result.file_type == "Text"
    assert result.title == "Office hours"
    assert result.outline == []
    assert result.text.endswith("Monday at 3")


def test_title_skips_lines_of_stray_symbols(tmp_path):
    path = write(tmp_path / "lec.txt", "X\n1\n}|\n6.867 Machine learning, lecture 7\nTopics\n")
    assert extract(path).title == "6.867 Machine learning, lecture 7"


# Code

PYTHON = '''"""Homework 3: Kernel Methods.

Starter code for the class.
"""
import numpy as np


def rbf_kernel(x, y):
    return np.exp(-np.sum((x - y) ** 2))


async def fetch():
    pass


class Trainer:
    """Fits a model."""

    def fit(self, data):
        def helper():
            pass

    def predict(self, x):
        return 1
'''


def test_python_title_outline_and_text(tmp_path):
    result = extract(write(tmp_path / "hw3_solution.py", PYTHON))
    assert result.file_type == "Python code"
    assert result.title == "Homework 3: Kernel Methods."
    assert result.outline == [
        "rbf_kernel()",
        "fetch()",
        "class Trainer",
        "Trainer.fit()",
        "Trainer.predict()",
    ]
    assert result.text == PYTHON.strip()


def test_python_without_docstring_takes_its_first_comment_as_title(tmp_path):
    source = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n#\n# Perceptron demo\nx = 1\n"
    assert extract(write(tmp_path / "demo.py", source)).title == "Perceptron demo"


def test_a_comment_far_from_the_top_is_not_a_title(tmp_path):
    source = "import os\n" * 12 + "\n\ndef run():\n    # x = (x - x.min()) / x.max()\n    pass\n"
    result = extract(write(tmp_path / "util.py", source))
    assert result.title is None
    assert result.outline == ["run()"]


def test_python_that_does_not_parse_still_gives_names(tmp_path):
    source = (
        "# Old perceptron script\nprint 'hello'\n\ndef train(X, y):\n    print 'training'\n\n"
        "class Model:\n    def predict(self, x):\n        return 1\n"
    )
    result = extract(write(tmp_path / "old.py", source))
    assert result.title == "Old perceptron script"
    assert result.outline == ["train()", "class Model", "predict()"]
    assert result.text == source.strip()


# A lab cell with a blank left for the student: it does not parse, and its docstring is prose.
UNFINISHED = '''def compute_loss(labels, logits):
  """TODO: call the loss
  function and the hyperparameters created above.
  def is short for define, and this is one
  class of losses."""
  loss = # TODO
  return loss

class Model(Base):
  async def call(self, x):
    return x
'''


def test_prose_in_python_that_does_not_parse_is_not_taken_for_a_name(tmp_path):
    names = ["compute_loss()", "class Model", "call()"]  # no "and()", "is()" or "class of"
    assert extract(write(tmp_path / "todo.py", UNFINISHED)).outline == names
    notebook = make_notebook(tmp_path / "todo.ipynb", [("code", UNFINISHED)])
    assert extract(notebook).outline == names


# Other languages: file name, type label, title, outline, source.
OTHER_CODE = [
    (
        "perceptron_train.m",
        "MATLAB code",
        "[theta, k] = perceptron_train(X,y)",
        ["perceptron_train()"],
        """\
function [theta, k] = perceptron_train(X,y)
% [theta, k] = perceptron_train(X,y)
%   train a Perceptron classifier on given data
num_correct = 0;
""",
    ),
    (
        "stats.R",
        "R code",
        "Regression helpers",
        ["fit_model()", "rmse()"],
        """\
# Regression helpers
fit_model <- function(x, y) {
  lm(y ~ x)
}
rmse = function(a, b) sqrt(mean((a - b)^2))
scale <- 2
""",
    ),
    (
        "model.jl",
        "Julia code",
        "Kernel regression",
        ["struct Model", "fit!()"],
        """\
# Kernel regression
mutable struct Model
    w::Vector{Float64}
end

function fit!(model, X, y)
end
""",
    ),
    (
        "timer.js",
        "JavaScript code",
        "Quiz timer widget",
        ["startTimer()", "stopTimer()", "class Timer"],
        """\
// Quiz timer widget
export function startTimer(seconds) {
}
const stopTimer = () => {
};
class Timer extends Widget {
}
const total = (a + b) * 2;
""",
    ),
    (
        "rubric.ts",
        "TypeScript code",
        "Shared types for the grader",
        ["interface Rubric", "score()"],
        """\
/*
 * Shared types for the grader
 */
export interface Rubric {
}
export const score = async (work: string): Promise<number> => {
  return 1;
};
""",
    ),
    (
        "Main.java",
        "Java code",
        "Entry point for the search homework.",
        ["class Main", "main()", "load()"],
        """\
/** Entry point for the search homework. */
public class Main {
    public static void main(String[] args) {
        if (args.length > 0) {
            run(args[0]);
        }
    }

    private static List<String> load(String path) throws IOException {
        return null;
    }
}
""",
    ),
    (
        "sort.c",
        "C code",
        "Sorting routines for lab 2",
        ["struct node", "compare()", "main()"],
        """\
// Sorting routines for lab 2
#include <stdio.h>

struct node {
    int key;
};

int compare(const void *a, const void *b);

static int compare(const void *a, const void *b)
{
    return 0;
}

int main(int argc, char **argv) {
    int n = compare(0, 0);
    printf("%d", n);
    return compare(0, 0);
}
""",
    ),
    (
        "vec.cpp",
        "C++ code",
        "A growable vector",
        ["Vec::push()"],
        """\
// A growable vector
void Vec::push(int x) {
    data.push_back(x);
}
""",
    ),
    (
        "list.h",
        "C header",
        "Linked list for lab 1",
        ["struct list"],
        """\
/* Linked list for lab 1 */
typedef struct list {
    int head;
} list;
""",
    ),
    (
        "run.sh",
        "Shell script",
        "Runs every experiment",
        ["setup()", "cleanup()"],
        """\
#!/bin/bash
# Runs every experiment
setup() {
  echo hi
}
function cleanup {
  echo bye
}
setup
""",
    ),
    (
        "hw3.tex",
        "LaTeX source",
        "Homework 3 source",
        ["Kernel methods", "Problem 1: the $\\mathcal{L}$ loss"],
        """\
% Homework 3 source
\\documentclass{article}
\\begin{document}
\\section{Kernel methods}
\\subsection*{Problem 1: the $\\mathcal{L}$ loss}
\\end{document}
""",
    ),
    (
        "schema.sql",
        "SQL code",
        "Gradebook schema",
        ["TABLE students", "view top_students"],
        """\
-- Gradebook schema
CREATE TABLE students (
  id INTEGER
);
create view if not exists top_students as select 1;
""",
    ),
]


@pytest.mark.parametrize(
    ("name", "file_type", "title", "outline", "source"),
    OTHER_CODE,
    ids=[case[0] for case in OTHER_CODE],
)
def test_other_languages(tmp_path, name, file_type, title, outline, source):
    result = extract(write(tmp_path / name, source))
    assert result.file_type == file_type
    assert result.title == title
    assert result.outline == outline
    assert result.text == source.strip()


# Notebooks


def test_notebook_title_outline_and_text(tmp_path):
    magics = "%matplotlib inline\n!pip install torch\n"  # with these, the cell is not Python
    cells = [
        ("markdown", ["# Lab 1: Intro to PyTorch\n", "Welcome to the lab.\n"]),
        ("code", "import torch\n\ndef build_model():\n    return None\n"),
        ("markdown", "## 1.1 Tensors"),
        ("code", magics + "class Trainer:\n    def fit(self):\n        pass"),
    ]
    result = extract(make_notebook(tmp_path / "lab1.ipynb", cells))
    assert result.file_type == "Jupyter notebook, 4 cells"
    assert result.title == "Lab 1: Intro to PyTorch"
    assert result.outline == [
        "Lab 1: Intro to PyTorch",
        "1.1 Tensors",
        "build_model()",
        "class Trainer",
        "fit()",
    ]
    assert result.text.startswith("# Lab 1: Intro to PyTorch\nWelcome to the lab.\n\nimport torch")
    assert result.text.endswith("def fit(self):\n        pass")
    assert "SECRET OUT" not in result.text


def test_notebook_without_a_markdown_heading_has_no_title(tmp_path):
    result = extract(make_notebook(tmp_path / "one.ipynb", [("code", "def run():\n    pass")]))
    assert result.file_type == "Jupyter notebook, 1 cell"
    assert result.title is None
    assert result.outline == ["run()"]


def test_notebook_title_skips_a_leading_copyright_heading(tmp_path):
    cells = [  # how the MIT 6.S191 labs open: a links table and a license cell come first
        ("markdown", '<table align="center">\n  <td>Run in Google Colab</td>\n</table>\n'),
        ("markdown", ["# Copyright Information\n"]),
        ("code", "# Copyright 2026 MIT Introduction to Deep Learning. All Rights Reserved.\n"),
        (
            "markdown",
            "# Lab 1: Intro to PyTorch and Music Generation with RNNs\n\n# Part 1: Intro\n",
        ),
        ("code", "def func(x):\n    return x\n"),
    ]
    result = extract(make_notebook(tmp_path / "PT_Part1_Intro.ipynb", cells))
    assert result.title == "Lab 1: Intro to PyTorch and Music Generation with RNNs"
    assert result.outline == [  # the outline stays complete
        "Copyright Information",
        "Lab 1: Intro to PyTorch and Music Generation with RNNs",
        "Part 1: Intro",
        "func()",
    ]


def test_notebook_with_only_a_copyright_heading_takes_it_as_title(tmp_path):
    cells = [("markdown", "## Copyright Information"), ("code", "def run():\n    pass")]
    result = extract(make_notebook(tmp_path / "bare.ipynb", cells))
    assert result.title == "Copyright Information"
    assert result.outline == ["Copyright Information", "run()"]


def test_half_a_surrogate_pair_in_a_notebook_is_replaced(tmp_path):
    content = (  # a broken emoji escape: valid JSON, but a string that UTF-8 cannot hold
        '{"cells": [{"cell_type": "markdown", "metadata": {}, '
        '"source": ["# Lab \\ud83d one\\n", "A broken \\udc00 emoji."]}], '
        '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    result = extract(write(tmp_path / "emoji.ipynb", content))
    assert result.title == "Lab \ufffd one"
    assert result.outline == ["Lab \ufffd one"]
    assert result.text == "# Lab \ufffd one\nA broken \ufffd emoji."


@pytest.mark.parametrize(
    "content",
    ["", "not json at all", "[1, 2, 3]", '{"cells": null}', '{"cells": [3, "x"], "nbformat": 4}'],
)
def test_damaged_notebook_gives_no_text(tmp_path, content):
    result = extract(write(tmp_path / "bad.ipynb", content))
    assert result.file_type.startswith("Jupyter notebook")
    assert result.text == ""


# PDFs


def test_pdf_title_outline_and_text(tmp_path):
    path = make_pdf(
        tmp_path / "notes.pdf",
        ["Lecture 1\nLinear classifiers", "Lecture 1, continued\nThe perceptron"],
        title="Kernel Methods: Lecture Notes",
        toc=[[1, "Introduction", 1], [2, "Linear classifiers", 1], [1, "The perceptron", 2]],
    )
    result = extract(path)
    assert result.file_type == "PDF, 2 pages"
    assert result.title == "Kernel Methods: Lecture Notes"
    assert result.outline == ["Introduction", "Linear classifiers", "The perceptron"]
    assert "Linear classifiers" in result.text
    assert result.text.index("Lecture 1\n") < result.text.index("The perceptron")


@pytest.mark.parametrize("stated", [None, "hw3.dvi", "Microsoft Word - hw3.doc", "Untitled"])
def test_pdf_without_a_sensible_title_takes_its_first_line(tmp_path, stated):
    path = make_pdf(tmp_path / "hw3.pdf", ["X\n1\nProblem Set 3\nDue Monday"], title=stated)
    result = extract(path)
    assert result.file_type == "PDF, 1 page"
    assert result.title == "Problem Set 3"
    assert result.outline == []


def test_pdf_with_no_text_layer_gives_no_text(tmp_path):
    result = extract(make_pdf(tmp_path / "scan.pdf", [None, None]))
    assert result == Extracted(file_type="PDF, 2 pages")


@pytest.mark.parametrize("content", [b"", b"%PDF-1.7\nnot really a pdf\n", b"plain text"])
def test_damaged_pdf_gives_no_text(tmp_path, content):
    path = tmp_path / "bad.pdf"
    path.write_bytes(content)
    assert extract(path) == Extracted(file_type="PDF")


def test_locked_pdf_gives_no_text(tmp_path):
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Secret text")
    path = tmp_path / "locked.pdf"
    document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    document.close()
    assert extract(path) == Extracted(file_type="PDF")


# Slides


def test_slides_title_outline_and_text(tmp_path):
    result = extract(make_deck(tmp_path / "lecture5.pptx"))
    assert result.file_type == "Slides, 4"
    assert result.title == "Lecture 5: Support Vector Machines"
    assert result.outline == ["Lecture 5: Support Vector Machines", "Margins", "Kernels"]
    assert result.text == (
        "Lecture 5: Support Vector Machines\nCS 4641, Fall 2025\n\n"
        "Margins\nMaximize the margin\nbetween the classes\n\n"
        "A slide without a title\n\n"
        "Kernels\nKernel Formula\nRBF exp(-g d^2)\ninside a group"
    )


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04 not a real deck", b"plain text"])
def test_damaged_deck_gives_no_text(tmp_path, content):
    path = tmp_path / "bad.pptx"
    path.write_bytes(content)
    assert extract(path) == Extracted(file_type="Slides")


def test_deck_without_slides_gives_no_text(tmp_path):
    Presentation().save(str(tmp_path / "empty.pptx"))
    assert extract(tmp_path / "empty.pptx") == Extracted(file_type="Slides, 0")


# HTML

OCW_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>Syllabus | Machine Learning | MIT OpenCourseWare</title>
  <style>body { color: red; }</style>
  <script>var menu = "<main>not content</main>";</script>
</head>
<body>
  <h3>Browse Course Material</h3>
  <nav class="course-nav"><ul><li><a href="/">Calendar</a></li><li>Exams</li></ul></nav>
  <div class="panel"><h3>Course Info</h3><p>Instructors and departments</p></div>
  <header><h2>Syllabus</h2></header>
  <article class="content">
    <main id="course-content-section">
      <h3>Course Meeting
          Times</h3>
      <p>Lectures: 2 sessions / week,
         1.5 hours / session</p>
      <nav>On this page</nav>
      <h3>Grading</h3>
      <table>
        <tr><th>ACTIVITIES</th><th>PERCENTAGES</th></tr>
        <tr><td>Midterm</td><td>15%</td></tr>
      </table>
      <script>track("syllabus");</script>
      <p>Copying will <b>not</b> be tolerated.<br>Ask a TA &amp; read the policy.</p>
    </main>
  </article>
  <footer>Give now, about OCW</footer>
</body>
</html>
"""


def test_html_keeps_only_the_main_content(tmp_path):
    result = extract(write(tmp_path / "syllabus.html", OCW_PAGE))
    assert result.file_type == "HTML page"
    assert result.title == "Syllabus | Machine Learning | MIT OpenCourseWare"
    assert result.outline == ["Course Meeting Times", "Grading"]
    assert result.text == (
        "Course Meeting Times\n"
        "Lectures: 2 sessions / week, 1.5 hours / session\n"
        "Grading\n"
        "ACTIVITIES PERCENTAGES\n"
        "Midterm 15%\n"
        "Copying will not be tolerated.\n"
        "Ask a TA & read the policy."
    )


def test_html_article_is_used_when_there_is_no_main(tmp_path):
    source = (
        "<html><body><div>Site menu</div><article><h1>Project guidelines</h1>"
        "<p>Teams of two.</p></article><div>Footer links</div></body></html>"
    )
    result = extract(write(tmp_path / "project.htm", source))
    assert result.title == "Project guidelines"
    assert result.outline == ["Project guidelines"]
    assert result.text == "Project guidelines\nTeams of two."


def test_html_without_main_or_article_keeps_the_page_but_not_its_menus(tmp_path):
    source = (
        "<html><head><title>Calendar</title></head><body><nav>Home | Syllabus</nav>"
        "<h1>Calendar</h1><h2>Week 1</h2><h4>Too deep for the outline</h4>"
        "<pre>line one\nline two</pre><noscript>Turn on scripts</noscript></body></html>"
    )
    result = extract(write(tmp_path / "calendar.html", source))
    assert result.title == "Calendar"
    assert result.outline == ["Calendar", "Week 1"]
    assert result.text == "Calendar\nWeek 1\nToo deep for the outline\nline one\nline two"


def test_html_title_falls_back_to_an_h1_outside_the_main_content(tmp_path):
    source = (
        "<html><body><header><h1><img src='logo.png'></h1><h1>Machine Learning</h1></header>"
        "<main><h2>Week 1</h2><p>Perceptrons.</p></main></body></html>"
    )
    result = extract(write(tmp_path / "week1.html", source))
    assert result.title == "Machine Learning"  # no <title>, and the first <h1> holds only a logo
    assert result.outline == ["Week 1"]
    assert result.text == "Week 1\nPerceptrons."


def test_html_title_prefers_the_h1_inside_the_main_content(tmp_path):
    source = "<header><h1>Course site</h1></header><main><h1>Syllabus</h1><p>Grading.</p></main>"
    result = extract(write(tmp_path / "syllabus.html", source))
    assert result.title == "Syllabus"
    assert result.outline == ["Syllabus"]


def test_html_with_broken_markup_does_not_raise(tmp_path):
    source = "<p>before</p><![FOO[ stuff ]]><div class='x"
    assert "before" in extract(write(tmp_path / "odd.html", source)).text


# Binary files, odd bytes, and the limits on title and outline


@pytest.mark.parametrize("name", ["photo.png", "archive.zip", "weights", "notes.txt", "hw.md"])
def test_binary_file_gives_no_text(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR some words here" * 20)
    result = extract(path)
    assert result.text == ""
    assert result.title is None
    assert result.outline == []


def test_unknown_binary_file_is_labelled_as_binary(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(bytes(range(256)) * 4)
    assert extract(path) == Extracted(file_type="Binary file")


@pytest.mark.parametrize("name", ["a.md", "a.txt", "a.py", "a.html", "a.ipynb", "a.tex", "a"])
def test_empty_file_gives_no_text(tmp_path, name):
    result = extract(write(tmp_path / name, " \n\n"))
    assert result.text == ""
    assert result.title is None
    assert result.outline == []


@pytest.mark.parametrize(
    "name", ["gone.md", "gone.txt", "gone.py", "gone.ipynb", "gone.pdf", "gone.pptx", "gone.html"]
)
def test_file_that_cannot_be_opened_gives_no_text(tmp_path, name):
    result = extract(tmp_path / name)  # deleted between the listing and the read, say
    assert result.text == ""
    assert result.title is None
    assert result.outline == []


def test_pictures_written_out_as_base64_are_dropped_from_the_text(tmp_path):
    picture = "data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+/9=" * 400
    note = f"# Lab notes\n\nBefore. ![plot.png]({picture}) After.\n"
    result = extract(write(tmp_path / "notes.md", note))
    assert result.text == "# Lab notes\n\nBefore. ![plot.png]() After."
    cells = [("markdown", f"# Lab 2\n![plot.png]({picture})\nWords after the picture.")]
    result = extract(make_notebook(tmp_path / "lab2.ipynb", cells))
    assert result.text == "# Lab 2\n![plot.png]()\nWords after the picture."
    drawing = "data:image/svg+xml;charset=utf-8;base64,PHN2Zz48L3N2Zz4="
    result = extract(write(tmp_path / "drawing.txt", f"See <{drawing}> for the data: 3 points."))
    assert result.text == "See <> for the data: 3 points."


def test_bytes_that_are_not_utf8_are_replaced_not_fatal(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("Résumé of the course\n".encode("latin-1"))
    result = extract(path)
    assert result.title == result.text == "R\ufffdsum\ufffd of the course"


def test_byte_order_mark_and_windows_line_endings_are_handled(tmp_path):
    path = tmp_path / "win.md"
    path.write_bytes(b"\xef\xbb\xbf# Syllabus\r\n\r\nWeek one\r\n")
    result = extract(path)
    assert result.title == "Syllabus"
    assert result.text == "# Syllabus\n\nWeek one"


def test_control_characters_are_dropped_from_the_text(tmp_path):
    path = write(tmp_path / "math.txt", "\x1a\nKernel\x01 methods\x0cnext page\ttabbed\n")
    result = extract(path)
    assert result.title == "Kernel methods"
    assert result.text == "Kernel methods\nnext page\ttabbed"


def test_outline_is_capped_at_40_entries_of_120_characters(tmp_path):
    headings = [f"## Section {number} " + "long " * 40 for number in range(60)]
    result = extract(write(tmp_path / "long.md", "\n\n".join(headings)))
    assert len(result.outline) == 40
    assert all(len(entry) <= 120 for entry in result.outline)
    assert result.outline[0].startswith("Section 0 long long")
    assert result.outline[0].endswith("...")


def test_title_is_capped_at_200_characters(tmp_path):
    result = extract(write(tmp_path / "wide.txt", "word " * 500))
    assert len(result.title) <= 200
    assert len(result.text) == len("word " * 500) - 1
