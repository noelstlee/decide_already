"""Write the ten invented course files for the Phase 0 smoke test (build step 0).

Run from the repo root:

    .venv/bin/python scripts/make_smoke_vault.py [--out sandbox/smoke] [--force]

The files go into <out>/_dump/. They describe an imaginary course, "CS 4990 Applied Machine
Learning", at an unnamed university: nothing is copied from a real course, and no real person or
institution is named. Every run writes the same bytes (no randomness, no clock, fixed file
metadata). The script writes the files and nothing else: no vault state and no groups.

The code comes first. The text of the ten files is below it, and SMOKE_FILES near the end maps
each file to the function that writes it and to its content.
"""

import argparse
import io
import json
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path

import pymupdf
from pptx import Presentation

from decide_already.config import DEFAULT_DUMP_DIR

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "sandbox" / "smoke"  # sandbox/ is gitignored
AUTHOR = "CS 4990 course staff"  # goes into PDF and slide metadata; a role, not a person
STAMP = datetime(2025, 8, 18, 9, 0, 0)  # the fixed "saved at" time for all file metadata

PAGE_WIDTH, PAGE_HEIGHT, MARGIN = 612, 792, 64  # US Letter, in points
LINE_HEIGHT = 1.35  # distance between two lines of text, as a multiple of the font size
# How each kind of PDF paragraph is drawn: (built-in PyMuPDF font, size, space above, left indent).
PDF_STYLES = {
    "title": ("hebo", 17, 0, 0),
    "heading": ("hebo", 12, 12, 0),
    "body": ("helv", 10, 6, 0),
    "item": ("helv", 10, 3, 18),
}
# The first word of a paragraph in a PDF source picks its style. Anything else is body text.
PDF_MARKERS = {"#": "title", "##": "heading", "-": "item", "---": "break"}


def write_text(path: Path, text: str) -> None:
    """Write a text file as UTF-8 with Unix line endings, whatever the platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Break text into lines that fit a width in points, measured in a built-in PDF font."""
    lines = [""]
    for word in text.split():
        longer = f"{lines[-1]} {word}".strip()
        if lines[-1] and pymupdf.get_text_length(longer, fontname=font, fontsize=size) > width:
            lines.append(word)
        else:
            lines[-1] = longer
    return lines


def write_pdf(path: Path, source: str, title: str = "", bookmarks: bool = False) -> None:
    """Typeset a PDF source onto Letter pages and save it, the same bytes on every run.

    The source is plain text with a blank line between paragraphs. A paragraph that starts with
    "# " is the title, "## " a heading and "- " an indented item; "---" alone starts a new page.
    `title` is stored as the document's title metadata. With `bookmarks`, every heading also
    becomes an entry in the PDF's table of contents.
    """
    doc = pymupdf.open()
    page, y = None, PAGE_HEIGHT  # y is how far down the page the text has reached; full at first
    toc = []
    for paragraph in source.strip().split("\n\n"):
        text = " ".join(paragraph.split())
        marker, _, rest = text.partition(" ")
        style = PDF_MARKERS.get(marker, "body")
        if style == "break":
            y = PAGE_HEIGHT  # nothing fits below this, so the next paragraph opens a page
            continue
        if style != "body":
            text = rest
        font, size, space_above, indent = PDF_STYLES[style]
        lines = wrap(text, font, size, PAGE_WIDTH - 2 * MARGIN - indent)
        if y + space_above + len(lines) * size * LINE_HEIGHT > PAGE_HEIGHT - MARGIN:
            page, y = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT), MARGIN
        else:
            y += space_above
        if style == "heading" and bookmarks:
            toc.append([1, text, doc.page_count])
        first_line = (MARGIN + indent, y + size * LINE_HEIGHT)
        page.insert_text(first_line, lines, fontname=font, fontsize=size, lineheight=LINE_HEIGHT)
        y += len(lines) * size * LINE_HEIGHT
    if toc:
        doc.set_toc(toc)
    saved_at = f"D:{STAMP:%Y%m%d%H%M%S}Z"
    doc.set_metadata(
        {"title": title, "author": AUTHOR, "creationDate": saved_at, "modDate": saved_at}
    )
    doc.save(path, deflate=True, no_new_id=True)  # PyMuPDF would otherwise add a random file id
    doc.close()


def write_notebook(path: Path, cells: list[tuple[str, str]]) -> None:
    """Write (cell type, source) pairs as a Jupyter notebook (nbformat 4.5) with no outputs.

    Cell ids are numbered instead of random, so the file is the same on every run.
    """
    notebook_cells = []
    for number, (cell_type, source) in enumerate(cells, start=1):
        cell = {
            "cell_type": cell_type,
            "id": f"cell-{number:02d}",
            "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        notebook_cells.append(cell)
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_text(path, json.dumps(notebook, indent=1) + "\n")


def write_deck(path: Path, slides: list[tuple[str, list[str]]]) -> None:
    """Write (title, bullets) pairs as a .pptx slide deck, the same bytes on every run."""
    deck = Presentation()
    for title, bullets in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[1])  # the "Title and Content" layout
        slide.shapes.title.text = title
        slide.placeholders[1].text_frame.text = "\n".join(bullets)  # one paragraph per bullet
    # Replace the template's own properties, which name the library's author and dates in 2013.
    properties = deck.core_properties
    properties.title = slides[0][0]
    properties.author = properties.last_modified_by = AUTHOR
    properties.created = properties.modified = STAMP
    saved = io.BytesIO()
    deck.save(saved)
    # python-pptx stamps every zip entry with the current time, so copy them with a fixed time.
    with zipfile.ZipFile(saved) as source, zipfile.ZipFile(path, "w") as target:
        for entry in source.infolist():
            fixed = zipfile.ZipInfo(entry.filename, date_time=STAMP.timetuple()[:6])
            fixed.compress_type = zipfile.ZIP_DEFLATED
            fixed.external_attr = entry.external_attr
            target.writestr(fixed, source.read(entry))


def readable_chars(path: Path) -> int:
    """Count the text characters that can be read back from a written file.

    PDFs are reopened with PyMuPDF, the deck with python-pptx and the notebook as JSON. A count
    above zero proves the file holds real text that an extractor can get at.
    """
    if path.suffix == ".pdf":
        with pymupdf.open(path) as doc:
            return sum(len(page.get_text()) for page in doc)
    if path.suffix == ".pptx":
        shapes = [shape for slide in Presentation(str(path)).slides for shape in slide.shapes]
        return sum(len(shape.text_frame.text) for shape in shapes if shape.has_text_frame)
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        return sum(len("".join(cell["source"])) for cell in cells)
    return len(path.read_text(encoding="utf-8"))


def main() -> None:
    """Write the ten files, then print what can be read back from each and the suggested groups."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"vault folder; the files go into its {DEFAULT_DUMP_DIR}/ (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"write even if {DEFAULT_DUMP_DIR}/ is not empty: the ten files are replaced, "
        "anything else in there is left alone",
    )
    args = parser.parse_args()

    dump = args.out.resolve() / DEFAULT_DUMP_DIR
    if dump.is_dir() and any(dump.iterdir()) and not args.force:
        raise SystemExit(f"error: {dump} is not empty; pass --force to write there anyway")

    targets = {relative: dump / relative for relative in SMOKE_FILES}
    # A symlink inside the dump folder could lead somewhere else. Check before writing anything.
    for target in targets.values():
        if not target.resolve().is_relative_to(dump):
            raise SystemExit(f"error: {target} leads outside {dump}; nothing was written")
    for relative, (write, content) in SMOKE_FILES.items():
        targets[relative].parent.mkdir(parents=True, exist_ok=True)
        write(targets[relative], content)

    print(f"Wrote {len(targets)} files to {dump}")
    print(f"{'file':<38}{'bytes':>8}{'text chars':>12}")
    no_text = []
    for relative, target in targets.items():
        chars = readable_chars(target)
        print(f"{relative:<38}{target.stat().st_size:>8}{chars:>12}")
        if chars == 0:
            no_text.append(relative)
    if no_text:
        raise SystemExit(f"error: no text could be read back from {', '.join(no_text)}")
    print()
    print(SUGGESTED_GROUPS, end="")


# ---------------------------------------------------------------------------------------------
# The text of the ten files. All of it is invented.
# ---------------------------------------------------------------------------------------------

# 1. hw3/hw3_spec.pdf: the homework handout. Part A fills page one and Part B page two.
HW3_SPEC = """\
# Homework 3: Kernel Methods

CS 4990 Applied Machine Learning, Fall 2025

Released Wednesday, September 24. Due Friday, October 10 at 11:59 pm on the course submission
site. 100 points: 45 for the written questions in Part A and 55 for the programming in Part B.

## Instructions

Submit two files: hw3_written.pdf with your answers to Part A and your results for Question 8,
and your completed hw3_starter.py. You may discuss the problems with classmates, but write your
answers and your code on your own, and list the people you worked with at the top of your PDF.
The late policy in the syllabus applies: three free late days across the term, then 10 points
off per day.

## Part A: Written questions (45 points)

Question 1 (10 points). Feature maps. Let x and z be vectors in R^2, and let
k(x, z) = (x . z + 1)^2.

- (a) (6 points) Write down a feature map phi such that k(x, z) = phi(x) . phi(z). How many
  features does it have?

- (b) (4 points) For inputs in R^d, how many features does the same kernel correspond to? Compare
  the cost of computing k(x, z) directly with the cost of computing phi(x) . phi(z).

Question 2 (10 points). Building kernels. Suppose k1 and k2 are valid kernels on the same inputs.
For each function below, prove that it is a valid kernel or give a counterexample.

- (a) (3 points) k(x, z) = k1(x, z) + k2(x, z)

- (b) (3 points) k(x, z) = c * k1(x, z) for a constant c > 0

- (c) (4 points) k(x, z) = k1(x, z) - k2(x, z)

Question 3 (12 points). The RBF kernel is k(x, z) = exp(-gamma * ||x - z||^2) with gamma > 0.

- (a) (3 points) Show that k(x, x) = 1 for every x, and that 0 < k(x, z) <= 1 for every x and z.

- (b) (4 points) Describe the Gram matrix of n distinct training points when gamma is very large,
  and when gamma is close to zero.

- (c) (5 points) Use your answer to (b) to explain why a very large gamma overfits and a very
  small gamma underfits.

Question 4 (13 points). The soft-margin SVM.

- (a) (4 points) Write the primal problem of the soft-margin SVM. Say in one sentence each what
  the slack variables and the constant C do.

- (b) (4 points) In the dual solution, where does a training point sit relative to the margin
  when alpha_i = 0, when 0 < alpha_i < C, and when alpha_i = C?

- (c) (5 points) A classmate trains an RBF SVM and reports 100% training accuracy and 61%
  validation accuracy. Name two changes to try first, and explain why each should help.

---

## Part B: Programming (55 points)

Complete the functions marked TODO in hw3_starter.py. Do not change function names or arguments:
the autograder imports your file and calls them. Use only NumPy in Questions 5 and 7. In
Question 6 you may call the SVM solver in scikit-learn, but only with a precomputed kernel.

Question 5 (15 points). Implement rbf_kernel(X, Z, gamma). It returns the matrix K with
K[i, j] = exp(-gamma * ||X[i] - Z[j]||^2). The implementation must be vectorized, with no Python
loops over rows. The autograder compares your output with a reference on random inputs
(tolerance 1e-8) and checks that two inputs of 2,000 rows each finish in under 2 seconds.

Question 6 (20 points). Implement train_svm(X, y, C, gamma). Train a soft-margin kernel SVM on
labels in {-1, +1}. Build the Gram matrix with your own rbf_kernel and pass it to the solver.
Return an SVMModel with the support vectors, their dual coefficients, the intercept and gamma.

Question 7 (10 points). Implement predict(model, X). Return a vector of -1 and +1 labels from the
decision function f(x) = sum_i alpha_i y_i k(x_i, x) + b, where the sum runs over the support
vectors.

Question 8 (10 points). Experiments. The starter code generates the two-moons data: 400 training
points and 200 validation points.

- (a) (5 points) Run a grid search over C in {0.1, 1, 10, 100} and gamma in {0.01, 0.1, 1, 10}.
  Report the validation accuracy of all 16 settings in a table.

- (b) (5 points) Plot the decision boundary for your best setting and for C = 100, gamma = 10.
  In two or three sentences, explain the difference between the two plots.

## What to submit

- hw3_written.pdf: your answers to Questions 1 to 4, and the table, plots and explanation for
  Question 8.

- hw3_starter.py: your completed code. Do not rename the file.

## Grading

Questions 5 to 7 are graded by the autograder. You can submit as many times as you like before
the deadline, and only your last submission counts. Written answers are graded on correctness
and on how clearly you explain your reasoning. Regrade requests are open for one week after
grades are released.

## Getting help

Start early: Question 6 takes longer than it looks. Lecture 9 (kernels) and Lecture 10 (support
vector machines) cover everything you need. Ask questions on the course forum or at office hours,
and do not post your code where other students can see it.
"""

# 2. hw3/hw3_starter.py: starter code with stubs and TODO comments.
HW3_STARTER = '''\
"""Homework 3 starter code: kernel methods.

CS 4990 Applied Machine Learning, Fall 2025.

Fill in every TODO below, then run `python hw3_starter.py` to try your code on the two-moons
data. Do not change function names or arguments: the autograder imports this file and calls
rbf_kernel, train_svm and predict. The questions, point values and due date are in hw3_spec.pdf.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SVMModel:
    """What train_svm returns and predict reads."""

    support_vectors: np.ndarray  # shape (m, d): the training points with alpha > 0
    dual_coef: np.ndarray  # shape (m,): alpha_i * y_i for each support vector
    intercept: float  # the bias term b
    gamma: float  # the RBF width used in training


def make_moons(n: int, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate the two-moons data: n points in 2-D with labels in {-1, +1}. Provided for you."""
    rng = np.random.default_rng(seed)
    half = n // 2
    angles = rng.uniform(0.0, np.pi, size=half)
    upper = np.column_stack([np.cos(angles), np.sin(angles)])
    lower = np.column_stack([1.0 - np.cos(angles), 0.5 - np.sin(angles)])
    X = np.vstack([upper, lower]) + rng.normal(scale=noise, size=(2 * half, 2))
    y = np.concatenate([np.ones(half), -np.ones(half)])
    return X, y


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    """Question 5 (15 points). Return K with K[i, j] = exp(-gamma * ||X[i] - Z[j]||^2).

    X has shape (n, d) and Z has shape (m, d), so K has shape (n, m).
    Your code must be vectorized: no Python loops over rows.
    """
    # TODO: compute every squared distance at once.
    # Hint: ||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2, and X @ Z.T holds every x.z.
    # TODO: rounding can make a distance slightly negative. Clip at zero before the exp.
    raise NotImplementedError("Question 5: implement rbf_kernel")


def train_svm(X: np.ndarray, y: np.ndarray, C: float, gamma: float) -> SVMModel:
    """Question 6 (20 points). Train a soft-margin kernel SVM on labels in {-1, +1}.

    Build the Gram matrix with your own rbf_kernel and pass it to a solver that accepts a
    precomputed kernel. Do not use the solver's built-in RBF kernel.
    """
    # TODO: build the n by n Gram matrix with rbf_kernel(X, X, gamma).
    # TODO: fit sklearn.svm.SVC(C=C, kernel="precomputed") on the Gram matrix and y.
    # TODO: read the support vector indices, the dual coefficients and the intercept from the
    #       fitted solver, and return them in an SVMModel together with gamma.
    raise NotImplementedError("Question 6: implement train_svm")


def predict(model: SVMModel, X: np.ndarray) -> np.ndarray:
    """Question 7 (10 points). Return a vector of -1 and +1 labels, one per row of X.

    The decision function is
    f(x) = sum_i dual_coef[i] * k(support_vectors[i], x) + intercept.
    """
    # TODO: compute the kernel between the rows of X and the support vectors.
    # TODO: evaluate the decision function and take its sign. Label a score of exactly 0 as +1.
    raise NotImplementedError("Question 7: implement predict")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of predictions that match the labels. Provided for you."""
    return float(np.mean(y_true == y_pred))


def main() -> None:
    """Question 8 (10 points). Grid search over C and gamma on the two-moons data."""
    X_train, y_train = make_moons(400, seed=0)
    X_val, y_val = make_moons(200, seed=1)
    # TODO: for each C in [0.1, 1, 10, 100] and each gamma in [0.01, 0.1, 1, 10], train on the
    #       training set and print the validation accuracy. Report all 16 numbers as a table.
    # TODO: plot the decision boundary of your best setting and of C=100, gamma=10.
    model = train_svm(X_train, y_train, C=1.0, gamma=1.0)
    print(f"validation accuracy: {accuracy(y_val, predict(model, X_val)):.3f}")


if __name__ == "__main__":
    main()
'''

# 3. hw3/hw3_solution.ipynb: the staff solution, as (cell type, source) pairs.
HW3_SOLUTION = [
    (
        "markdown",
        """\
# Homework 3 Solutions: Kernel Methods

CS 4990 Applied Machine Learning, Fall 2025

**Staff solution. Do not post or share.** This notebook has reference implementations for
Questions 5 to 7, the experiments for Question 8, and answer sketches with the grading rubric
for the written Questions 1 to 4.
""",
    ),
    (
        "code",
        """\
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
""",
    ),
    (
        "markdown",
        """\
## Question 5: rbf_kernel (15 points)

The trick is to expand the squared distance so that everything is a matrix product:
`||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2`. Rounding can make a distance slightly negative, so we
clip at zero before the exponential.

Rubric: 10 points for matching the reference, 5 points for passing the speed test. A solution
with a double loop over rows gets the 10 correctness points only.
""",
    ),
    (
        "code",
        '''\
def rbf_kernel(X, Z, gamma):
    """Return K with K[i, j] = exp(-gamma * ||X[i] - Z[j]||^2), without loops."""
    sq_x = np.sum(X**2, axis=1)[:, None]  # shape (n, 1)
    sq_z = np.sum(Z**2, axis=1)[None, :]  # shape (1, m)
    sq_dist = sq_x - 2.0 * (X @ Z.T) + sq_z  # shape (n, m)
    sq_dist = np.maximum(sq_dist, 0.0)  # rounding can give tiny negatives
    return np.exp(-gamma * sq_dist)
''',
    ),
    (
        "code",
        """\
# Check against the slow definition on random inputs.
rng = np.random.default_rng(0)
A = rng.normal(size=(5, 3))
B = rng.normal(size=(4, 3))
slow = np.array([[np.exp(-0.7 * np.sum((a - b) ** 2)) for b in B] for a in A])
assert np.allclose(rbf_kernel(A, B, 0.7), slow, atol=1e-8)
assert np.allclose(np.diag(rbf_kernel(A, A, 0.7)), 1.0)
""",
    ),
    (
        "markdown",
        """\
## Question 6: train_svm (20 points)

We build the Gram matrix ourselves and give it to the solver with `kernel="precomputed"`. The
solver returns the indices of the support vectors and their dual coefficients `alpha_i * y_i`,
which is all that `predict` needs.

Rubric: 8 points for building the Gram matrix with the student's own `rbf_kernel`, 8 points for
returning the right support vectors and coefficients, 4 points for the intercept. Using the
solver's built-in RBF kernel loses the first 8 points.
""",
    ),
    (
        "code",
        '''\
@dataclass
class SVMModel:
    """What train_svm returns and predict reads."""

    support_vectors: np.ndarray  # shape (m, d)
    dual_coef: np.ndarray  # shape (m,): alpha_i * y_i
    intercept: float
    gamma: float


def train_svm(X, y, C, gamma):
    """Fit a soft-margin SVM on our own Gram matrix and keep only what predict needs."""
    K = rbf_kernel(X, X, gamma)
    solver = SVC(C=C, kernel="precomputed")
    solver.fit(K, y)
    return SVMModel(
        support_vectors=X[solver.support_],
        dual_coef=solver.dual_coef_.ravel(),
        intercept=float(solver.intercept_[0]),
        gamma=gamma,
    )
''',
    ),
    (
        "markdown",
        """\
## Question 7: predict (10 points)

The decision function is `f(x) = sum_i alpha_i y_i k(x_i, x) + b`, summed over the support
vectors. A score of exactly zero is labeled +1.

Rubric: 6 points for the decision function, 4 points for returning labels in {-1, +1} with the
right shape.
""",
    ),
    (
        "code",
        '''\
def predict(model, X):
    """Label each row of X with -1 or +1 using the support vectors."""
    K = rbf_kernel(X, model.support_vectors, model.gamma)  # shape (n, m)
    scores = K @ model.dual_coef + model.intercept
    return np.where(scores >= 0, 1.0, -1.0)


def accuracy(y_true, y_pred):
    """Share of predictions that match the labels."""
    return float(np.mean(y_true == y_pred))
''',
    ),
    (
        "markdown",
        """\
## Question 8: experiments on the two-moons data (10 points)

Part (a) is a 4 by 4 grid over `C` and `gamma`. Part (b) compares the decision boundary of the
best setting with an overfit one.
""",
    ),
    (
        "code",
        '''\
def make_moons(n, noise=0.2, seed=0):
    """The same generator as in hw3_starter.py."""
    rng = np.random.default_rng(seed)
    half = n // 2
    angles = rng.uniform(0.0, np.pi, size=half)
    upper = np.column_stack([np.cos(angles), np.sin(angles)])
    lower = np.column_stack([1.0 - np.cos(angles), 0.5 - np.sin(angles)])
    X = np.vstack([upper, lower]) + rng.normal(scale=noise, size=(2 * half, 2))
    y = np.concatenate([np.ones(half), -np.ones(half)])
    return X, y


X_train, y_train = make_moons(400, seed=0)
X_val, y_val = make_moons(200, seed=1)
''',
    ),
    (
        "code",
        """\
Cs = [0.1, 1, 10, 100]
gammas = [0.01, 0.1, 1, 10]
val_acc = np.zeros((len(Cs), len(gammas)))
for i, C in enumerate(Cs):
    for j, gamma in enumerate(gammas):
        model = train_svm(X_train, y_train, C, gamma)
        val_acc[i, j] = accuracy(y_val, predict(model, X_val))

print("validation accuracy (rows: C, columns: gamma)")
print(np.round(val_acc, 3))
best_i, best_j = np.unravel_index(np.argmax(val_acc), val_acc.shape)
best_C, best_gamma = Cs[best_i], gammas[best_j]
print(f"best: C={best_C}, gamma={best_gamma}, accuracy={val_acc[best_i, best_j]:.3f}")
""",
    ),
    (
        "code",
        '''\
def plot_boundary(model, X, y, title):
    """Shade the two predicted regions and draw the training points on top."""
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    ys = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    labels = predict(model, grid).reshape(grid_x.shape)
    plt.contourf(grid_x, grid_y, labels, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12)
    plt.title(title)
    plt.show()


plot_boundary(train_svm(X_train, y_train, best_C, best_gamma), X_train, y_train, "best setting")
plot_boundary(train_svm(X_train, y_train, 100, 10), X_train, y_train, "C=100, gamma=10")
''',
    ),
    (
        "markdown",
        """\
### What a full-credit answer to Question 8 says

- (a) 3 points for a complete 4 by 4 table, 2 points for naming the best setting. The exact
  numbers depend on the solver version, so grade the pattern and not the digits: the
  `gamma = 0.01` column is the weakest, because the boundary there is close to a straight line,
  and the best cells have a `gamma` of 1 or 10.
- (b) 2 points for the two plots, 3 points for the explanation. The best setting gives a smooth
  boundary that follows the two moons. With `C = 100` and `gamma = 10` the boundary bends around
  individual noisy points: training accuracy goes up and validation accuracy does not.
""",
    ),
    (
        "markdown",
        """\
## Written questions: answer sketches and rubric

**Question 1 (10 points).** (a) Expand `(x . z + 1)^2 = (x1 z1 + x2 z2 + 1)^2`. One feature map
is `phi(x) = (x1^2, x2^2, sqrt(2) x1 x2, sqrt(2) x1, sqrt(2) x2, 1)`, which has 6 features. Give
4 points for a correct map and 2 for the count. (b) For inputs in `R^d` the map has
`(d + 1)(d + 2) / 2` features. The kernel costs `O(d)` to evaluate and the explicit inner
product costs `O(d^2)`. 2 points each.

**Question 2 (10 points).** (a) Valid. For any vector `v`,
`v^T (K1 + K2) v = v^T K1 v + v^T K2 v >= 0`. (b) Valid. `v^T (c K1) v = c v^T K1 v >= 0` because
`c > 0`. (c) Not valid in general. Take `k2 = 2 k1`. Then `k = -k1`, and `k(x, x) < 0` for any
`x` with `k1(x, x) > 0`, so the Gram matrix is not positive semidefinite. Full credit needs a
proof or a concrete counterexample, not just the words "valid" or "not valid".

**Question 3 (12 points).** (a) `k(x, x) = exp(0) = 1`. The exponent is never positive, so
`0 < k(x, z) <= 1`. (b) For a large `gamma` the Gram matrix approaches the identity, because
every point is similar only to itself. For a `gamma` near zero it approaches the all-ones
matrix, because every pair of points looks the same. (c) With an identity Gram matrix the model
can fit any labeling of the training set and says nothing about new points: overfitting. With
an all-ones matrix every point gets the same score, so the model can only predict one class:
underfitting.

**Question 4 (13 points).** (a) Minimize `(1/2) ||w||^2 + C sum_i xi_i` subject to
`y_i (w . phi(x_i) + b) >= 1 - xi_i` and `xi_i >= 0`. The slack `xi_i` measures how far point
`i` is on the wrong side of its margin. `C` is the price of slack: a larger `C` fits the
training data harder. (b) `alpha_i = 0`: on the correct side, on or outside the margin, and not
a support vector. `0 < alpha_i < C`: exactly on the margin. `alpha_i = C`: inside the margin or
misclassified. (c) Any two of: lower `gamma`, lower `C`, standardize the features, collect more
training data. Each one makes the boundary smoother or the fit less tight. 2.5 points per
change that comes with a reason.
""",
    ),
]

# 4. syllabus.md
SYLLABUS = """\
# CS 4990 Applied Machine Learning: Syllabus

Fall 2025. 3 credit hours. Lectures on Mondays and Wednesdays, 2:00 to 3:15 pm, in Engineering
Hall 1.120.

## Course description

This course teaches you to build machine learning systems that work on real data. We cover the
main supervised and unsupervised methods (linear models, kernel methods, trees and ensembles,
neural networks, clustering and dimensionality reduction) with an emphasis on practice: choosing
a model, tuning it without fooling yourself, reading the results, and explaining them to someone
else.

## Prerequisites

- Data structures, and one course in probability or statistics
- Linear algebra: vectors, matrices, eigenvalues
- Comfortable programming in Python. We use NumPy from the first week.

## Course staff and office hours

| Who | When | Where |
| --- | --- | --- |
| Instructor | Tuesdays, 3:30 to 5:00 pm | Engineering Hall 2.214 |
| Teaching assistants | Mondays, 4:00 to 6:00 pm | Engineering Hall 1.300 |
| Teaching assistants | Thursdays, 10:00 am to 12:00 pm | Engineering Hall 1.300 |
| Online office hours | Fridays, 1:00 to 2:00 pm | Link on the course site |

Ask questions about lectures and homework on the course forum, so that everyone sees the answer.
For personal matters (grades, accommodations, illness) email cs4990-staff@university.example.
We answer within one working day.

## Materials

There is no required textbook. Slides and lecture notes are posted on the course site after each
class, and optional readings are linked from the schedule. You need a laptop that can run Python
3.11 or newer. Setup instructions are in Homework 1.

## Grading breakdown

| Component | Weight |
| --- | --- |
| Homework (5 assignments, 7% each) | 35% |
| Midterm exam | 20% |
| Final exam | 20% |
| Course project (teams of 3) | 20% |
| Participation (in-class polls) | 5% |

Letter grades: A is 90 and above, B is 80 to 89, C is 70 to 79, D is 60 to 69, F is below 60.
We may lower these cutoffs at the end of the term. We will not raise them.

Participation is graded on answering the in-class polls, not on getting them right. You can miss
four lectures without losing participation credit.

## Homework

There are five assignments, each with written questions and a programming part. Homework is
released on the course site and is due at 11:59 pm on the due date, through the course
submission site. The programming parts are checked by an autograder, and you can resubmit as
often as you like before the deadline. All dates are in schedule_and_deadlines.txt.

## Exams

- Midterm exam: Wednesday, October 15, in class. It covers Lectures 1 to 13.
- Final exam: Wednesday, December 10, 2:00 to 4:30 pm, in Engineering Hall 1.120. It is
  cumulative, with more weight on the second half of the course.

Both exams are closed book. You may bring one letter-size sheet of handwritten notes, both
sides. There are no makeup exams except for documented emergencies. Tell us before the exam if
you can.

## Course project

Teams of three apply machine learning to a problem of their choice. The project has a proposal,
a milestone report, a poster presentation and a final report. See project_guidelines.pdf for
what we expect, and the project logistics sheet for dates and submission rules.

## Late policy

- Every student has three free late days for homework across the term. A late day extends a
  deadline by 24 hours. You can use at most two on any one assignment.
- Once your late days are used up, late homework loses 10% of the assignment's points per day.
- No homework is accepted more than three days after its deadline, because we release solutions.
- Late days do not apply to exams or to project deliverables. The project has its own late rules.
- You do not need to ask before using a late day. The submission site counts them for you.

## Regrade requests

Submit regrade requests through the submission site within one week of the grade being
released. Explain in one or two sentences what you think was graded incorrectly. We regrade the
whole question, so the score can go up or down.

## Academic integrity

You may discuss homework problems with classmates at the level of ideas. Everything you submit
must be written by you: your own words and your own code. List the people you discussed each
assignment with at the top of your submission.

Not allowed: copying code or written answers from another student, from earlier terms, or from
solutions posted online; sharing your solutions with other students; posting course materials
or solutions on public sites.

AI coding assistants may be used to explain concepts and error messages. They may not be used to
produce the answers or the code that you submit. If you used one, say how in your submission.

Suspected violations are referred to the university's academic integrity office. The usual
penalty is a zero on the assignment and a one-letter reduction in the course grade.

## Accommodations

If you need an accommodation for a disability, contact the disability services office and send
us your letter in the first two weeks of the term, so that we can arrange exams in time.
"""

# 5. schedule_and_deadlines.txt: every weekday below is right for the calendar year 2025.
SCHEDULE = """\
CS 4990 Applied Machine Learning: Schedule and Deadlines
Fall 2025

Lectures are Mondays and Wednesdays, 2:00 to 3:15 pm, in Engineering Hall 1.120.
Everything is due at 11:59 pm on the date shown, on the course submission site.
The late policy is in the syllabus. This file is the reference for dates: if a
date changes, we update it here and post a note on the course forum.

WEEK 1
  Mon Aug 25   Lecture 1: Course overview and the machine learning workflow
  Wed Aug 27   Lecture 2: Linear regression and least squares
  Wed Aug 27   OUT  Homework 1 (linear regression and gradient descent)

WEEK 2
  Mon Sep 1    No class (holiday)
  Wed Sep 3    Lecture 3: Gradient descent and feature scaling

WEEK 3
  Mon Sep 8    Lecture 4: Overfitting and regularization (ridge and lasso)
  Wed Sep 10   Lecture 5: Model selection and cross-validation
  Wed Sep 10   OUT  Homework 2 (regularization and logistic regression)
  Fri Sep 12   DUE  Homework 1

WEEK 4
  Mon Sep 15   Lecture 6: Logistic regression
  Wed Sep 17   Lecture 7: Evaluation metrics and class imbalance
  Fri Sep 19   DUE  Project team sign-up

WEEK 5
  Mon Sep 22   Lecture 8: Linear classifiers and the perceptron
  Wed Sep 24   Lecture 9: Kernels and the kernel trick
  Wed Sep 24   OUT  Homework 3 (kernel methods)
  Fri Sep 26   DUE  Homework 2

WEEK 6
  Mon Sep 29   Lecture 10: Support vector machines
  Wed Oct 1    Lecture 11: Kernel SVMs in practice (worked example, bring a laptop)
  Fri Oct 3    DUE  Project proposal (1 page)

WEEK 7
  Mon Oct 6    Lecture 12: Decision trees
  Wed Oct 8    Lecture 13: Bagging and random forests
  Fri Oct 10   DUE  Homework 3

WEEK 8
  Mon Oct 13   Lecture 14: Midterm review
  Wed Oct 15   MIDTERM EXAM, in class, covers Lectures 1 to 13

WEEK 9
  Mon Oct 20   Lecture 15: Boosting
  Mon Oct 20   OUT  Homework 4 (trees and ensembles)
  Wed Oct 22   Lecture 16: Neural networks 1, from perceptrons to multilayer networks

WEEK 10
  Mon Oct 27   Lecture 17: Neural networks 2, backpropagation
  Wed Oct 29   Lecture 18: Training deep networks in practice
  Wed Oct 29   OUT  Homework 5 (neural networks)
  Fri Oct 31   DUE  Homework 4

WEEK 11
  Mon Nov 3    Lecture 19: Convolutional networks
  Wed Nov 5    Lecture 20: Clustering with k-means and mixture models
  Fri Nov 7    DUE  Project milestone report (2 pages)

WEEK 12
  Mon Nov 10   Lecture 21: Dimensionality reduction and PCA
  Wed Nov 12   Lecture 22: Recommender systems
  Fri Nov 14   DUE  Homework 5

WEEK 13
  Mon Nov 17   Lecture 23: Machine learning in production (leakage, drift, monitoring)
  Wed Nov 19   Lecture 24: Fairness and interpretability

WEEK 14
  Mon Nov 24   Project work session (staff in the room, no lecture)
  Wed Nov 26   No class (fall break)

WEEK 15
  Mon Dec 1    Project poster session, part 1
  Wed Dec 3    Project poster session, part 2, and course wrap-up
  Fri Dec 5    DUE  Project final report, code and peer evaluation

FINALS WEEK
  Wed Dec 10   FINAL EXAM, 2:00 to 4:30 pm, Engineering Hall 1.120, cumulative

ALL DEADLINES IN ONE PLACE
  Fri Sep 12   Homework 1
  Fri Sep 19   Project team sign-up
  Fri Sep 26   Homework 2
  Fri Oct 3    Project proposal
  Fri Oct 10   Homework 3
  Wed Oct 15   Midterm exam
  Fri Oct 31   Homework 4
  Fri Nov 7    Project milestone report
  Fri Nov 14   Homework 5
  Mon Dec 1    Project poster session, part 1
  Wed Dec 3    Project poster session, part 2
  Fri Dec 5    Project final report, code and peer evaluation
  Wed Dec 10   Final exam
"""

# 6. lectures/lecture09_kernels.pptx: six slides, as (title, bullets) pairs.
LECTURE09_SLIDES = [
    (
        "Lecture 9: Kernels and the Kernel Trick",
        [
            "CS 4990 Applied Machine Learning, Wednesday, September 24",
            "Today: feature maps, the kernel trick, and the three kernels you will use most",
            "Homework 3 (kernel methods) is out today and due Friday, October 10",
            "Next lecture: support vector machines",
        ],
    ),
    (
        "Why linear models are not enough",
        [
            "A linear classifier draws one straight boundary: w . x + b = 0",
            "Some data cannot be split by any line: XOR, or one class inside a ring of the other",
            "Fix: add features such as x1^2, x2^2 and x1 * x2, then fit a linear model on those",
            "Problem: d inputs have about d^p products of degree p, far too many to store",
        ],
    ),
    (
        "Feature maps",
        [
            "A feature map phi sends each input x to a longer vector phi(x)",
            "A linear model on phi(x) is a nonlinear model on x",
            "Example in 2-D: phi(x) = (x1^2, sqrt(2) x1 x2, x2^2)",
            "Then phi(x) . phi(z) = (x . z)^2: the features never have to be written out",
        ],
    ),
    (
        "The kernel trick",
        [
            "Many algorithms touch the data only through inner products x_i . x_j",
            "Replace every inner product with a kernel k(x, z) = phi(x) . phi(z)",
            "We never compute phi(x), so the feature space can be huge, even infinite",
            "All the algorithm needs is the n by n Gram matrix K, with K[i][j] = k(x_i, x_j)",
            "k is a valid kernel when every Gram matrix is symmetric and positive semidefinite",
        ],
    ),
    (
        "Kernels you will actually use",
        [
            "Linear: k(x, z) = x . z. The baseline, and the right choice for sparse text features",
            "Polynomial: k(x, z) = (x . z + c)^p. Feature interactions up to degree p",
            "RBF (Gaussian): k(x, z) = exp(-gamma * ||x - z||^2). Similarity fades with distance",
            "Large gamma: only very close points count. Wiggly boundary, overfits",
            "Small gamma: far points count too. Smooth boundary, underfits",
            "Standardize the features before using any of them",
        ],
    ),
    (
        "Summary and what is next",
        [
            "A kernel is a similarity function that acts as an inner product in a feature space",
            "The kernel trick buys nonlinear models for the price of an n by n Gram matrix",
            "Memory and time grow with n^2: slow beyond tens of thousands of examples",
            "Lecture 10 (Monday, September 29): support vector machines, where kernels pay off",
            "Homework 3: write rbf_kernel yourself, then train and tune a kernel SVM",
        ],
    ),
]

# 7. lectures/lecture10_notes.md: long on purpose (over 15,000 characters), so that the card
# builder has to take excerpts from the middle and the end as well as the opening.
LECTURE10_NOTES = """\
# Lecture 10: Support Vector Machines

CS 4990 Applied Machine Learning, Fall 2025. Lecture notes for Monday, September 29.

These notes cover what we did on the board in Lecture 10 and fill in the steps we skipped in
class. They pick up where the Lecture 9 slides on kernels stopped. Read sections 1 to 4 before
you start Homework 3. Sections 5 and 6 explain why the starter code asks you for a Gram matrix.

## 1. Which separating line is best?

Take a binary classification problem with inputs `x` in `R^d` and labels `y` in `{-1, +1}`. A
linear classifier predicts `sign(w . x + b)`. If the two classes can be separated by a line (a
hyperplane in higher dimensions), then infinitely many lines separate them, and all of them
have zero training error. Logistic regression picks one of them. The perceptron from Lecture 8
picks another, and which one it picks depends on the order of the training data.

Training error cannot tell these lines apart, but intuition can. A line that passes very close
to a training point is fragile: a little noise in that point, or a new point right next to it,
lands on the wrong side. A line that stays far away from every training point leaves room for
error. The support vector machine (SVM) turns this intuition into an optimization problem:
among all separating hyperplanes, choose the one whose distance to the nearest training point
is largest. That distance is called the margin.

## 2. Margins

### 2.1 Distance from a point to a hyperplane

The hyperplane is the set of points with `w . x + b = 0`, and the vector `w` is perpendicular
to it. The signed distance from a point `x` to the hyperplane is

```
distance(x) = (w . x + b) / ||w||
```

For a training pair `(x_i, y_i)` that is classified correctly, `y_i (w . x_i + b)` is positive,
so its distance to the hyperplane is `y_i (w . x_i + b) / ||w||`. The margin of the classifier
is the smallest of these distances over the training set:

```
margin = min over i of  y_i (w . x_i + b) / ||w||
```

### 2.2 Fixing the scale

Multiplying `w` and `b` by the same positive number does not move the hyperplane, so `(w, b)`
is only defined up to scale. We remove that freedom by requiring that the closest points
satisfy `y_i (w . x_i + b) = 1` exactly. With this convention every training point satisfies

```
y_i (w . x_i + b) >= 1
```

and the margin is simply `1 / ||w||`. No training point lies strictly between the two parallel
hyperplanes `w . x + b = +1` and `w . x + b = -1`. The width of that empty band is `2 / ||w||`.

## 3. The hard-margin SVM

Maximizing `1 / ||w||` is the same as minimizing `||w||^2`, which is easier to work with
because it is smooth and convex. The hard-margin SVM is

```
minimize over w, b:    (1/2) ||w||^2
subject to:            y_i (w . x_i + b) >= 1     for i = 1, ..., n
```

Three facts are worth remembering.

1. It is a convex quadratic program: a quadratic objective with linear constraints. It has one
   global minimum and no other local minima, and standard solvers handle it.
2. The solution depends only on the points that sit exactly on the margin, where the constraint
   holds with equality. These points are the support vectors. If you delete any other training
   point and retrain, you get the same `w` and `b`.
3. It has no solution when the classes are not linearly separable, because the constraints
   cannot all hold at once. Real data is almost never separable, which is why we need section 4.

### 3.1 A small example by hand

Take three points in the plane: a positive point at `(2, 2)` and negative points at `(1, 0)`
and `(0, 0)`. The closest pair with opposite labels is `(2, 2)` and `(1, 0)`. No separating
hyperplane can have a margin larger than half the distance between them, and the perpendicular
bisector of the segment between them reaches exactly that. Its normal vector points along
`(2, 2) - (1, 0) = (1, 2)`, so `w = a (1, 2)` for some `a > 0`. Requiring `w . x + b = +1` at
`(2, 2)` and `w . x + b = -1` at `(1, 0)` gives two equations:

```
6a + b = +1
 a + b = -1        so   a = 0.4,   b = -1.4,   w = (0.4, 0.8)
```

The margin is `1 / ||w|| = 1 / sqrt(0.8) = 1.118`, which is half the distance between the two
closest points, as promised. The third point `(0, 0)` has `y (w . x + b) = 1.4`. That is more
than 1, so the point satisfies its constraint with room to spare and is not a support vector.
Move it farther away from the boundary, or delete it, and the answer does not change.

## 4. Soft margins

### 4.1 Slack variables

We let each point violate the margin by an amount `xi_i >= 0`, called its slack, and we pay for
the total violation:

```
minimize over w, b, xi:   (1/2) ||w||^2 + C * sum_i xi_i
subject to:               y_i (w . x_i + b) >= 1 - xi_i     for i = 1, ..., n
                          xi_i >= 0                         for i = 1, ..., n
```

The slack of a point tells you where it is:

| Slack | Where the point is |
| --- | --- |
| `xi_i = 0` | on the margin or outside it, correctly classified |
| `0 < xi_i < 1` | inside the margin band, still correctly classified |
| `xi_i = 1` | exactly on the decision boundary |
| `xi_i > 1` | on the wrong side: a training error |

So the sum of the slacks is an upper bound on the number of training errors.

### 4.2 What C does

`C` is the price of one unit of slack. It trades margin width against violations.

- Large `C`: violations are expensive. The optimizer narrows the margin to get more training
  points right. Low bias, high variance. As `C` grows without bound we get the hard margin back.
- Small `C`: violations are cheap. The optimizer keeps a wide margin and tolerates points
  inside it or on the wrong side. Higher bias, lower variance.

`C` plays the role that `1 / lambda` plays in ridge regression (Lecture 4): it is an inverse
regularization strength. A common exam mistake is to say that a larger `C` means more
regularization. It is the other way around.

### 4.3 The hinge loss view

At the optimum each slack is as small as its constraints allow, so
`xi_i = max(0, 1 - y_i (w . x_i + b))`. Substituting this into the objective gives a problem
with no constraints:

```
minimize over w, b:   (1/2) ||w||^2 + C * sum_i max(0, 1 - y_i f(x_i)),   where f(x) = w . x + b
```

The function `max(0, 1 - y f(x))` is the hinge loss. This form shows that an SVM is a
regularized linear model like the ones from Lectures 4 and 6, with a different loss. Writing
`m = y f(x)`:

| Model | Loss on one example |
| --- | --- |
| Perceptron | `max(0, -m)` |
| Logistic regression | `log(1 + exp(-m))` |
| SVM | `max(0, 1 - m)` |

The hinge loss is exactly zero once an example is on the right side with room to spare
(`m >= 1`). Such examples add nothing to the objective or to its gradient, which is the
loss-function explanation of why only some training points matter. The logistic loss is never
exactly zero, so every training point pulls on a logistic regression fit at least a little.

Because this form has no constraints, a linear SVM can be trained by subgradient descent. With
`V` the set of examples that currently violate the margin (`y_i f(x_i) < 1`):

```
subgradient for w:   w - C * sum over i in V of  y_i x_i
subgradient for b:     - C * sum over i in V of  y_i
```

## 5. The dual problem

### 5.1 Why bother

The problem in section 4 (the primal) has one variable per feature. There is an equivalent
problem (the dual) with one variable per training example. The dual matters for two reasons.
It shows exactly which training points are support vectors. And the data enters it only through
inner products `x_i . x_j`, which is what lets us plug in a kernel in section 6.

### 5.2 Deriving it

Introduce a multiplier `alpha_i >= 0` for each margin constraint and a multiplier `mu_i >= 0`
for each constraint `xi_i >= 0`. The Lagrangian is

```
L = (1/2) ||w||^2 + C * sum_i xi_i
    - sum_i alpha_i * [ y_i (w . x_i + b) - 1 + xi_i ]
    - sum_i mu_i * xi_i
```

Setting the derivatives with respect to the primal variables to zero gives

```
dL/dw    = 0    gives    w = sum_i alpha_i y_i x_i
dL/db    = 0    gives    sum_i alpha_i y_i = 0
dL/dxi_i = 0    gives    alpha_i = C - mu_i,   so   0 <= alpha_i <= C
```

Substituting these back removes `w`, `b` and the slacks, and leaves the dual:

```
maximize over alpha:   sum_i alpha_i - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
subject to:            0 <= alpha_i <= C     for i = 1, ..., n
                       sum_i alpha_i y_i = 0
```

The first equation says that the weight vector is a weighted sum of training points. The upper
bound `C` is the only place where the soft margin shows up: the hard-margin dual is the same
problem without the upper bound.

### 5.3 Reading the solution

The complementary slackness conditions, which are part of the KKT conditions, say what each
`alpha_i` means:

| Multiplier | Margin | Meaning |
| --- | --- | --- |
| `alpha_i = 0` | `y_i f(x_i) >= 1` | not a support vector; deleting the point changes nothing |
| `0 < alpha_i < C` | `y_i f(x_i) = 1` | a support vector exactly on the margin |
| `alpha_i = C` | `y_i f(x_i) <= 1` | a support vector inside the margin or misclassified |

Predictions need only the support vectors:

```
f(x) = sum_i alpha_i y_i (x_i . x) + b        (the sum runs over the support vectors)
```

To recover `b`, take any support vector with `0 < alpha_i < C`. It sits exactly on the margin,
so `y_i f(x_i) = 1`, which gives `b = y_i - sum_j alpha_j y_j (x_j . x_i)`. In practice we
average this over all such points to reduce rounding error.

In the three-point example from section 3.1 the multipliers are `0.4` for `(2, 2)`, `0.4` for
`(1, 0)` and `0` for `(0, 0)`. Check: `0.4 (2, 2) - 0.4 (1, 0) = (0.4, 0.8)`, which is the `w`
we found by hand, and the multipliers of the two classes cancel, as the second equation
requires.

The number of support vectors is a useful diagnostic. If nearly every training point is a
support vector, the model is either underfitting badly or memorizing the data, and prediction
is slow.

## 6. Kernel SVMs

### 6.1 Plugging in a kernel

The dual objective and the prediction rule touch the inputs only through inner products.
Replace every `x_i . x_j` with a kernel `k(x_i, x_j)`, as in Lecture 9, and the SVM becomes a
linear classifier in the feature space of that kernel, without ever computing the features:

```
maximize over alpha:   sum_i alpha_i - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j k(x_i, x_j)
f(x) = sum_i alpha_i y_i k(x_i, x) + b
```

The training data enters only through the `n` by `n` Gram matrix `K` with
`K[i][j] = k(x_i, x_j)`. This is why `train_svm` in Homework 3 builds the Gram matrix first and
hands it to the solver.

Notice what we gave up. With a kernel there is usually no explicit `w` to store, so the model
has to keep the support vectors themselves, and each prediction costs one kernel evaluation per
support vector.

### 6.2 The RBF kernel and gamma

The kernel you will use most is the RBF (Gaussian) kernel:

```
k(x, z) = exp(-gamma * ||x - z||^2)
```

With this kernel `f(x)` is a weighted vote of the support vectors in which each vote fades with
distance. `gamma` sets how fast it fades.

- Large `gamma`: each support vector influences only its immediate neighborhood, and the
  boundary can wrap around individual points. Training accuracy approaches 100 percent while
  validation accuracy drops. This is overfitting.
- Small `gamma`: each support vector influences a wide region, the bumps blur together, and the
  boundary becomes almost linear. This is underfitting.

`gamma` and `C` interact. A smoother kernel (a smaller `gamma`) usually needs a larger `C` to
fit the same data, so the good settings tend to lie along a diagonal band of the (`C`, `gamma`)
grid. That is why we search over both at once instead of tuning one and then the other.

### 6.3 Other kernels

- Linear, `k(x, z) = x . z`: the right default when there are many features compared with
  examples, as with bag-of-words text. It is fast and gives back an explicit `w` to inspect.
- Polynomial, `k(x, z) = (x . z + c)^p`: models feature interactions up to degree `p`. It is
  sensitive to feature scale, and high degrees are numerically touchy.
- RBF: the default for dense data of low to medium dimension when a linear model underfits.

## 7. Training an SVM in practice

### 7.1 Scale the features first

Kernels are built from distances and inner products, so a feature measured in thousands drowns
out a feature measured in fractions. Standardize every feature to zero mean and unit variance.
Compute the mean and variance on the training split only, then apply the same transformation
to the validation and test splits. Forgetting this step is the most common reason an SVM does
badly on the homework.

### 7.2 Choose C and gamma by cross-validation

Start with a coarse grid on a logarithmic scale, then refine around the best cell:

```
C      in  {0.01, 0.1, 1, 10, 100, 1000}
gamma  in  {0.001, 0.01, 0.1, 1, 10}
```

For each pair, run 5-fold cross-validation on the training set and record the mean validation
accuracy. Pick the best pair, retrain on the whole training set, and only then evaluate on the
test set, once. If the best cell is on the edge of the grid, extend the grid in that direction
and repeat. A reasonable first guess for `gamma` on standardized data with `d` features is
`1 / d`.

### 7.3 Know the cost

Training a kernel SVM needs the Gram matrix, which has `n^2` entries, and solver time grows
somewhere between `n^2` and `n^3`. That is comfortable up to tens of thousands of examples and
painful beyond. For large `n`, train a linear SVM by stochastic gradient descent on the hinge
loss, or approximate the kernel with random features. Prediction time is proportional to the
number of support vectors, which matters when the model has to answer quickly.

### 7.4 More than two classes, and probabilities

An SVM is a binary classifier. For `m` classes, libraries train either one classifier per class
against the rest (`m` models) or one per pair of classes (`m (m - 1) / 2` models) and combine
their votes. You rarely need to do this by hand.

The output `f(x)` is a signed score, not a probability. If you need probabilities, fit a
logistic function to the scores on held-out data. Do not read `f(x) = 2.3` as "90 percent sure".

### 7.5 Unbalanced classes

When one class is rare, the plain objective sacrifices it. Give the rare class a larger `C`
(most libraries call this a class weight), and judge the model by precision and recall, or by
the area under the ROC curve from Lecture 7, instead of by accuracy.

## 8. When to use an SVM

| Situation | Reasonable first choice |
| --- | --- |
| Many sparse features, such as text | linear SVM or logistic regression |
| Up to about 50,000 examples, dense features, nonlinear boundary | RBF SVM |
| Hundreds of thousands of examples or more | linear model or boosted trees |
| Mixed categorical and numeric columns, missing values | tree ensembles (Lectures 12 and 13) |
| You need calibrated probabilities | logistic regression |
| Images, audio, long text | neural networks (Lectures 16 to 19) |

## 9. Common mistakes

1. Not scaling the features (section 7.1).
2. Tuning `C` and `gamma` on the test set. The number you report is then optimistic and
   meaningless.
3. Saying that a large `C` means strong regularization. A large `C` means weak regularization.
4. Reporting training accuracy for an RBF SVM. With a large enough `gamma` it is always close
   to 100 percent and tells you nothing.
5. Running a kernel SVM on a million rows and waiting for it to finish.
6. Fitting the scaler on all the data before splitting it. That leaks information from the
   validation set into training.

## 10. Summary

- An SVM picks the separating hyperplane with the largest margin, `1 / ||w||`.
- The soft margin allows violations at a price of `C` per unit of slack. It is the same as
  minimizing the hinge loss plus an L2 penalty.
- The solution is determined by the support vectors: the points on the margin, inside it, or
  on the wrong side of it.
- The dual problem uses only inner products, so a kernel turns the linear SVM into a nonlinear
  classifier. The RBF kernel adds a width parameter `gamma`.
- In practice: standardize, grid search `C` and `gamma` with cross-validation, and look at the
  number of support vectors.

## 11. Check your understanding

These questions are not graded. We go over the answers at the start of Lecture 11.

1. You train a hard-margin SVM and then delete a training point that is not a support vector.
   Does the decision boundary change? What if you delete a support vector?
2. In a soft-margin SVM, what is the slack of a point that lies exactly on the decision
   boundary?
3. You multiply every feature by 2 and retrain an RBF SVM with the same `gamma`. Is the model
   the same? Which `gamma` would give the same model as before?
4. Your grid search picks `C = 1000` and `gamma = 10`, both on the edge of the grid. What do
   you do next?
5. Why does a linear SVM on 20,000 bag-of-words features often match an RBF SVM on the same
   data?

## Next time

Lecture 11 (Wednesday, October 1) is a worked example: we tune an RBF SVM on a small tabular
dataset from start to finish, mistakes included. Bring a laptop. Homework 3 is due Friday,
October 10 at 11:59 pm.
"""

# 8. project/project_guidelines.pdf: what the project is. Dates live in the logistics sheet.
PROJECT_GUIDELINES = """\
# Course Project Guidelines

CS 4990 Applied Machine Learning, Fall 2025

## Goals

The project is your chance to take a machine learning problem from a vague question to a result
you can defend. In a team of three you pick a question, find data, build a simple baseline,
improve on it, and report honestly what worked and what did not. The project is worth 20% of
the course grade. By the end you should be able to:

- Frame a real problem as a learning task with a clear target and a clear metric.

- Build a simple baseline first, then improve on it one step at a time.

- Evaluate on held-out data without leaking information from the test set.

- Analyze the errors: say what the model gets wrong, and why.

- Explain the work in a short report and at a poster.

## Choosing a topic

A good project asks one clear question and has its data in hand within the first two weeks. Use
a dataset that is public, or one that you are allowed to share with the course staff. Aim small:
a careful comparison of two methods on one dataset beats an ambitious system that never trains.

## Example topics

- Bike share demand. Predict hourly rentals per station from weather and calendar features.
  Compare ridge regression with gradient boosted trees.

- Leaf disease photos. Classify photos of plant leaves as healthy or as one of three diseases
  by fine-tuning a small pretrained convolutional network.

- Forum spam. Detect spam posts on a student club message board with bag-of-words features, a
  linear SVM and a small neural network.

- Building energy. Forecast the next day's electricity use of campus buildings from past load
  and outdoor temperature.

- Elective recommendations. Recommend elective courses from synthetic enrollment histories with
  matrix factorization.

- Kernel methods on small tables. Compare RBF SVMs with random forests on five tabular datasets
  of under 5,000 rows each, with careful tuning for both.

Not allowed: a project that you also submit to another course, data with private information
about other people, or a literature survey with no experiments. If you are unsure, ask at
office hours before the proposal is due.

## Deliverables

- Proposal (1 page). The question, the dataset and how you will get it, the baseline, the
  metric, and the biggest risk. We return comments within a week.

- Milestone report (2 pages). A description of the data with at least one plot, baseline
  results on a validation split, and a plan for the remaining weeks.

- Poster presentation. A poster and a five-minute explanation at the in-class poster session.
  Every team member should be able to answer questions.

- Final report (6 pages). Introduction, related work, data, methods, experiments, error
  analysis and conclusions. Include one table that compares every method with the baseline.

- Code. A repository or zip file with a README and one command that reproduces the main results
  table.

- Peer evaluation. A short individual form about how the work was shared in your team.

## Grading rubric

The project is graded out of 100 points.

- Proposal: 10 points. A clear question, a feasible plan, a sensible baseline and metric.

- Milestone report: 15 points. The data is in hand and described, a baseline runs, and the next
  steps are concrete.

- Poster presentation: 15 points. A clear story, readable figures, good answers to questions.

- Final report: 45 points. Problem and data 10, methods and baseline 10, evaluation and error
  analysis 15, writing and figures 10.

- Code and reproducibility: 10 points. The command in the README reproduces the main table.

- Peer evaluation: 5 points. Based on the forms your teammates fill in.

Negative results are fine. A carefully evaluated method that does not beat the baseline can
earn full marks. A high score from a leaky evaluation cannot.

## Compute

Projects should run on a laptop or on the department's shared GPU queue, where each team gets
40 GPU hours for the term. Plan experiments that finish in under two hours each, and keep a log
of every run that you report.

## Dates and submission

Deadlines, team size rules, file naming and the late policy are on the project logistics sheet.
Read it before you sign up for a team.
"""

# 9. project/project_logistics_sheet.md: about the project, but all dates and rules. It sits
# between a "Project" group and a "Logistics" group on purpose.
PROJECT_LOGISTICS = """\
# Course Project: Logistics Sheet

CS 4990 Applied Machine Learning, Fall 2025

This sheet has the dates and the rules for the course project. What the project is, example
topics and the grading rubric are in project_guidelines.pdf. If the two documents disagree
about a date or a rule, this sheet wins.

## Deadlines

| What | Due | Where |
| --- | --- | --- |
| Team sign-up | Friday, September 19, 11:59 pm | Sign-up form on the course site |
| Proposal (1 page) | Friday, October 3, 11:59 pm | Submission site |
| Milestone report (2 pages) | Friday, November 7, 11:59 pm | Submission site |
| Poster session | Monday, December 1 and Wednesday, December 3 | In class |
| Final report (6 pages) and code | Friday, December 5, 11:59 pm | Submission site |
| Peer evaluation | Friday, December 5, 11:59 pm | Form on the course site |

## Team size

- Teams have three students.
- A team of two or four needs approval. Email the course staff before the sign-up deadline and
  say why. A team of four is expected to do proportionally more.
- No solo projects.
- If you have no team by the sign-up deadline, fill in the form on your own and we will place
  you in a team by Monday, September 22.
- Changing teams after the proposal is due needs approval from the instructor.

## Submission rules

- One submission per team. One member uploads and adds the others as group members on the
  submission site. Check that every name is on the submission: we cannot add names after
  grading.
- Reports are PDFs in the course report template, with an 11 point font and one inch margins.
  Pages over the limit are not read. References and an optional appendix do not count toward
  the limit.
- Name the files `teamNN_proposal.pdf`, `teamNN_milestone.pdf`, `teamNN_final.pdf` and
  `teamNN_code.zip`, where NN is your team number from the sign-up sheet.
- The code zip must be under 50 MB. Leave out datasets and model checkpoints, and put download
  instructions in the README instead.
- Each team presents on one of the two poster session days. The assignments are posted on
  Monday, November 24. The department prints your poster if you upload the PDF by 12:00 pm on
  Wednesday, November 26.

## Late policy for the project

- Homework late days cannot be used on project deliverables.
- Proposal and milestone report: 10% of that deliverable's points off per day late, for at most
  two days. After that the deliverable gets a zero.
- Poster session: you must present on your assigned day. Two teams may swap days if both email
  us at least 48 hours ahead.
- Final report, code and peer evaluation: no late submissions. Final grades are due to the
  registrar the following week.
- If something serious happens to a team member (illness, a family emergency), email us before
  the deadline and we will work something out.

## Questions

Post general questions on the course forum under the "project" tag. For anything about your own
team, email cs4990-staff@university.example or come to office hours (the times are in the
syllabus).
"""

# 10. misc/apartment_move_checklist.txt: personal, unrelated, and should fit no course group.
MOVE_CHECKLIST = """\
APARTMENT MOVE CHECKLIST

Moving from the studio on Maple Street to the two-bedroom on Alder Avenue.
Move day: Saturday, June 14. Keys for the new place: Friday, June 13, after 3 pm.

SIX WEEKS BEFORE
[x] Give written notice to the current landlord (the lease needs 30 days)
[x] Measure the new bedroom and the stairwell: will the couch make it around the corner?
[ ] Get three quotes: a rental truck plus friends, or movers for half a day
[ ] Start a box of things to donate

FOUR WEEKS BEFORE
[ ] Book the truck (15 foot is enough) and reserve the elevator at the new building
[ ] Collect boxes from the grocery store and the bike shop; buy tape, markers, bubble wrap
[ ] Set up electricity and internet at the new address, starting June 13
[ ] Ask the upstairs neighbor about borrowing the hand truck

TWO WEEKS BEFORE
[ ] File the change of address with the post office
[ ] Update the address: bank, phone plan, renter's insurance, voter registration, pharmacy
[ ] Pack the books, the winter clothes and the kitchen things I do not use every day
[ ] Use up what is in the freezer and stop buying groceries in bulk
[ ] Confirm who is helping on Saturday and at what time (pizza and drinks are on me)

THE WEEK OF THE MOVE
[ ] Defrost the freezer on Thursday night
[ ] Pack a first-night bag: sheets, towel, toothbrush, phone charger, coffee, one pan, one mug
[ ] Take photos of every room in the old apartment after cleaning, for the deposit
[ ] Label every box with the room it goes to, on the side and not on the top
[ ] Keep the lease, the passport and the plant in the car, not in the truck

MOVE DAY
[ ] Pick up the truck at 8:00 am, check the fuel level and photograph any dents
[ ] Load heavy furniture first, mattresses along the walls, boxes last
[ ] Final walk-through with the landlord, hand over both sets of keys
[ ] Return the truck before 6:00 pm with a full tank

FIRST WEEK IN THE NEW PLACE
[ ] Test the smoke detector and find the breaker box and the water shut-off
[ ] Assemble the bed first and the desk second; everything else can wait
[ ] Say hello to the neighbors across the hall
[ ] Send the old landlord the forwarding address for the deposit
[ ] Hang the shelves (borrow a drill and a level)
"""

# The ten smoke files. Path inside the dump folder: (the function that writes it, its content).
SMOKE_FILES = {
    "hw3/hw3_spec.pdf": (write_pdf, HW3_SPEC),
    "hw3/hw3_starter.py": (write_text, HW3_STARTER),
    "hw3/hw3_solution.ipynb": (write_notebook, HW3_SOLUTION),
    "syllabus.md": (write_text, SYLLABUS),
    "schedule_and_deadlines.txt": (write_text, SCHEDULE),
    "lectures/lecture09_kernels.pptx": (write_deck, LECTURE09_SLIDES),
    "lectures/lecture10_notes.md": (write_text, LECTURE10_NOTES),
    # The one PDF with title metadata and bookmarks. The homework handout has neither, as many
    # real PDFs do, so the smoke test sees a PDF with an outline and one without.
    "project/project_guidelines.pdf": (
        partial(write_pdf, title="CS 4990 Course Project Guidelines", bookmarks=True),
        PROJECT_GUIDELINES,
    ),
    "project/project_logistics_sheet.md": (write_text, PROJECT_LOGISTICS),
    "misc/apartment_move_checklist.txt": (write_text, MOVE_CHECKLIST),
}

# Printed at the end as a suggestion only. The descriptions follow the example in docs/phase0.md.
SUGGESTED_GROUPS = """\
Suggested starting groups (not created here; add each one with `p0 group add`):
  HW
    covers:   homework specs, starter code, solutions, tests
    excludes: project files
  Logistics
    covers:   syllabus, schedule, deadlines, grading and submission rules
  Project
    covers:   project specs, examples, starter material
    excludes: deadlines and policies
Add later, to replay the walkthrough in docs/phase0.md:
  Lecture slides and notes
    covers:   lecture slides, lecture notes
"""

if __name__ == "__main__":
    main()
