"""Source code: the file itself is the text; names of functions and classes are the outline."""

import ast
import re
import warnings
from pathlib import Path
from typing import NamedTuple

from .base import MAX_OUTLINE, Extracted, as_outline, first_line, read_text

HEADER_LINES = 10  # a comment counts as the file's title only this close to the top
MAX_DEFINITION_LINE = 300  # longer lines are minified or generated, and are not searched for names

_MODIFIERS = (
    r"(?:(?:export|default|async|static|public|private|protected|abstract|final|mutable|typedef)"
    r"\s+)*"
)

# How a definition line looks, one pattern per style, each with the way its match is shown.
_FUNCTION = (  # function name(...), and MATLAB's function out = name(...)
    re.compile(rf"\s*{_MODIFIERS}function\b\s*\*?\s*(?:[^=(]*=\s*)?([A-Za-z_$][\w$.!]*)"),
    "{}()",
)
_CLASS = (  # class Name, struct Name, interface Name, classdef Name; not "struct node *head;"
    re.compile(rf"\s*{_MODIFIERS}(class|classdef|struct|interface)\s+([A-Za-z_$][\w$]*)[^;]*$"),
    "{} {}",
)
_ASSIGNED = (  # name <- function(...) in R; const name = (...) => in JavaScript
    re.compile(
        r"\s*(?:(?:export|const|let|var)\s+)*([A-Za-z_$.][\w$.]*)\s*(?:<-|=)\s*(?:async\s+)?"
        r"(?:function\b|\([^()]*\)\s*(?::[^=]*)?=>|[A-Za-z_$][\w$]*\s*=>)"
    ),
    "{}()",
)
# Python that does not parse is often half prose (a notebook cell with a TODO in the middle), so
# these two are strict: "function and the rest" or "class of models" in a docstring is no name.
_PY_DEF = (re.compile(r"\s*(?:async\s+)?def\s+([^\W\d]\w*)\s*[(\[]"), "{}()")  # def name(
_PY_CLASS = (re.compile(r"\s*(class)\s+([^\W\d]\w*)\s*[(:\[]"), "{} {}")  # class Name: or Name(
_C_FUNCTION = (  # static int name(int a, char **b) {    (a definition, not a call or a prototype)
    re.compile(
        r"\s*(?!(?:if|else|for|while|switch|return|catch|throw|new|delete|do|case)\b)"
        r"(?:[\w:<>\[\],~]+[\s*&]+)+([A-Za-z_~][\w:~]*)\s*\([^;{}()]*\)[^;{}()=]*(?:\{.*)?$"
    ),
    "{}()",
)
_SHELL = (re.compile(r"\s*([A-Za-z_][\w-]*)\s*\(\s*\)"), "{}()")  # name() {
_SECTION = (  # \section{Title}: a LaTeX file has headings where code has functions
    re.compile(
        r"\s*\\(?:chapter|(?:sub){0,2}section)\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}"
    ),
    "{}",
)
_CREATE = (  # CREATE TABLE name
    re.compile(
        r"\s*create\s+(?:or\s+replace\s+)?(table|view|function|procedure)\s+"
        r"(?:if\s+not\s+exists\s+)?([\w.]+)",
        re.IGNORECASE,
    ),
    "{} {}",
)

_HASH = ("#",)
_PERCENT = ("%",)
_SLASHES = ("//", "/*", "* ")
_DASHES = ("--", "/*", "* ")
_COMMENT_START = "#%/*-=!' \t"  # comment marks and decoration before the words of a comment
_COMMENT_END = "#%/*-= \t"


class Language(NamedTuple):
    """How to read one kind of source file."""

    label: str  # the file type shown on the card
    comments: tuple[str, ...]  # what a comment line starts with
    definitions: tuple[tuple[re.Pattern, str], ...]  # the patterns that find names


LANGUAGES = {
    ".py": Language("Python code", _HASH, (_PY_DEF, _PY_CLASS)),  # only when it does not parse
    ".m": Language("MATLAB code", _PERCENT, (_FUNCTION, _CLASS)),
    ".r": Language("R code", _HASH, (_ASSIGNED,)),
    ".jl": Language("Julia code", _HASH, (_FUNCTION, _CLASS)),
    ".js": Language("JavaScript code", _SLASHES, (_FUNCTION, _CLASS, _ASSIGNED)),
    ".ts": Language("TypeScript code", _SLASHES, (_FUNCTION, _CLASS, _ASSIGNED)),
    ".java": Language("Java code", _SLASHES, (_C_FUNCTION, _CLASS)),
    ".c": Language("C code", _SLASHES, (_C_FUNCTION, _CLASS)),
    ".h": Language("C header", _SLASHES, (_C_FUNCTION, _CLASS)),
    ".cpp": Language("C++ code", _SLASHES, (_C_FUNCTION, _CLASS)),
    ".sh": Language("Shell script", _HASH, (_FUNCTION, _SHELL)),
    ".tex": Language("LaTeX source", _PERCENT, (_SECTION,)),
    ".sql": Language("SQL code", _DASHES, (_CREATE,)),
}


def _names(text: str, definitions: tuple[tuple[re.Pattern, str], ...]) -> list[str]:
    """Names of the definitions found line by line, in the order they appear."""
    names: list[str] = []
    for line in text.splitlines():
        if len(line) > MAX_DEFINITION_LINE:
            continue
        for pattern, shown in definitions:
            if match := pattern.match(line):
                names.append(shown.format(*match.groups()))
                break
        if len(names) == MAX_OUTLINE:
            break
    return names


def _parse(source: str) -> ast.Module | None:
    """Parse Python source into a syntax tree, or return None when it is not valid Python 3."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the file's own syntax warnings are not ours to show
            return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _tree_names(tree: ast.Module) -> list[str]:
    """Top-level functions and classes of a parsed module, each class followed by its methods."""
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, functions):
            names.append(f"{node.name}()")
        elif isinstance(node, ast.ClassDef):
            names.append(f"class {node.name}")
            names += [
                f"{node.name}.{item.name}()" for item in node.body if isinstance(item, functions)
            ]
    return names


def python_names(source: str) -> list[str]:
    """Function and class names in Python source. Also used for notebook cells.

    Read from the syntax tree. Source that does not parse (Python 2, notebook magics) is
    searched line by line instead.
    """
    tree = _parse(source)
    if tree is None:
        return _names(source, LANGUAGES[".py"].definitions)
    return _tree_names(tree)


def _first_comment(text: str, marks: tuple[str, ...]) -> str | None:
    """The first comment line at the top of a file that says something, without its marks.

    Only the first few lines count: a comment further down describes one step, not the file.
    """
    for line in text.split("\n", HEADER_LINES)[:HEADER_LINES]:
        comment = line.strip()
        if comment.startswith(marks) and not comment.startswith("#!") and "-*-" not in comment:
            title = first_line(comment.lstrip(_COMMENT_START).rstrip(_COMMENT_END))
            if title:
                return title
    return None


def extract_code(path: Path) -> Extracted:
    """Read a source file whole. The title is the first line of its docstring or first comment."""
    suffix = path.suffix.lower()
    language = LANGUAGES[suffix]
    text = read_text(path) or ""
    tree = _parse(text) if suffix == ".py" else None
    if tree is None:
        title, names = None, _names(text, language.definitions)
    else:
        title, names = first_line(ast.get_docstring(tree) or ""), _tree_names(tree)
    return Extracted(
        file_type=language.label,
        title=title or _first_comment(text, language.comments),
        outline=as_outline(names),
        text=text,
    )
