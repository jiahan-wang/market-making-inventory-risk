"""Hygiene checks that need nothing outside the standard library.

Run from the repository root: python tools/style_check.py

Checks, in order:
  * every module parses and carries a module docstring
  * no tabs, no trailing whitespace, no missing final newline
  * no imported name that the module never uses
  * generated .txt artifacts use LF, not CRLF

The last one is the regression guard for a real bug: Python's write_text() emits
CRLF on Windows unless newline is passed, while .gitattributes checks *.txt out
with eol=lf, so a regenerated table would stop matching its committed copy.
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP = {".git", ".ci_repro", "reproduced", "work", "__pycache__"}


def walk(suffix: str):
    for path in sorted(ROOT.rglob(f"*{suffix}")):
        if SKIP & set(path.parts):
            continue
        yield path


def unused_imports(tree: ast.AST) -> list[tuple[str, int]]:
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported[alias.asname or alias.name] = node.lineno
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    return [(name, line) for name, line in imported.items()
            if name not in used and name != "annotations"]


def main() -> int:
    problems: list[str] = []

    for path in walk(".py"):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            problems.append(f"{rel}: does not parse ({exc})")
            continue
        if not ast.get_docstring(tree):
            problems.append(f"{rel}: no module docstring")
        if not text.endswith("\n"):
            problems.append(f"{rel}: missing final newline")
        for number, line in enumerate(text.splitlines(), 1):
            if "\t" in line:
                problems.append(f"{rel}:{number}: tab character")
            if line != line.rstrip():
                problems.append(f"{rel}:{number}: trailing whitespace")
        for name, line in unused_imports(tree):
            problems.append(f"{rel}:{line}: {name} imported but unused")

    for path in walk(".txt"):
        if b"\r\n" in path.read_bytes():
            problems.append(f"{path.relative_to(ROOT)}: CRLF line endings")

    for line in problems:
        print(line)
    print(f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
