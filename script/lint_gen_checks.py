#!/usr/bin/env python3
"""Stage F4 — static audit per H3.

Walks `linux_voice_assistant/` and reports every function whose name
matches one of the K.1 callback patterns but lacks `@gen_checked` or
`@gen_independent`.

Exit status is non-zero when one or more violations are found, so this
script can run in CI as a regression guard.

The audit is deliberately conservative — only obvious callback shapes
are checked. Methods called directly (not via callback registration)
are exempt by virtue of not matching the regex.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Functions whose names match this regex are required to carry generation
# discipline (per H3 audit). Matches the patterns from the v1 docs.
CALLBACK_REGEX = re.compile(
    r"^(?:_on_[a-z_]+"
    r"|[a-z_]+_handler"
    r"|[a-z_]+_callback"
    r"|[a-z_]+_timer_fired"
    r"|[a-z_]+_finished"
    r"|[a-z_]+_detected)$",
)

# Allowlist of method names that match the regex but are deliberately
# gen-independent and don't carry an explicit `@gen_independent` marker
# (typically infrastructure methods on libraries we don't own).
EXPLICIT_ALLOWLIST = {
    # paho-mqtt callback signatures used by HABridge / cancel subscriber.
    "_on_connect",
    "_on_disconnect",
    "_on_message",
    # Entity API setters — not lifecycle callbacks despite the *_callback
    # name suffix. These are HA-side write-handlers, gen-independent by
    # nature (e.g. volume changes are intentionally cross-session).
    "set_volume_callback",
    # `attach_*_handler` / `*_handler` setters on HABridge — wire-time
    # plumbing, not K.1 callback handlers.
    "_attach_enrollment_handler",
}

GEN_DECORATORS = {"gen_checked", "gen_independent"}


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    names: List[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            names.append(dec.func.id)
    return names


def _scan_file(path: Path) -> List[Tuple[str, int, str]]:
    """Return a list of (relpath, lineno, funcname) violations.

    Only top-level / class-level functions are audited; nested closures
    are exempt by design (they capture their parent's scope, including
    a gen-checked outer wrapper).
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        print(f"::error file={path}::syntax error {exc}", file=sys.stderr)
        return []
    violations: List[Tuple[str, int, str]] = []

    def visit_body(body):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit_body(node.body)
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not CALLBACK_REGEX.match(node.name):
                continue
            if node.name in EXPLICIT_ALLOWLIST:
                continue
            decs = set(_decorator_names(node))
            if decs & GEN_DECORATORS:
                continue
            violations.append((str(path), node.lineno, node.name))

    visit_body(tree.body)
    return violations


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent / "linux_voice_assistant",
        help="Directory to scan",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output when no violations are found",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    violations: List[Tuple[str, int, str]] = []
    for py in sorted(root.rglob("*.py")):
        violations.extend(_scan_file(py))

    if violations:
        for path, lineno, name in violations:
            print(f"{path}:{lineno}: callback {name!r} lacks @gen_checked / @gen_independent")
        print(f"\n{len(violations)} gen-check violation(s) found", file=sys.stderr)
        return 1
    if not args.quiet:
        print("gen_checks: OK (no callback handlers missing discipline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
