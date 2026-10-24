#!/usr/bin/env python3
"""Gate PRs that change the importable `litellm` API without declaring it.

Scope is the pip package's SDK surface: what `import litellm` exposes. The
proxy's contract is HTTP, not Python, so `litellm.proxy.*` is out of scope,
as are re-exported stdlib names (`from typing import Union`) whose canonical
home is another package. Over a sample 17-day window that scoping took the
finding count from 46 to 11, and all 11 were real.

Two layers, both computed with griffe against the PR's base ref:

  1. Breaking changes (removed objects, changed signatures, changed defaults).
     Allowed only when the PR declares a breaking change the Conventional
     Commits way: a `!` in the title type, or a `BREAKING CHANGE:` footer.
  2. Newly exported top-level names (`litellm.<name>`) that the package owns.
     Allowed only under a `feat` or `fix` title, so a `chore`/`refactor` PR
     cannot quietly widen the public surface.

Attribute *value* changes are advisory: `Union[A, B] -> A | B` rewrites and
f-string conversions trip that check constantly without breaking anyone.

Usage:
    check_api_breaking_changes.py --base-ref origin/main --pr-title "feat: x"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    from griffe import Alias, Breakage, ExplanationStyle, Module, Object

DEFAULT_PACKAGE: Final = "litellm"

OUT_OF_SCOPE_PREFIXES: Final = ("litellm.proxy.",)

ADVISORY_KINDS: Final = frozenset({"ATTRIBUTE_CHANGED_VALUE"})

SURFACE_WIDENING_TYPES: Final = frozenset({"feat", "fix"})

CONVENTIONAL_TITLE_RE: Final = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?(?P<bang>!)?:\s*\S")

BREAKING_FOOTER_RE: Final = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)

ANSI_RE: Final = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class ApiFinding:
    kind: str
    path: str
    detail: str
    file: str | None
    line: int | None


@dataclass(frozen=True, slots=True)
class ApiDelta:
    blocking: tuple[ApiFinding, ...]
    advisory: tuple[ApiFinding, ...]
    added_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Declaration:
    commit_type: str | None
    breaking: bool


@dataclass(frozen=True, slots=True)
class Approved:
    pass


@dataclass(frozen=True, slots=True)
class UndeclaredBreakingChanges:
    findings: tuple[ApiFinding, ...]


@dataclass(frozen=True, slots=True)
class UndeclaredSurfaceWidening:
    names: tuple[str, ...]
    commit_type: str | None


Verdict = Approved | UndeclaredBreakingChanges | UndeclaredSurfaceWidening


def parse_declaration(pr_title: str, pr_body: str) -> Declaration:
    match: Final = CONVENTIONAL_TITLE_RE.match(pr_title.strip())
    bang: Final = bool(match and match.group("bang"))
    footer: Final = bool(BREAKING_FOOTER_RE.search(pr_body))
    return Declaration(
        commit_type=match.group("type") if match else None,
        breaking=bang or footer,
    )


def decide(delta: ApiDelta, declaration: Declaration) -> Verdict:
    if delta.blocking and not declaration.breaking:
        return UndeclaredBreakingChanges(delta.blocking)
    if delta.added_names and declaration.commit_type not in SURFACE_WIDENING_TYPES:
        return UndeclaredSurfaceWidening(delta.added_names, declaration.commit_type)
    return Approved()


def source_location(obj: Object | Alias) -> tuple[str | None, int | None]:
    try:
        return str(obj.filepath), obj.lineno
    except Exception:
        return None, None


def lookup(roots: tuple[Module, ...], path: str) -> Object | Alias | None:
    parts: Final = path.split(".")
    for root in roots:
        if parts[0] != root.name:
            continue
        try:
            return root.get_member(parts[1:])
        except Exception:
            continue
    return None


def home_path(obj: Object | Alias, roots: tuple[Module, ...] = (), seen: frozenset[str] = frozenset()) -> str | None:
    try:
        return obj.canonical_path
    except Exception:
        target: Final = getattr(obj, "target_path", None)
        if target is None or target in seen:
            return None
        next_hop: Final = lookup(roots, target)
        if next_hop is None:
            return target
        return home_path(next_hop, roots, seen | {target}) or target


def is_in_scope(obj: Object | Alias, package: str, roots: tuple[Module, ...] = ()) -> bool:
    if obj.path.startswith(OUT_OF_SCOPE_PREFIXES):
        return False
    home: Final = home_path(obj, roots)
    return home is not None and home.startswith(f"{package}.")


def to_finding(breakage: Breakage, style: ExplanationStyle) -> ApiFinding:
    file, line = source_location(breakage.obj)
    return ApiFinding(
        kind=breakage.kind.name,
        path=breakage.obj.path,
        detail=ANSI_RE.sub("", breakage.explain(style=style)).strip(),
        file=file,
        line=line,
    )


def top_level_names(module: Module, package: str, roots: tuple[Module, ...]) -> frozenset[str]:
    return frozenset(
        name
        for name, member in module.members.items()
        if not name.startswith("_") and is_in_scope(member, package, roots)
    )


def build_delta(
    old: Module,
    new: Module,
    breakages: Iterator[Breakage],
    style: ExplanationStyle,
    package: str = DEFAULT_PACKAGE,
) -> ApiDelta:
    roots: Final = (new, old)
    findings: Final = tuple(
        dict.fromkeys(
            to_finding(breakage, style) for breakage in breakages if is_in_scope(breakage.obj, package, roots)
        )
    )
    return ApiDelta(
        blocking=tuple(f for f in findings if f.kind not in ADVISORY_KINDS),
        advisory=tuple(f for f in findings if f.kind in ADVISORY_KINDS),
        added_names=tuple(sorted(top_level_names(new, package, roots) - top_level_names(old, package, roots))),
    )


def collect_delta(package: str, repo: Path, base_ref: str, head_ref: str | None) -> ApiDelta:
    import griffe

    old: Final = griffe.load_git(package, ref=base_ref, repo=repo, allow_inspection=False)
    new: Final = (
        griffe.load_git(package, ref=head_ref, repo=repo, allow_inspection=False)
        if head_ref
        else griffe.load(package, search_paths=[repo], allow_inspection=False)
    )
    return build_delta(old, new, griffe.find_breaking_changes(old, new), griffe.ExplanationStyle.ONE_LINE, package)


def render_annotations(findings: Sequence[ApiFinding]) -> str:
    return "\n".join(
        f"::error file={f.file},line={f.line}::{f.detail}" if f.file else f"::error::{f.detail}" for f in findings
    )


def render_summary(delta: ApiDelta, verdict: Verdict) -> str:
    header: Final = _verdict_header(verdict)
    blocking: Final = "\n".join(f"- `{f.path}` {f.detail}" for f in delta.blocking)
    added: Final = "\n".join(f"- `{name}`" for name in delta.added_names)
    advisory: Final = "\n".join(f"- `{f.path}` {f.detail}" for f in delta.advisory)
    sections: Final = (
        ("Breaking changes", blocking),
        ("New public names", added),
        ("Advisory (value changes, not gated)", advisory),
    )
    body: Final = "\n\n".join(f"### {title}\n{content}" for title, content in sections if content)
    return f"## Public API check\n\n{header}\n\n{body}".rstrip() + "\n"


def _verdict_header(verdict: Verdict) -> str:
    match verdict:
        case Approved():
            return "No undeclared public API changes."
        case UndeclaredBreakingChanges(findings):
            return (
                f"{len(findings)} breaking change(s) to the public `litellm` API are not declared. "
                "Add `!` after the type in the PR title (`feat!: ...`) or a `BREAKING CHANGE:` "
                "footer in the PR body, and document the migration."
            )
        case UndeclaredSurfaceWidening(names, commit_type):
            found: Final = f"`{commit_type}`" if commit_type else "an unparseable title"
            return (
                f"{len(names)} new public name(s) added under {found}. Widening the public API "
                "needs a `feat:` or `fix:` PR title."
            )
        case _:
            assert_never(verdict)


def _write(path_env: str, content: str) -> None:
    destination: Final = os.environ.get(path_env)
    if not destination:
        return
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(content + "\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", default=None)
    parser.add_argument("--pr-title", default="")
    parser.add_argument("--pr-body", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args: Final = _parse_args(argv)
    delta: Final = collect_delta(args.package, args.repo, args.base_ref, args.head_ref)
    declaration: Final = parse_declaration(args.pr_title, args.pr_body)
    verdict: Final = decide(delta, declaration)

    summary: Final = render_summary(delta, verdict)
    print(summary)
    _write("GITHUB_STEP_SUMMARY", summary)

    match verdict:
        case Approved():
            return 0
        case UndeclaredBreakingChanges(findings):
            print(render_annotations(findings))
            return 1
        case UndeclaredSurfaceWidening():
            return 1
        case _:
            assert_never(verdict)


if __name__ == "__main__":
    sys.exit(main())
