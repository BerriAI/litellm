"""Phase 0 spike — does `match` give us *compile-time* exhaustiveness under basedpyright strict?

This is the one genuine technical risk of the FP approach (Full Migration Plan, Mini Phase 0,
goal 1). We answer it before any OAuth logic commits to the style. Verdict (see README.md):
the bet HOLDS, with two non-obvious rules taken from the sibling v2 package `litellm/translation`.

Two `match` shapes are exhaustiveness-checked and are the only ones we use:

  (A) `match` over a closed `Enum`  -> what `resolve()` dispatches on (`outbound_credentials/types.py`).
  (B) `match self.tag` over an Expression `@tagged_union` whose `tag` is a `Literal`
      -> what `CredError` uses for its `summary`.

Both end in `assert_never(...)`: if any arm is deleted, the scrutinee no longer narrows to
`Never`, so `assert_never` becomes a type error — that "remove-an-arm" failure is the proof
the gate bites (reproduced in README.md).

Rejected patterns and why (do NOT use; recorded so the finding is not re-litigated):
  - `tag: str` + `match err:` over the union object -> NOT exhaustiveness-checked: one class,
    `str` tag never narrows to `Never`. basedpyright demands `case _:` and ignores missing tags.
  - `expression.Result` -> single class carrying both `.ok` and `.error`; unguarded `.ok`
    access is invisible to the checker. Replaced by the vendored `Ok | Error` union (result.py).
"""

from __future__ import annotations

from typing_extensions import assert_never

from .outbound_credentials.types import AuthSpecKind, CredError


# (A) Enum dispatch — the load-bearing case (`resolve()` uses exactly this shape).
def label_enum(kind: AuthSpecKind) -> str:
    match kind:
        case AuthSpecKind.authorization_code:
            return "per-user 3LO"
        case AuthSpecKind.client_credentials:
            return "service account"
        case AuthSpecKind.token_exchange:
            return "on-behalf-of"
        case AuthSpecKind.api_key:
            return "static header"
        case AuthSpecKind.passthrough:
            return "client-forwarded"
        case AuthSpecKind.none:
            return "no upstream auth"
        case AuthSpecKind.aws_sigv4:
            return "aws sigv4 signing"
    # Reached only if an enum member has no arm above. basedpyright then narrows `kind` to that
    # uncovered member (not `Never`), so this `assert_never` is a type error => the gate bit.
    assert_never(kind)


# (B) @tagged_union discriminated on a Literal `tag` — matched via the tag, not the object.
def http_status(err: CredError) -> int:
    match err.tag:
        case "unauthorized":
            return 401
        case "misconfigured":
            return 500
        case "upstream_unavailable":
            return 503
        case "unsupported_mode":
            return 500
        case "precondition_required":
            return 412
    assert_never(err.tag)


__all__ = ["label_enum", "http_status"]
