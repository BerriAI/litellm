# MCP Gateway v2 — Mini Phase 0 (the typed OAuth seam)

This directory is the clean-room v2 mini-chassis. It is the pre-Phase-1 graft that stands up
just enough of the typed/FP spine for the OAuth credential subdomain, so the highest-risk
security code is v2-shaped from line one instead of rewritten when the full S0 chassis lands in
Phase 2. It imports nothing from v1; v1 will only ever reach it through a thin adapter built in
Phase 1.

Scope is deliberately small: the typed credential seam (`oauth/types.py`), the vendored `Result`
(`result.py`), and the basedpyright match-exhaustiveness spike (`_spike_exhaustiveness.py`). No
transport, registry, CI gate, semgrep rules, import-linter layers, LOC caps, or composition root;
those land with the full S0 in Phase 2.

## The spike verdict: the bet HOLDS

The one genuine technical risk of the whole FP approach is whether Expression's `@tagged_union`
plus `match` actually satisfies basedpyright strict `reportMatchNotExhaustive` at compile time.
Mini Phase 0 answers that before any OAuth logic commits to the style. The answer is yes, with two
non-obvious rules taken from the sibling v2 effort. `_spike_exhaustiveness.py` pins the only two
`match` shapes we rely on, and both pass strict:

1. `match` over a closed `Enum` (what `resolve()` dispatches on against `AuthSpecKind`).
2. `match self.tag` over an Expression `@tagged_union` whose `tag` is a `Literal` (what `CredError`
   uses for its `summary`).

Both end in `assert_never(...)`. The exhaustiveness guarantee is real precisely because deleting an
arm breaks that tail: the scrutinee stops narrowing to `Never`, so `assert_never` becomes a type
error. That is the property we want, so the gate is load-bearing rather than decorative.

### Rejected patterns (do not reintroduce)

- `tag: str` plus `match err:` over the union object is NOT exhaustiveness-checked. One class with a
  `str` tag never narrows to `Never`, so basedpyright demands a `case _:` and silently ignores
  missing tags. We discriminate on a `Literal` tag instead.
- `expression.Result` is a single class carrying both `.ok` and `.error`, so an unguarded `.ok`
  access is invisible to the checker. We vendor an `Ok | Error` union (`result.py`) so reaching for
  the wrong side before a `match`/`isinstance` is a type error.

## Reproducing the proof

Run the gate (passes clean, 0 errors):

```
basedpyright --project litellm/proxy/gateway/mcp/pyrightconfig.json
```

Now prove the gate actually bites. Delete any one arm from `label_enum` in
`_spike_exhaustiveness.py` (for example the `AuthSpecKind.api_key` case) and re-run. basedpyright
reports two errors, which is the proof the exhaustiveness check is doing its job:

```
_spike_exhaustiveness.py:33:11 - error: Cases within match statement do not exhaustively handle all values
    Unhandled type: "Literal[AuthSpecKind.api_key]"  (reportMatchNotExhaustive)
_spike_exhaustiveness.py:44:18 - error: Argument of type "Literal[AuthSpecKind.api_key]" cannot be
    assigned to parameter "arg" of type "Never" in function "assert_never"  (reportArgumentType)
```

Restore the arm and the errors disappear. This is why adding a new `AuthSpecKind` member without a
`resolve()` arm fails the type gate rather than failing at runtime. `AuthSpecKind` covers v1's full
`MCPAuth` surface: the three OAuth grants (`authorization_code`, `client_credentials`,
`token_exchange`), the collapsed static-header family (`api_key`), client `passthrough`, `none`
(no upstream auth), and `aws_sigv4` (per-request signing).

## Toolchain notes

`expression` is declared in the litellm `proxy` extra and `basedpyright` in the dev group
(`pyproject.toml`); the base SDK install is untouched. The spike was validated against
`expression 5.6.0` and `basedpyright 1.39.8` (pyright 1.1.410) on Python 3.10.
