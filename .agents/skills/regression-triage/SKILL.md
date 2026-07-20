---
name: regression-triage
description: Standard protocol for triaging any regression in BerriAI/litellm. Use whenever a report says something "used to work and now is broken", "worked in vX but not vY", a customer/issue reports a behavior change, or a test that previously passed now fails. Walks you through pinning the regression, finding the offending PR, doing a 5 Whys root cause analysis, and checking/adding e2e coverage in tests/e2e.
---

# Regression triage protocol

The standard, repeatable protocol for handling any regression in `BerriAI/litellm`. A regression is any behavior that used to work and now does not (or changed). Work the four phases below in order and produce the deliverable at the end. Do not skip a phase; if a phase genuinely does not apply, say so explicitly in the report with why.

The four phases, in order:

1. Reproduce and pin the regression (know exactly what broke and between which two points)
2. Find the offending PR that caused it
3. 5 Whys on what caused this
4. Check whether we have e2e tests for this in `tests/e2e` (and add one so it can never regress silently again)

## Phase 0: frame the regression

Before touching git, write down in one or two sentences:

- The exact broken behavior: the input (request/config/model), the expected output, and the actual output. Prefer a concrete `curl` against a live proxy or a minimal SDK snippet, not a prose description
- The "last known good" and "first known bad" reference points. These can be litellm versions (`v1.xx.y`), git SHAs, dates, or PR numbers. Get at least one good and one bad point; the offending-PR search needs a range to work in
- Which surface is affected (SDK `completion`/`embedding`, proxy route, router, UI, a specific provider). This decides which files and which DRI matter

If the reporter did not give a good/bad range, establish one before proceeding: check out a recent tag/commit that predates the report and confirm the good behavior, then confirm bad on `litellm_internal_staging` (or the reported version).

## Phase 1: reproduce and pin

Get a deterministic reproduction first; a regression you cannot reproduce cannot be bisected.

- Reproduce on the current `litellm_internal_staging` (or the exact bad version the reporter names). Capture the command and the wrong output verbatim
- Reproduce with real provider APIs where feasible, matching how the user hits it. Keys live in `.env`; never print, commit, or paste them into issues/PRs. For proxy-surface bugs, run a local proxy the way the root `CLAUDE.md` prescribes: `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log` and `curl` it
- Confirm the good reference point actually shows the correct behavior with the same command. Now you have a good SHA and a bad SHA

Repo is often a shallow clone. Before any history/bisect work run:

```bash
git rev-parse --is-shallow-repository   # if true:
git fetch --unshallow
```

## Phase 2: find the offending PR

Use the cheapest technique that pins the change, escalating to bisect only when needed.

Targeted history search (fast, use first when you know the symbol or line):

```bash
# who last changed a specific line / function
git log -L :<function_name>:<path/to/file.py>

# every commit that added or removed a specific string/symbol (pickaxe)
git log -S '<symbol_or_literal>' --oneline -- <path>
git log -G '<regex>' --oneline -- <path>          # regex variant

# blame the exact lines that produce the wrong behavior
git blame -L <start>,<end> <path/to/file.py>

# what changed in a directory between the good and bad points
git log --oneline <good_sha>..<bad_sha> -- <paths>
```

Bisect (authoritative when the cause is not obvious). Script it so it is reproducible:

```bash
git bisect start <bad_sha> <good_sha>
git bisect run bash -c '<build-if-needed> && <one-line repro that exits non-zero on the bug>'
git bisect reset
```

The repro command must exit `0` when good and non-zero when bad. Reuse the Phase 1 reproduction: wrap the `curl`/SDK check in a script that greps the output and exits accordingly. Prefer the SDK path over a full proxy boot inside `bisect run` when the bug reproduces at the SDK layer, because it is far faster per step.

Map the commit to its PR:

```bash
git show -s --format='%H %an <%ae>%n%s' <first_bad_sha>
git log --merges --oneline --ancestry-path <first_bad_sha>..HEAD | tail   # find the merge PR that carried it
```

The commit subject and the `(#NNNNN)` suffix give you the PR number and the author. Read the PR with `git_view_pr` (diff, description, and comments) to understand intent; do not assume the change was wrong, understand why it was made before proposing anything.

Record: the offending PR number and link, the author, the SHA, and the one-line explanation of what that PR changed that broke the behavior.

## Phase 3: 5 Whys

Do a real root cause analysis, not five restatements of the symptom. Each "why" answers the previous answer, drilling from the surface symptom to the systemic cause. The last why should land on something process/architecture-level that, if fixed, prevents a whole class of these, not just this one bug.

Template to fill in:

```
Problem statement: <what the user observes that is wrong>

Why 1 (why does the user see this?):      <immediate technical cause in the code path>
Why 2 (why did that happen?):             <the code condition / missing branch / wrong default>
Why 3 (why was it introduced?):           <what the offending PR did and the assumption it made>
Why 4 (why did review/CI let it through?):<the missing guard: no test, wrong test, gap in coverage>
Why 5 (why was that guard missing?):      <systemic: no e2e for this surface, convention gap, etc>

Root cause:  <the single systemic cause the fix should address>
Contributing factors: <secondary causes worth noting>
```

Rules for a good 5 Whys:

- Ground every "why" in evidence (a file/line, the offending diff, a missing test), not speculation. Link the lines
- At least one why must reach the test/coverage gap. Almost every regression is also a testing regression; naming that gap is what feeds Phase 4
- Distinguish the root cause (fix this to prevent recurrence) from the immediate fix (what unbreaks the user now). Both belong in the report
- If two independent causes exist, run two branches of whys rather than forcing one chain

## Phase 4: do we have e2e tests for this in tests/e2e?

Answer the literal question first, then close the gap.

Check existing coverage:

```bash
# is the behavior exercised by a live e2e suite?
rg -n '<feature_or_route_or_symbol>' tests/e2e/
# is it in the coverage registry (the definition of done)?
rg -n '<feature>' tests/e2e/coverage_registry/
# unit/regression coverage that mirrors the source file
rg -n '<symbol>' tests/test_litellm/
```

Map the affected surface to its suite using `tests/e2e/CLAUDE.md` (the suite folders section): LLM/provider translation -> `llm_translation/`, auth/permissions -> `access_control/`, fallbacks/cooldowns -> `router/`, budgets/rate limits/spend -> `quota_management/`, logging/guardrails -> `logging/` and `security/`, management routes -> `management/`, MCP -> `mcp/`. State the finding explicitly in the report: "covered by `<path::TestClass::test>`", or "no e2e coverage for this behavior".

Then add coverage so this specific bug can never recur silently:

- Prefer a `tests/e2e/` test that walks the behavior against a live proxy and real provider, following `tests/e2e/CONTRIBUTING.md` (CREATE -> CONFIGURE -> ACT -> SETTLE -> ASSERT recorded state -> ASSERT enforced behavior -> TEARDOWN) and `tests/e2e/CLAUDE.md` (shared transport only, typed pydantic bodies in `models.py`, `Result[R]` via `unwrap(...)`, `@pytest.mark.e2e`, and a `@pytest.mark.covers("...")` registry id). Never substitute a unit test for e2e feature coverage, and never add a mock/monkeypatch e2e test
- For a pure SDK/transformation bug with no product-facing proxy surface, add a focused regression test in the mapped `tests/test_litellm/` file (extend the existing mapped file for a bug fix; do not create a new one). Per the root `CLAUDE.md` the test must fail before the fix and pass after; verify by checking it out at the bad SHA (or reverting the fix) and watching it fail. Aim for a mutation-killing assertion, not a coverage-filler
- The test must encode the exact reproduction from Phase 1 so the regression is pinned forever

If you find a genuine product gap while writing the test, call it out in the PR rather than reshaping the test until it passes.

## Deliverable

Produce a single regression report (post it to the Slack thread and, if a fix PR is opened, put it in the PR description). Keep it high-signal and follow the public-facing writing rules in the root `CLAUDE.md` (no emojis, no em dashes, prose over lists where reasonable, no trailing periods on paragraphs). Structure:

```
Regression: <one-line symptom>
Repro: <exact curl/SDK command + wrong vs expected output>
Range: good <sha/version> -> bad <sha/version>
Offending PR: #<NNNNN> (<title>) by <author>, SHA <sha>
  what it changed that broke this: <one line, link the lines>

5 Whys:
  1 ... / 2 ... / 3 ... / 4 ... / 5 ...
  Root cause: <systemic cause>

e2e coverage: <"covered by <path>" | "no coverage; added <path::test> (covers=<id>)">
Immediate fix: <what unbreaks the user>  (link PR if opened)
```

Route the report to the area's DRI. Per the org's DRI router, ping the owner for the affected area by Slack ID (Rust -> Ishaan `<@U0773S394AD>`, LLM -> Mateo `<@U0ATSFX0VK2>`, UI -> Ryan `<@U0A2PA5KJJZ>`, Security -> Yuneng `<@U09NUN7L01K>`, MCP -> Tin `<@U0B1TDTHL4Q>`, Cloud/perf and Logging+Guardrails -> Yassin `<@U0AT0P5SY95>`). If no area clearly applies, determine the owner from git history of the touched files (`git shortlog -sne -- <paths>`) and confirm with Krrish.

## Checklist (do not report done until all are true)

- [ ] Deterministic reproduction captured (command + wrong output), on a named bad point
- [ ] Good and bad reference points established
- [ ] Offending PR identified with link, author, SHA, and the one-line "what broke"
- [ ] 5 Whys completed, grounded in linked evidence, reaching a systemic root cause and the coverage gap
- [ ] tests/e2e coverage question answered explicitly (covered where, or not covered)
- [ ] Regression test added (e2e preferred; unit in the mapped `tests/test_litellm/` file for SDK-only bugs) that fails before the fix and passes after
- [ ] Report posted and routed to the correct DRI
