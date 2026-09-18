---
name: live-pr-risk
description: For a litellm PR or working branch, build the full code-dependency graph of every changed symbol (callers, subclass overrides, string-dispatch and registry lookups, kwargs pass-through, response fields, config keys, DB columns, dashboard and SDK consumers), find which of those dependent paths NO test exercises, then prove what happens on them by A/B-ing a real proxy at the base and at the head with real Postgres and real provider APIs and zero mocks. Flags regressions, backward-incompatible changes, and breaking-change risk on the untested paths. Missing credentials are requested from the user, never mocked or silently skipped. Triggers on "live test this PR", "what breaks in this PR", "what depends on this function", "regression check", "backward compat check", "blast radius", "untested callers", "is this PR safe to ship", "run live-pr-risk".
version: 2.0.0
---

Answers one question: what else uses the thing you changed, and what happens to it that nobody has ever tested?

A change to a function, a variable, a default, or a response field is safe on the path you were thinking about. The risk lives in the paths you were not: a caller three modules away, a subclass overriding the method you re-signed, a registry that reaches your function by string name, a dashboard reading a field you renamed. Those paths usually have no test, which is exactly why the change looks green.

So this skill does three things in order: build the dependency graph of the change, subtract the parts that tests already cover, and then drive whatever remains on a real proxy so the answer is observed rather than reasoned.

### When to use

The user points at a PR or the current branch and wants the blast radius: "what breaks", "what depends on this", "regression check this", "is this safe to ship", "who calls this and did we test it". Also correct before shipping a bundled or stacked PR where many changes land at once.

Not a code review. `/code-review` judges the diff against house rules; this judges what the diff does to everything downstream of it.

This skill is also the PR gate for this repo: a PreToolUse hook (`.claude/hooks/live_pr_risk_gate.py`) blocks PR creation until a passing run is recorded for the exact HEAD being shipped. See phase 8.

Read `references/rig.md` for the runnable commands. Below is the operating model.

### Operating principles

* A dependent path with no test is the deliverable. Tested dependents are already covered by CI; untested ones are where the regression ships. Rank the whole run by that.
* grep finds the easy half. The dependents that actually bite are the ones a text search misses: string dispatch, registries, `**kwargs` forwarding, subclass overrides, JSON field names read by the dashboard. Phase 3 exists for those specifically.
* No baseline, no regression claim. A symptom on the head that also reproduces at the merge-base is a pre-existing bug. Every scenario runs on both sides.
* Zero mocks, zero stubs, zero recorded fixtures. Real proxy process, real Postgres, real upstream calls. A leg that cannot run for real is reported unverified, never faked.
* Real destinations too, not just real upstreams. When the change affects what gets exported (spans, logs, events), point the proxy at the real backend and read the result back through that vendor's own API; a local OTLP collector proves the wire format, not that the destination stored or renders it. Same rule for the deliverable: ship screenshots of the vendor UI, not a pasted attribute dump. Credentials are usually already in the main checkout's `.env`.
* Ask for credentials; never invent one, never quietly drop a leg.
* Drive the rig yourself; fan out the graph walk. The proxy is stateful and flaky to delegate. Spawn parallel agents for the reference-kind sweeps in phase 3, one per kind, and keep the rig in the foreground.
* Reason to a hypothesis, then go observe it. "This caller would now get `None`" is a lead. The finding is the curl that shows it.

### Phases (track with TaskCreate)

### 1. Resolve the target and its baseline

With a PR: take the live head, never a stale local branch or a reconstructed short SHA.

```bash
gh pr view <N> --json number,headRefName,baseRefName,headRefOid,changedFiles
```

Without a PR (branch not yet pushed): head is your working tree, base is `git merge-base origin/litellm_internal_staging HEAD`. Commit or copy the tree aside first so the base-side worktree is reproducible; never `git stash`.

Note merge-ref drift: what ships is the PR merged into current staging. If staging moved since the branch point, re-run the top scenarios on a merged tree at the end, and when that merge conflicts, report the conflict and say the merge-ref check did not run.

### 2. Extract the changed surface, and how each item changed

Not just which symbols moved, but what about them moved, because the "what" determines which dependents are at risk.

```bash
git diff <base>...<head> -U0 | grep -E '^[-+]\s*(async def |def |class |[A-Z_]+ *=)'
git diff <base>...<head> --stat
```

For every changed function, method, class, module-level constant, Pydantic field, config key, env var, and DB column, record the delta in these terms:

* signature: params added, removed, reordered, renamed, retyped, defaults changed
* return: shape, type, field names, nullability, ordering
* raises: new exception, changed exception class, an error now swallowed
* defaults: a changed default breaks every caller who never passed the value
* side effects: writes, cache invalidation, callbacks fired, DB rows, ordering of any of these
* timing/async: sync becoming async, blocking becoming deferred, a `create_task` added
* name: a rename is a dependency break for every non-grep-able reference kind in phase 3

A symbol whose body changed but whose contract did not still matters if its side effects or timing changed. Say which of the above applies; "changed" alone is not enough to target the graph walk.

### 3. Build the dependency graph, both directions

Downstream (dependents, who relies on this): these break when the contract changes. Upstream (dependencies, what this now calls, or calls differently): these break when you feed them values or contexts they never saw.

Start with the cheap sweep, over `litellm/`, `enterprise/`, `tests/`, `ui/litellm-dashboard/`, and `litellm/proxy/client/`:

```bash
grep -rn "<symbol>" litellm/ enterprise/ tests/ ui/ --include='*.py' --include='*.ts' --include='*.tsx' --include='*.yaml' --include='*.md'
```

Then the reference kinds that sweep will not find. Each gets its own pass, and in this codebase each has bitten someone:

* String dispatch and registries. Functions reached by name through a map or a config string: provider dispatch on `custom_llm_provider`, callback and logging-integration names in `litellm.callbacks` / `success_callback`, guardrail names, router strategy names. Grep the string literal, not the symbol, and grep `litellm/constants.py` for the key.
* Subclass overrides and duck-typed implementers. Any class overriding a method you re-signed, and any class satisfying the same informal protocol without inheriting. `grep -rn "def <method_name>"` across the repo, not just the class you edited.
* `**kwargs` pass-through. Signature changes vanish through forwarding layers. Follow every hop that forwards `**kwargs` into your function and check what the top of that chain actually passes.
* Pydantic/TypedDict field names as wire contract. A renamed or retyped field surfaces in HTTP responses. Consumers: `ui/litellm-dashboard` (grep the field name in `.ts`/`.tsx`), `litellm/proxy/client/`, and the OpenAPI-derived `schema.d.ts`.
* Config keys and env vars. `config.yaml` semantics, `general_settings`, `litellm_settings`, `os.environ` reads. Same name with new meaning is worse than a rename because nothing errors.
* DB columns and Prisma schema. A field read back by a different code path, a migration that is not reversible, a composite key matched on one column.
* Sentinel and placeholder values a writer will accept back. When a diff starts masking or defaulting a value in a response (`***REDACTED***`, `"unknown"`, `0`, `null`), the risk is not the read, it is the echo: some client will GET the object and POST it back, and the write path usually has no rule distinguishing the placeholder from a real value. The tell is a response field whose new value is a plausible instance of its own type. Check every writer that accepts that field, not just the endpoint that changed, and check whether the placeholder survives validation and encryption on the way in. In one run a masked credential round-tripped into a second team and was stored (and encrypted at rest) as the literal marker, silently killing that team's logging with no error anywhere. Grep the marker literal across the repo and see whether any writer branches on it; if some endpoints restore-on-marker and yours does not, that asymmetry is the finding.
* In-place mutation of shared state, with readers ordered by when they run. When a diff starts mutating an object that outlives the function, a request/context dict, a cached config, anything passed by reference down a pipeline, enumerating readers is not enough, because the same reader is unaffected before the write and silently changed after it. Sort them by execution order relative to your write, then split them by what they do with the value: readers that report it (logs, metrics, traces, spend rows) versus readers that decide with it (guardrails, policy and quota checks, scanners, auth, routing). A wrong value reaching a reporter is a cosmetic bug; a wrong value reaching a decider is a silent policy failure, and it will not raise. The shape to fear is a setting in one concern quietly changing behavior in another, a logging or privacy toggle that alters what a security check inspects. Never settle this by reading call order or trusting a nearby comment that asserts the two are independent; both are routinely stale. Drive the decider.
* Values classified by their shape rather than their name. When a diff decides what something is by inspecting its runtime shape (detail is a dict vs a string, a field is a list vs a scalar, a payload has key X), the dependents are every site that PRODUCES that shape, and grep for the consuming symbol will not find one of them. Enumerate the producers instead: `grep -n "<ExceptionClass>("`, every `return`/`raise` of the classified type in the module and its bases, and check each against the predicate. In one run "a dict detail means policy block" held for every raise site but one, where an unparseable-response handler raised a 500 detailing a dict, so a service outage was delivered as HTTP 200 while both sibling endpoints returned 500. The tell is a predicate over a structural property of a value that several unrelated code paths construct. Prefer a discriminator over a field the producers must set deliberately (a status code, an explicit tag) and verify by listing producers, not callers.
* Docs. User docs live in the sibling `litellm-docs` checkout, not in the code repo. A code-repo grep of `docs/` returns nothing and reads as "undocumented" for something documented for months.

Fan these out to parallel agents, one kind each, and reconcile. Output is one list: every dependent path, with its entrypoint (proxy route, SDK call, background job, dashboard page, CLI command) and which changed item it depends on.

### 4. Subtract what tests already cover, the gap is the target list

For each dependent path, decide whether any test exercises it through your change. A test that covers the caller while stubbing your function does not count.

```bash
uv run --no-sync pytest tests/test_litellm/<mapped paths> \
  --cov=litellm/<dependent modules> --cov-report=term-missing
```

Cross-check by name: `grep -rn "<dependent_function>" tests/`. Coverage that never touches the line, or a dependent with no test naming it at all, goes on the target list.

Rank the target list by reachability: paths reachable through the proxy's HTTP surface first (those are what a user hits), then SDK entrypoints, then background jobs and CLI, then anything reachable only in enterprise config. State the ranking in the report so the reader sees what was prioritized and what was left.

### 5. Enumerate credentials, then ask for what is missing

Build the credential list from the target list, check the main checkout's `.env` and the shell env, and probe each key with a minimal direct call before building on it, since a billing-dead key fails every model and looks like a proxy-side 429 from inside the rig.

For anything missing or dead, stop and ask the user, naming the exact variable, what it unlocks, and which targets go untested without it. Never generate a placeholder, never substitute a mock, never silently drop a target. Declined or unavailable credentials put those targets in the report's "Not verified" section, individually named.

Two env traps: `AWS_BEARER_TOKEN_BEDROCK` overrides SigV4 and splits your AWS identity, so unset it in any SigV4 rig; premium-gated surfaces need `LITELLM_LICENSE` sourced from the main checkout's `.env` rather than a patched `is_premium`.

### 6. Drive the target list live, base vs head

Two worktrees, two venvs, two namespaced Postgres containers, two proxy ports, same config, same scenarios, same order. Commands in `references/rig.md`. Non-negotiables:

* Boot both sides with at least 2 uvicorn workers (`--num_workers 2`), the multi-pod Kubernetes shape customers actually run, and when a changed path involves per-process in-memory state, run each side as two proxy processes sharing one database and Redis with requests alternating across them. Same topology on both sides, named in the report.
* `export PYTHONPATH=<worktree-root>` and assert the loaded `litellm.__file__` sits inside it. A worktree-launched proxy otherwise imports the main checkout and the A/B compares a tree against itself.
* Namespace every shared resource off a per-run token; a hardcoded `:4000` may be someone else's proxy.
* Each scenario names the dependent path it is exercising, so the report maps finding to graph edge.
* Wire a consumer of every hook kind into the rig, not just the kind you changed. A rig that only attaches the extension kind you were thinking about cannot observe what the other kinds receive, so those paths drop off the target list despite being reachable through the same entrypoint. When a hook's behavior is set by a constructor argument or an instance attribute rather than a config key, it is unreachable from config alone; load a minimal real implementation through the documented extension point and have it record what it was handed. That is a real code path rather than a mock, and it is often the only way to see the hook's input at all.
* Capture status, body, latency, and the spend row on both sides. Identical bodies still differ in side effects.
* For a dependent path with no HTTP surface (a background job, an internal hook), drive it through the smallest real entrypoint that reaches it in a running proxy rather than importing it in isolation; an in-process call proves the function, not the wiring.

Where the two sides differ, root-cause to a specific commit before writing it up. `git bisect` across the PR's own commits is cheap when the head is a bundle.

### 7. Report

Lead with the verdict, then group:

* Breaking: a dependent path that previously worked now fails or answers differently. Give both observed results and the graph edge that explains it.
* Backward incompatible: a contract changed (signature, field, default, config meaning, exception class, schema) even where current callers survive.
* Regression risk: an untested dependent this run could not reach, with why it is risky and what it would take to exercise.
* Dependency graph: every dependent found, marked tested / untested / verified-live / unreachable, so the reader can see the sweep was complete rather than lucky.
* Not verified: every target skipped for missing credentials or an unreachable environment, named individually. This section existing is honest; omitting it is worse than a long one.

Every finding states its evidence. Never present a reasoned inference as an observation.

### 8. Record the verdict for the PR gate

A run passes only when the report contains zero Breaking findings, and every "Not verified" entry exists because a credential was declined or an environment was unreachable, never because a reachable target was skipped. Backward-incompatible findings do not block, but each one must be intentional and called out in the PR body.

On a pass, record it for the head that was tested:

```bash
printf 'PASS %s\n' "$(git rev-parse HEAD)" > "$(git rev-parse --path-format=absolute --git-path live-pr-risk-pass)"
```

The marker binds to that exact commit, so any new commit stales it and PR creation blocks again until this skill re-runs. Never write the marker without a passing run; fix the breaking findings and re-run instead.

### Gotchas

* The dangerous dependent is the one grep cannot see. If phase 3 produced only call sites, it is not finished. Registries, overrides, kwargs chains, and JSON field consumers are where the shipped regressions come from.
* A rename looks safest and is not. Direct callers fail loudly at import; string-dispatch lookups fail silently at runtime, and only for the config that uses them.
* Code added after the graph walk needs its own graph walk. Fixes land mid-run, usually answering a review comment, and the reflex is to A/B the new commit on the scenarios already standing. That re-tests the dependents you were already watching and says nothing about the dependents of the new symbol. If phase 2 would have classified the addition as a new side effect, a new mutation, or a new raises, rerun phase 3 for it instead of inheriting the previous target list. A late addition is the most likely thing in the diff to ship unexamined, precisely because the rig is already built and green.
* A verdict of "safe" that came from reading rather than running is a hypothesis. Agents return SAFE and BREAKS in the same confident register, and a SAFE reached by tracing call order is exactly the kind that a running proxy overturns. Re-read the evidence line in the report: if it says "traced" or "inferred" for anything touching enforcement, ordering, or shared state, that item still belongs on the target list.
* Green CI means the tested paths pass. It says nothing about the untested dependents, which is the entire subject here.
* A scout reading files while you edit can report phantom findings. If an agent claims a symbol is missing, check `git show HEAD:<file>` before reacting; it may have read between two of your edits.
* When an agent calls something "the established pattern", count it before you repeat it. Agents describe two call sites and a precedent in the same confident register. Before recommending that a change follow "the house convention", run `grep -rln <marker>` to count the files and `git log -S <symbol> --format='%ad %an'` to see how many authors and how recently. In one run a "house pattern with tests" turned out to be two endpoints touched by one author the previous month, which flipped the recommendation from "follow the convention" to "do not add a third bespoke implementation". The reviewer who challenges this will be right, and relaying an agent's framing unchecked is how a triage report acquires an error the code never had.
* Both sides must prove which tree they ran. Print `pwd` and grep for a head-only symbol inside the same command as the test. Two runs reporting identical results mean nothing if both ran the same tree, which has happened via unquoted shell variables and a missing `cd`.
* Fresh worktrees lack gitignored files, so no `.env`, no `node_modules`, no synced `.venv`. One-shot: `uv sync --inexact --frozen --group proxy-dev --extra proxy --extra extra_proxy` at the worktree root, plus `npm ci` in `ui/litellm-dashboard`.
* The dashboard is part of the surface. If a changed field reaches `ui/`, load `http://127.0.0.1:<port>/ui` and click the page. A white screen is a breaking change no curl will find.
* Never `git stash`, including inside throwaway diagnostic one-liners; a concurrent pop elsewhere can drop the entry.
* Do not tear down a rig you did not build. Idle ports do not mean another session finished.
* Secondary, cheap, worth one command: if the diff also moves `uv.lock`, `pyproject.toml`, `package*.json`, `Dockerfile*`, or `schema.prisma`, a package bump changes runtime behavior with no symbol to trace. Diff the resolved sets (`uv pip freeze` per side, not the lockfile text) and fold anything high-risk into the target list. This is a footnote to the code-dependency work, not a substitute for it.

### How to maintain this skill

When a run turns up a reference kind phase 3 did not tell you to look for, add it to that list with the tell that identifies it. When a dependent path turns out to have been untested and to have broken, note the shape so the next run predicts it. Keep everything general across PRs; anything true of only one PR belongs in that PR's report.
