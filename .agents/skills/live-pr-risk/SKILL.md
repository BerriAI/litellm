---
name: live-pr-risk
description: For a litellm PR or working branch, build the full code-dependency graph of every changed symbol (callers, subclass overrides, string-dispatch and registry lookups, kwargs pass-through, response fields, config keys, DB columns, dashboard and SDK consumers), find which of those dependent paths NO test exercises, then prove what happens on them by A/B-ing a real proxy at the base and at the head with real Postgres and real provider APIs and zero mocks. Flags regressions, backward-incompatible changes, and breaking-change risk on the untested paths. Missing credentials are requested from the user, never mocked or silently skipped. Triggers on "live test this PR", "what breaks in this PR", "what depends on this function", "regression check", "backward compat check", "blast radius", "untested callers", "is this PR safe to ship", "run live-pr-risk".
version: 2.0.0
---

Ported from BerriAI/litellm-internal-skills. Slash commands this skill mentions but does not define (`/team`, `/code-review`, `/devin-pr`) live in that repo; the phases below stand on their own without them

Answers one question: what else uses the thing you changed, and what happens to it that nobody has ever tested?

A change to a function, a variable, a default, or a response field is safe on the path you were thinking about. The risk lives in the paths you were not: a caller three modules away, a subclass overriding the method you re-signed, a registry that reaches your function by string name, a dashboard reading a field you renamed. Those paths usually have no test, which is exactly why the change looks green.

So this skill does three things in order: build the dependency graph of the change, subtract the parts that tests already cover, and then drive whatever remains on a real proxy so the answer is observed rather than reasoned.

### When to use

The user points at a PR or the current branch and wants the blast radius: "what breaks", "what depends on this", "regression check this", "is this safe to ship", "who calls this and did we test it". Also correct before shipping a bundled or stacked PR where many changes land at once.

Not a code review. `/code-review` judges the diff against house rules; this judges what the diff does to everything downstream of it.

Also invoked from `/team` phase 5. That phase does the symbol trace on paper. Call this skill when the traced dependents need to be exercised rather than reasoned about, which is any time a dependent path is reachable through the proxy and has no test.

Read `references/rig.md` for the runnable commands. Below is the operating model.

### Operating principles

* A dependent path with no test is the deliverable. Tested dependents are already covered by CI; untested ones are where the regression ships. Rank the whole run by that.
* grep finds the easy half. The dependents that actually bite are the ones a text search misses: string dispatch, registries, `**kwargs` forwarding, subclass overrides, JSON field names read by the dashboard. Phase 3 exists for those specifically.
* No baseline, no regression claim. A symptom on the head that also reproduces at the merge-base is a pre-existing bug. Every scenario runs on both sides.
* Zero mocks, zero stubs, zero recorded fixtures. Real proxy process, real Postgres, real upstream calls. A leg that cannot run for real is reported unverified, never faked.
* The client response is where the comparison starts, not where it ends. A real provider or vendor answers the request and swallows it, so the one thing a real destination cannot show you is what litellm sent it. Every outbound boundary the changed path crosses (provider deployment, guardrail vendor, OTLP endpoint, Langfuse, webhook) gets a forwarding recorder in front of the real destination on both sides, and the recorded request is diffed base vs head: key set, values, and how many times it was sent. A recorder that logs and forwards is a probe at a boundary, not a mock; the real destination still answers. Added 2026-09-17 after a gauntlet over eight merged PRs found six regressions on surfaces the rigs never captured (a fallback replaying an unredacted request, guardrail vendors receiving the whole conversation, a provider hit three times behind an unchanged status) while every PR body attested the live A/B had passed.
* Real destinations too, not just real upstreams. When the change affects what gets exported (spans, logs, events), point the proxy at the real backend and read the result back through that vendor's own API; a local OTLP collector proves the wire format, not that the destination stored or renders it. Same rule for the deliverable: ship screenshots of the vendor UI, not a pasted attribute dump. Credentials are usually already in the main checkout's `.env`.
* Ask for credentials; never invent one, never quietly drop a leg.
* Drive the rig yourself; fan out the graph walk. The proxy is stateful and flaky to delegate. Spawn parallel agents for the reference-kind sweeps in phase 3, one per kind, and keep the rig in the foreground.
* Reason to a hypothesis, then go observe it. "This caller would now get `None`" is a lead. The finding is the curl that shows it.
* Score every leg on the whole diff of observables, never on the field the PR was written for. "Lookup by the new id returns 200 after and 404 before" proves the feature landed and says nothing about what the old id was displaced by, who else read it, or what else moved in the same object. A leg passes when the set-diff of everything captured (status, body, headers, recorded outbound requests, attempt count, spend row, emitted trace or log object) contains only the differences the PR names.

### Phases (track with TaskCreate)

### 1. Resolve the target and its baseline

With a PR: take the live head, never a stale local branch or a reconstructed short SHA.

```bash
gh pr view <N> --json number,headRefName,baseRefName,headRefOid,changedFiles
```

Without a PR (the `/team` case, branch not yet pushed): head is your working tree, base is `git merge-base origin/main HEAD` (main is the default branch again since 2026-09-13). Commit or copy the tree aside first so the base-side worktree is reproducible; never `git stash`.

Note merge-ref drift: what ships is the PR merged into current `main`. If `main` moved since the branch point, re-run the top scenarios on a merged tree at the end, and when that merge conflicts, report the conflict and say the merge-ref check did not run.

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
* Config-selected families that consume a shared input generically. When the change alters an input every member of a family receives (the mapping handed to every guardrail hook, the kwargs handed to every callback, the header dict handed to every pass-through route, the field every trace exporter reads), grep for the changed key finds only the members that already special-cased it. The members that regress are the ones that serialize the whole mapping into their vendor payload without naming any key, and those never appear in the grep. Enumerate the family from its registry or dispatch table (`litellm/proxy/guardrails/guardrail_registry.py`, `litellm/litellm_core_utils/custom_logger_registry.py`, the pass-through route table, the trace exporter list), and every member that builds its output from the shared mapping goes on the target list under its own config cell. Pinning the grep hits with regression tests and driving one custom echo implementation live covers the members you already knew about and none of the others; that is how three vendor integrations started receiving the full request conversation on 2026-09-17 with a passing live A/B in the PR body.
* Snapshots replayed after hooks ran. When a diff captures a copy of request state to replay later (a fallback, a retry, a hedged or mirrored request), list every mutation that runs between the capture point and the replay, because the replay undoes all of them. Sort those mutations the way the in-place-mutation bullet above does, and drive the replay with a real deciding hook attached (a redacting pre-call guardrail on the primary deployment is the canonical one) plus a recorder on the replay destination. Meeting one such reader and special-casing it (a key-metadata flag the first pass wrote) is the tell that the class was never enumerated; a fallback that replayed the unredacted request shipped exactly that way.
* Sibling mechanisms for the same policy. A helper that processes whatever carrier it is handed has as many entrypoints as there are producers of that carrier: `forward_headers` and `x-pass-*` both feed the outbound header dict, caller metadata and proxy-injected key or team metadata both land in the same `metadata` mapping, body fields and their header equivalents both reach the same resolver. Enumerate the producers of the carrier, not the callers of the helper, and run the changed scenario once per producer. Proving `traceparent` handling under one producer and inferring the other is how the sibling dropped the caller's `tracestate`.
* Each downstream consumer's own precedence chain. When a change sets a field that several exporters or loggers read (a trace id, a request id, a session id), each consumer resolves it through its own precedence chain over its own controls, and your new value now outranks or displaces something in that chain. For every consumer, send that consumer's own controls (`metadata.session_id`, `trace_name`, `tags`, `x-litellm-*` headers) with the change active and diff the emitted object base vs head. A value that used to be caller-stable and is now per-request is Backward incompatible even when the consumer's grouping still works.
* Docs. User docs live in the sibling `litellm-docs` checkout, not in the code repo. A code-repo grep of `docs/` returns nothing and reads as "undocumented" for something documented for months.

Fan these out to parallel agents, one kind each, and reconcile. Output is one list: every dependent path, with its entrypoint (proxy route, SDK call, background job, dashboard page, CLI command) and which changed item it depends on.

### 4. Subtract what tests already cover, the gap is the target list

For each dependent path, decide whether any test exercises it through your change. A test that covers the caller while stubbing your function does not count.

```bash
uv run --no-sync pytest tests/test_litellm/<mapped paths> \
  --cov=litellm/<dependent modules> --cov-report=term-missing
```

Cross-check by name: `grep -rn "<dependent_function>" tests/`. Coverage that never touches the line, or a dependent with no test naming it at all, goes on the target list.

Then the tests that reach the change without naming any symbol in it. The legacy suites under `tests/llm_translation/` and `tests/local_testing/` drive provider behavior through the public entrypoints (`litellm.completion(model="<provider>/...")`, `get_supported_openai_params(..., custom_llm_provider="<provider>")`, `litellm.get_model_info`) and assert what the provider used to do, so a symbol grep never finds them and a behavior change leaves them asserting the old contract. Find them by the provider or config string instead: `grep -rln '"<provider>/\|custom_llm_provider="<provider>"' tests/llm_translation tests/local_testing`, and run every hit on the head. A change in a shared layer (exception mapping, the httpx handler, the router, cost or param utils) has no provider string to grep for, so grep these suites for the behavior it reclassifies instead: the exception classes it maps (`pytest.raises(litellm.APIConnectionError)` and `except litellm.APIConnectionError` alike), the response field, the config key, and run every hit on the head. These suites are CircleCI jobs, and CircleCI only runs on a PR carrying the `run-ci` label whose head branch matches `.circleci/config.yml`'s branch filter (`main` or `/litellm_.*/`: every job in the one workflow carries that filter), so a PR without the label, or one whose head is a `devin/`, fork, or other non-`litellm_` branch, shows all-green GitHub Actions while never having executed them (the label on a non-matching head creates a pipeline with zero workflows, which is what PR #38693 showed on 2026-08-29 and the user had to point out). A PR on a non-matching head is moved before anything else in this section: push its tip to a `litellm_`-prefixed branch on origin with the commits and authors intact, open the replacement PR through /devin-pr with the full body plus a first line naming the original PR and head sha, label it, close the original as superseded with a short pointer comment, and carry the ticket on the new vehicle, moving the run itself onto it: re-point or rebuild the worktree on the replacement branch, push every later commit there, and run the endgame only on the replacement PR, never another push, approval, or merge on the closed original (the internal-copy route in memory fork_push_recipe, precedent #37048 for #36633); the first run is on `main` after the merge, where the failure lands on whoever triages CI next. So label every PR that reaches these suites (`gh pr edit <n> --add-label run-ci`), wait for the CircleCI workflow at the tip, cycling the label off and back on when an already-labeled tip never produces one (the label event is what creates the pipeline, so a cycle kicks it off within seconds; the user had to do it by hand on #38818, ruling 2026-09-01), and read its failures as findings; the CHECKED line is never written while that workflow is missing or red. One run (#38265) passed its own CI, `/qa`, and this skill, then turned staging red on `tests/llm_translation/test_together_ai.py`, which asserted the exact behavior the PR changed; #38318 (2026-08-26) changed which exception a status-less transport error maps to, carried no label, and `tests/llm_translation/test_a2a.py` turned staging red two days later. A stale assertion in these suites is a finding to fix in the same PR, not a flake to wave off, and a test that fails because its input lacks the trigger the change keys on is a regression, never a stale assertion (test-expectation-guard blocks rewriting it without a recorded reason).

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
* Capture the outbound wire on both sides. Point every `api_base` the changed path reaches (provider deployment, guardrail vendor, OTLP or Langfuse endpoint) at a forwarding recorder that logs the request and forwards it to the real destination, and diff the recorded requests base vs head: key set, values, byte size, and count. Count is a first-class observable; a leg whose status matched with three recorded upstream attempts against one is a regression, and with a real provider that difference is invisible except as a bill. Recorder commands are in `references/rig.md` section 6.
* Hostile-input cells for every header or body field the diff starts reading, on an unauthenticated request. Send it as an int, a list, an empty string, a 5 KB string, and twice with the same value, then read back the row, span, or log line the request wrote on both sides. "Existing behavior on chat, now on a wider surface" is a widening, and parity with a pre-existing weakness does not make it safe; the leg still runs.
* When a step now runs once per item where it ran once per request (per choice, per content part, per streaming scan interval), run the multi-item fixture through the failure verdict as well as the happy path, diff the aggregate output (error detail, violation list, emitted spans) against base, and count the calls. Per-item processing that concatenates its verdicts duplicates them, and the cost caveat "one extra call per item is small" is a number to measure, not a sentence to write.
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

A caveat is not a disposition. When a Medium or Low caveat in the PR body describes a change a client, an operator, or a downstream vendor can observe (a status that now differs, a header no longer forwarded verbatim, an id a caller now controls, a payload a vendor now receives, a value that was stable and is now per-request), it belongs in Breaking or Backward incompatible above, and each such entry ends with either the commit that fixed it or the name of the person who approved shipping it and where they said so. The line "ran /live-pr-risk and found no regressions/backward incompatible risks" may not appear in a body whose caveats describe one; write the five sections instead. Ruling from the 2026-09-17 gauntlet retrospective: four of its eight headline findings were already written as caveats a few lines above that attestation, and all four merged without a recorded decision.

### Gotchas

* The dangerous dependent is the one grep cannot see. If phase 3 produced only call sites, it is not finished. Registries, overrides, kwargs chains, and JSON field consumers are where the shipped regressions come from.
* A rename looks safest and is not. Direct callers fail loudly at import; string-dispatch lookups fail silently at runtime, and only for the config that uses them.
* Code added after the graph walk needs its own graph walk. Fixes land mid-run, usually answering a review comment, and the reflex is to A/B the new commit on the scenarios already standing. That re-tests the dependents you were already watching and says nothing about the dependents of the new symbol. If phase 2 would have classified the addition as a new side effect, a new mutation, or a new raises, rerun phase 3 for it instead of inheriting the previous target list. A late addition is the most likely thing in the diff to ship unexamined, precisely because the rig is already built and green.
* A verdict of "safe" that came from reading rather than running is a hypothesis. Agents return SAFE and BREAKS in the same confident register, and a SAFE reached by tracing call order is exactly the kind that a running proxy overturns. Re-read the evidence line in the report: if it says "traced" or "inferred" for anything touching enforcement, ordering, or shared state, that item still belongs on the target list.
* Green CI means the tested paths pass. It says nothing about the untested dependents, which is the entire subject here.
* The rig is built from the same mental model that produced the fix, and so are its caveats. The scenarios you choose are the ones the diff made you think about, which is the complement of where the regression lives. Have an agent that has read the diff and the registries but not the PR description or your scenario list produce the target list independently, reconcile the two, and hand the rig config plus the target list to `/code-review` along with the diff so a reviewer can say "your rig has no guardrail" on a PR that changes what guardrails see.
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
