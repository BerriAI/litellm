<!-- The whole description's target audience is humans, not AI agents: write it in plain, simple,
     everyday engineering language, extremely parsable and readable at a glance. This goes double for
     the TLDR, User Flow, and Caveats sections -->

## TLDR

<!-- Fill in the bullets below and keep each one short and concrete: one line per bullet, roughly 10 words max -->

Problem this solves:

- <blah>
- ...

How it solves it:

- <blah>
- ...

## User Flow

<!-- Two ordered lists, Before and After, walking the same end user through the same task, written strictly from that user's seat
     Read the linked issue, ticket, or customer thread first so the flow reflects the real application and the routes its users actually hit; don't invent a generic scenario
     Lead each list with one plain sentence saying where the flow fails (Before) or succeeds (After), then number the steps
     Every step is something the user does or observes: the HTTP method and full URL they hit, what they sent, and what visibly came back (status code, error text, the shape of an ID). UI steps name the page URL and what is on screen
     No LiteLLM internals: never name functions, files, DB tables, config classes, hooks, callbacks, or code paths. "The upload hands back an ID that looks like OpenAI's own `file-abc123` instead of the scrambled one the gateway returned" is right, "no managed-file row was registered" is wrong
     Keep the two lists step-for-step identical until they diverge, so the changed step is obvious
     If the bug had a security or authorization consequence, end each list with what another user could or could no longer do
     Regenerate this section whenever new commits change the PR's behavior, so it never describes an older revision

Example:

Before: a developer whose app streams chat completions gets no token counts back, so their cost dashboard reads zero

1. They send POST https://litellm-domain/v1/chat/completions with `"stream": true` and no `stream_options`
2. The last SSE chunk arrives with `"usage": null`, so their app records 0 prompt and 0 completion tokens
3. They open https://litellm-domain/ui/?page=logs and see the request logged at $0 spend

After: the same request comes back with real token counts, so the dashboard shows real spend

1. The proxy admin sets `always_include_stream_usage: true` and restarts the proxy
2. The developer sends the same POST https://litellm-domain/v1/chat/completions with `"stream": true` and no `stream_options`
3. The last SSE chunk now carries a `usage` object with real prompt and completion token counts
4. https://litellm-domain/ui/?page=logs shows that request at non-zero spend
-->

## Relevant issues

<!-- e.g., "Fixes #000" -->

## Linear ticket

<!-- if you are an internal contributor, add "Resolves " followed by the Linear ticket e.g., "Resolves LIT-1234" to link the Linear ticket to the GitHub PR. If you don't have one, leave the section blank rather than guessing -->

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [ ] I have added meaningful tests
- [ ] The handful of test files covering my change pass locally, e.g. `uv run pytest tests/test_litellm/<your_test_file>.py -v`. Leave the suites (`make test-unit-*`, `make test-unit`) to CI: it finishes in ~15 minutes where a laptop takes an hour or more
- [ ] My PR passes all required CI/CD checks (e.g., lint, schema.d.ts sync check, etc.)
- [ ] My PR's scope is as isolated as possible; it only solves 1 specific problem
- [ ] I have received a Greptile **Confidence Score of at least 4/5** before requesting a maintainer review (Greptile reviews automatically once the PR is opened; only comment `@greptileai` to re-request a review after pushing changes)

## Delays in PR merge?

If you're seeing a delay in your PR being merged, ping the LiteLLM Team on [Slack (#pr-review)](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA).

## Screenshots / Proof of Fix

<!-- Include screenshots, screen recordings, or command (e.g., curl) + output demonstrating that your changes work as expected
     The proof must be completely e2e with no mocks, using actual LLM calls costing real $$$ if applicable. `pytest` commands are not enough
     Show ONLY the latest run: capture Before at the merge base and After at the PR's current tip, and when new commits change behavior, replace this whole section with the fresh run instead of stacking it on top of older ones. The run must be up to date. As soon as a new commit is made and it makes this PR description's after sha stale (it's no longer tip of PR), you must re-run the QA
     Structure the section exactly as below: Before and After one heading level below this section, each naming the commit hash it was captured at, one lower-level heading per case inside each, the same case names in the same order on both sides, and numbered steps (command, observed output) under every case, never loose prose; shared setup (config, payloads) goes above Before, and with a single case, drop the case headings and number the steps directly

### Before (<hash>)

#### <case 1>

1. ...
2. ...

#### <case 2>

1. ...

### After (<hash>)

#### <case 1>

1. ...
2. ...

#### <case 2>

1. ...

     For bug fixes: Before shows the reproduction, After shows the same steps passing
     For new features: Before shows the capability missing, After shows it working end-to-end
     If the change applies to all three LLM endpoints (/v1/responses, /v1/chat/completions, /v1/messages), make each endpoint its own case, not just one
     For UI changes: before/after screenshots under the same headings -->

## Type

<!-- Select the type of Pull Request -->
<!-- Keep only the necessary ones -->

🆕 New Feature
🐛 Bug Fix
🧹 Refactoring
📖 Documentation
🚄 Infrastructure
✅ Test

## Caveats (if any)

<!-- Group caveats under severity subheadings (### Severe, ### High, ### Medium, ### Low), with
     short bullet points inside each, just like the TLDR: one line per bullet, roughly 10 words max
     Call out known limitations, follow-up work, or anything a reviewer should watch out for
     Include only the tiers that have caveats; drop the empty ones
     - Severe: inherent to what the PR deliberately ships, there even when the code works as intended:
       it can degrade or take down a running deployment (e.g. a slow or table-locking boot migration),
       rewrite data by design, break an existing workflow on purpose, or change auth behavior. An
       operator must plan around it before rollout
     - High: an unintended hole: a correctness, security, data-loss, or backward-compatibility bug,
       unsafe to ship as is
     - Medium: a real gap someone can hit, but with a workaround or a narrow blast radius
     - Low: anything else worth noting: naming, cleanup, an edge case nobody hits
     Nest bullets as deep as helps: hierarchy beats one long line when it makes things clearer to a
     human reader
     Leave this section empty if there are none -->

## QA runbook

<!-- Only needed when your PR edits tests/e2e; delete this section otherwise

For each e2e test you added or changed, list the manual steps a reviewer can follow to reproduce it by hand against a live proxy, mapping 1:1 to what the test asserts: one top-level bullet per test giving its pytest node id followed by what it proves in plain words, then a nested "- [ ]" checklist where each item is a concrete action (route, request body, expected response) and the final item is the sanity-check step shown in the examples. Note environment prerequisites (provider credentials, config flags) and any nuances a manual run will hit. See PRs #32914 and #32963 for full examples

Example checklists:

- tests/e2e/quota_management/ratelimit/test_rate_limit_e2e.py::TestKeyRateLimits::test_rpm_limit_blocks_over_limit - a key allowed 2 requests a minute serves exactly 2 and refuses the 3rd
  - [ ] Generate a limited key: curl -X POST http://localhost:4000/key/generate -H "Authorization: Bearer sk-1234" -d '{"rpm_limit": 2}'
  - [ ] Send three /v1/chat/completions requests with that key inside one minute
  - [ ] Expect the first two to return 200 and the third to return 429 naming the rpm limit
  - [ ] Sanity check: this test makes sense to add and is not hand-wavey (e.g., assert actual expected spend instead of just spend > 0) or potentially flaky

- tests/e2e/management/test_management_e2e.py::TestModelRoutes::test_model_create_appears_in_ui - a deployment created through the API shows up on the Admin UI models page
  - [ ] POST /model/new with the master key, a bedrock model, and aws_region_name (needs STORE_MODEL_IN_DB=True and AWS credentials)
  - [ ] Open http://localhost:4000/ui/?page=models and expect a deployment row showing the returned model id
  - [ ] Sanity check: this test makes sense to add and is not hand-wavey (e.g., assert actual expected spend instead of just spend > 0) or potentially flaky
-->

## Final Attestation

- [ ] The tests check the right things, including the edge cases, and regressions in the respective real-world customer use-cases are not possible after this PR
