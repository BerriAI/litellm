## Relevant issues

<!-- e.g., "Fixes #000" -->

## Linear ticket

<!-- if you are an internal contributor, add "Resolves " followed by the Linear ticket e.g., "Resolves LIT-1234" to link the Linear ticket to the GitHub PR. If you don't have one, leave the section blank rather than guessing -->

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [ ] I have added meaningful tests
- [ ] My PR passes all CI/CD checks (e.g., lint, format, unit tests)
- [ ] My PR's scope is as isolated as possible; it only solves 1 specific problem
- [ ] I have received a Greptile **Confidence Score of at least 4/5** before requesting a maintainer review (Greptile reviews automatically once the PR is opened; only comment `@greptileai` to re-request a review after pushing changes)

## Delays in PR merge?

If you're seeing a delay in your PR being merged, ping the LiteLLM Team on [Slack (#pr-review)](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA).

## Screenshots / Proof of Fix

<!-- Include screenshots, screen recordings, or command (e.g., curl) + output demonstrating that your changes work as expected
     The proof must be completely e2e with no mocks, using, for example, actual LLM calls costing real $. `pytest` commands are not enough
     For bug fixes: show reproduction before the fix and passing behavior after
     Include the commit hash each proof was captured at, for both the before and the after runs
     For new features: show the feature working end-to-end
     For UI changes: include before/after screenshots -->

## Type

<!-- Select the type of Pull Request -->
<!-- Keep only the necessary ones -->

🆕 New Feature
🐛 Bug Fix
🧹 Refactoring
📖 Documentation
🚄 Infrastructure
✅ Test

## Changes

## QA runbook

<!-- This QA runbook and the Final Attestation below are only needed when your PR edits tests/e2e; delete both sections otherwise

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

### Final Attestation

<!-- Part of the QA runbook above; keep this only if you kept that section (your PR edits tests/e2e), otherwise delete this Final Attestation too -->

- [ ] The tests check the right things, including the edge cases, and regressions in the respective real-world customer use-cases are not possible after this PR
