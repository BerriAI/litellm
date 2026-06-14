<!--
👋 Hi there — please read before submitting.

To keep the review queue healthy for everyone, **every external PR is
auto-triaged** by an LLM bot ("Agent Shin") on open / reopen, regardless of
whether the PR is a draft or marked ready for review. PRs that don't meet
the rubric below are auto-closed with an explanation.

To pass triage, your PR must satisfy AT LEAST ONE of:

  (A) Link a related GitHub issue (e.g. "Fixes #1234" or "Resolves
      https://github.com/BerriAI/litellm/issues/1234"), OR

  (B) Provide ALL of the following IN THIS PR DESCRIPTION:
      - A clear problem description (what bug or missing feature this addresses)
      - Expected vs. actual behavior
      - Visual QA proof (before/after screenshots, screen recording, or
        terminal output demonstrating that the fix/feature works end-to-end)

Every external PR (including drafts, regardless of age) also receives a
Greptile code review. Any PR with a Greptile Confidence Score below 4/5 is
auto-closed.

If your PR was auto-closed and you've addressed the feedback, you have two
options to bring it back:

  - **Open a new PR** with the updated branch (recommended — GitHub does not
    let external contributors reopen a PR that was closed by a bot or
    maintainer).
  - **Or** comment `@agent-shin reconsider` on the closed PR. Agent Shin
    will re-run triage and reopen the PR if it now meets the bar.

Internal BerriAI contributors are exempt from this auto-triage — fill in the
Linear ticket section instead.
-->

## Relevant issues

<!-- e.g. "Fixes #000". If you have no related issue, fill in the
"Problem description / Expected vs. Actual / QA proof" sections below. -->

## Linear ticket

<!-- INTERNAL CONTRIBUTORS ONLY: add the Linear ticket e.g. "Resolves LIT-1234"
to magically link the Linear ticket to the GitHub PR. External contributors:
leave this blank and fill in the problem/expected-actual/QA sections below. -->

## Problem description

<!-- What bug or missing feature does this PR address? One or two paragraphs.
External contributors: required unless you linked a GitHub issue above. -->

## Expected vs. actual behavior

<!-- What did you expect to happen? What is happening today (before this PR)?
External contributors: required unless you linked a GitHub issue above. -->

## QA proof

<!-- Required for external contributors: include before/after screenshots,
a screen recording, or terminal/log output that demonstrates the fix or feature
works end-to-end. For UI changes, before/after screenshots are mandatory.
For backend changes, terminal output of a passing test or curl command is fine. -->

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [ ] I have added meaningful tests
- [ ] My PR passes all unit tests on [`make test-unit`](https://docs.litellm.ai/docs/extras/contributing_code)
- [ ] My PR's scope is as isolated as possible; it only solves 1 specific problem
- [ ] I have received a Greptile **Confidence Score of at least 4/5** before requesting a maintainer review (Greptile reviews automatically on open; comment `@greptileai` to re-trigger after pushing fixes)

## Delays in PR merge?

If you're seeing a delay in your PR being merged, ping the LiteLLM Team on [Slack (#pr-review)](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA).

## CI (LiteLLM team)

> **CI status guideline:**
>
> - 50-55 passing tests: main is stable with minor issues.
> - 45-49 passing tests: acceptable but needs attention
> - <= 40 passing tests: unstable; be careful with your merges and assess the risk.

- [ ] **Branch creation CI run**  
       Link:

- [ ] **CI run for the last commit**  
       Link:

- [ ] **Merge / cherry-pick CI run**  
       Links:

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
