<!--
PR title must follow Conventional Commits 1.0.0:

    <type>(<optional scope>): <description>

Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
Use `!` after the type/scope for breaking changes (e.g. feat(api)!: drop /v1/foo).

Examples:
    feat(mcp): add oauth2 authorization flow
    fix(ui-teams): refresh table on member change
    refactor(ui-playground): extract message renderer

A GitHub Actions check enforces this format. See:
    https://www.conventionalcommits.org/en/v1.0.0/
    CONTRIBUTING.md → Commit and PR Conventions
-->

## Relevant issues

<!-- e.g. "Fixes #000" -->

## Linear ticket

<!-- if you are an internal contributor, add the Linear ticket e.g. "Resolves LIT-1234" to magically link the Linear ticket to the GitHub PR -->

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [ ] I have Added testing in the [`tests/test_litellm/`](https://github.com/BerriAI/litellm/tree/main/tests/test_litellm) directory, **Adding at least 1 test is a hard requirement** - [see details](https://docs.litellm.ai/docs/extras/contributing_code)
- [ ] My PR passes all unit tests on [`make test-unit`](https://docs.litellm.ai/docs/extras/contributing_code)
- [ ] My PR's scope is as isolated as possible, it only solves 1 specific problem
- [ ] I have requested a Greptile review by commenting `@greptileai` and received a **Confidence Score of at least 4/5** before requesting a maintainer review

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

## Screenshots / Proof of Fix

<!-- Include screenshots, screen recordings, or log output demonstrating that your changes work as expected.
     For bug fixes: show reproduction before the fix and passing behavior after.
     For new features: show the feature working end-to-end.
     For UI changes: include before/after screenshots. -->

## Type

<!-- Should match the type used in the PR title (Conventional Commits). Keep only the relevant one(s). -->

- `feat` — New feature
- `fix` — Bug fix
- `refactor` — Refactoring (no behavior change)
- `docs` — Documentation
- `perf` — Performance improvement
- `test` — Tests only
- `build` — Build system / dependencies
- `ci` — CI / GitHub Actions
- `chore` — Maintenance / release / tooling
- `revert` — Revert a previous commit
- `style` — Formatting / whitespace

## Changes
