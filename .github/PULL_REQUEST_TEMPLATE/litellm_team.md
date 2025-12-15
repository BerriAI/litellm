## Title

<!-- e.g. "Implement user authentication feature" -->

## Relevant issues

<!-- e.g. "Fixes #000" -->

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [ ] I have Added testing in the [`tests/litellm/`](https://github.com/BerriAI/litellm/tree/main/tests/litellm) directory, **Adding at least 1 test is a hard requirement** - [see details](https://docs.litellm.ai/docs/extras/contributing_code)
- [ ] I have added a screenshot of my new test passing locally
- [ ] My PR passes all unit tests on [`make test-unit`](https://docs.litellm.ai/docs/extras/contributing_code)
- [ ] My PR's scope is as isolated as possible, it only solves 1 specific problem

## CI (LiteLLM team)

> 1. Don't merge if CI has insufficient funds issues; ping @AlexsanderHamir until they’re resolved.
> 2. Split your work into small, logical commits and push each one to trigger CI, making it easier **for you** to identify where something broke by comparing against your branch-creation and previous CI runs.
> 3. When merging your PR, choose `Merge & Squash` for convenience.
>
> **CI status guideline:**
>
> - >= 50 passing tests: main is stable
> - >= 45-49 passing tests: acceptable but needs attention
> - <= 40 passing tests: unstable; be careful with your merges and assess the risk.

- [ ] **Branch creation CI run**  
       Link:

- [ ] **CI run for the last commit**  
       Link:

- [ ] **Merges / cherry-picks into this branch**  
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
