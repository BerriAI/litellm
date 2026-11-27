# CI

Integration groups and replica runs remain in `workflows/test-integration.yml`. Do not duplicate them in another workflow.

CI helpers used by existing workflows live in `scripts/ci/`. Jobs that need merge bases check out full history. Removing a CI job does not count its tests as covered. List tests that no remaining job invokes explicitly in `ci-coverage-allowlist.yml`.

A new code-quality check that runs without importing litellm goes in the `code-quality` job in `workflows/test-linting.yml`. One that imports litellm goes in `lint-runtime` in `workflows/test-unit.yml`, which installs the shared Rust bridge wheel. Neither job compiles Rust, which keeps both inside the ten-minute pull request budget.

Pull request jobs run on `ubuntu-latest`, because larger runners are billed even on this public repository. When a job outgrows the budget, cut its setup and pytest startup cost or its slowest tests before adding a shard. Every extra shard pays that fixed cost again and takes another runner.
