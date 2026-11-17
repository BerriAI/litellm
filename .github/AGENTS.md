# CI

Integration groups and replica runs remain in `workflows/test-integration.yml`. Do not duplicate them in another workflow.

CI helpers used by existing workflows live in `scripts/ci/`. Jobs that need merge bases check out full history. Removing a CI job does not count its tests as covered. List tests that no remaining job invokes explicitly in `ci-coverage-allowlist.yml`.
