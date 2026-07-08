# Rate Limits E2E Suite

Put rate-limit enforcement tests here when the primary behavior under test is a
limit being applied, reset, or bypassed across keys, teams, models, tags, or
endpoint families.

If the primary behavior is provider translation or an LLM endpoint contract, keep
the test under `tests/e2e/llm_translation/` instead.

Coverage for this module is declared directly on tests with
`@pytest.mark.e2e_coverage(...)`. Use `module="rate_limits"`, the exercised
endpoint, `provider="proxy"`, and params such as `key_rpm_limit`,
`team_tpm_limit`, or `reset_window`.
