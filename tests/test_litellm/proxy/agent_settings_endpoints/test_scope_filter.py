"""
Unit tests for `partition_secrets_for_session` (LIT-2891 validation #3).

The scope filter is the single source of truth for "which secrets get pushed
into a hydrated agent session". Both the UI display path and the B2 hydrate
path call into here, so the access-control behavior must NEVER drift.

Critical paths covered:
1. `scope == "all"` matches every session.
2. List scope matches when session.repos intersects the list.
3. List scope does NOT match when there's no intersection — including the
   particularly subtle case where the session repo is from the same org
   but a different repo (e.g. `BerriAI/other` vs scoped `BerriAI/litellm`).
4. Empty session repos with a list scope = no match (don't accidentally
   leak when the session forgot to declare repos).
5. Repo URL normalization: `https://github.com/BerriAI/litellm.git` and
   `BerriAI/litellm` and the dict shapes all match.
"""

import pytest

from litellm.proxy.agent_settings_endpoints.scope_filter import (
    normalize_repos,
    partition_secrets_for_session,
    secret_in_scope,
)


class TestSecretInScopeAll:
    """`scope='all'` is the easy case but worth pinning."""

    def test_all_scope_matches_with_empty_repos(self):
        assert secret_in_scope("all", []) is True

    def test_all_scope_matches_with_repos(self):
        assert secret_in_scope("all", ["BerriAI/litellm"]) is True

    def test_all_scope_matches_with_dict_repos(self):
        assert secret_in_scope("all", [{"full_name": "BerriAI/litellm"}]) is True


class TestSecretInScopeList:
    """The validation #3 core: per-repo scope must isolate."""

    def test_list_scope_matches_intersecting_repo(self):
        assert secret_in_scope(["BerriAI/litellm"], ["BerriAI/litellm"]) is True

    def test_list_scope_rejects_different_repo_same_org(self):
        # This is the LIT-2891 validation #3 case: a secret scoped to
        # BerriAI/litellm must NOT appear in a hydrate payload for
        # BerriAI/other-repo, even though they're the same org.
        assert secret_in_scope(["BerriAI/litellm"], ["BerriAI/other"]) is False

    def test_list_scope_rejects_different_org(self):
        assert secret_in_scope(["BerriAI/litellm"], ["openai/openai-python"]) is False

    def test_list_scope_with_empty_session_repos_does_not_match(self):
        # Defense-in-depth: a session that forgot to declare repos must
        # NOT inherit list-scoped secrets.
        assert secret_in_scope(["BerriAI/litellm"], []) is False

    def test_empty_list_scope_never_matches(self):
        # `scope=[]` is treated as "no repos" — explicit safe default.
        assert secret_in_scope([], ["BerriAI/litellm"]) is False

    def test_multi_repo_scope_matches_any(self):
        scope = ["BerriAI/litellm", "BerriAI/litellm-docs"]
        assert secret_in_scope(scope, ["BerriAI/litellm-docs"]) is True
        assert secret_in_scope(scope, ["BerriAI/other"]) is False


class TestRepoNormalization:
    """Different repo reference shapes must canonicalize to the same form."""

    def test_plain_owner_name(self):
        assert normalize_repos(["BerriAI/litellm"]) == ["berriai/litellm"]

    def test_url_with_https_scheme(self):
        assert normalize_repos(["https://github.com/BerriAI/litellm"]) == [
            "berriai/litellm"
        ]

    def test_url_with_dot_git_suffix(self):
        assert normalize_repos(["https://github.com/BerriAI/litellm.git"]) == [
            "berriai/litellm"
        ]

    def test_dict_with_full_name(self):
        assert normalize_repos([{"full_name": "BerriAI/litellm"}]) == [
            "berriai/litellm"
        ]

    def test_dict_with_url(self):
        assert normalize_repos([{"url": "https://github.com/BerriAI/litellm.git"}]) == [
            "berriai/litellm"
        ]

    def test_case_insensitive(self):
        # Same repo with different casing — should dedupe to one.
        result = normalize_repos(
            ["BerriAI/LiteLLM", "berriai/litellm", "BERRIAI/LITELLM"]
        )
        assert result == ["berriai/litellm"]

    def test_unparseable_returns_none(self):
        # Garbage in, no entries out — must never crash the hydrate path.
        assert normalize_repos(["not-a-repo"]) == []
        assert normalize_repos([None]) == []
        assert normalize_repos([42]) == []


class TestPartitionSecretsForSession:
    """The public entrypoint used by B2 hydrate. Output ordering matters
    for the audit log so we pin it explicitly."""

    def test_partitions_in_and_out_of_scope(self):
        secrets = [
            ("DATABASE_URL", ["BerriAI/litellm"]),
            ("OPENAI_API_KEY", "all"),
            ("INTERNAL_TOKEN", ["BerriAI/internal"]),
        ]
        in_scope, out_of_scope = partition_secrets_for_session(
            secrets, ["BerriAI/litellm"]
        )
        assert in_scope == ["DATABASE_URL", "OPENAI_API_KEY"]
        assert out_of_scope == ["INTERNAL_TOKEN"]

    def test_preserves_input_order(self):
        # Audit logs read in order, so the partition has to keep order.
        secrets = [
            ("Z_SECRET", "all"),
            ("A_SECRET", "all"),
            ("M_SECRET", ["BerriAI/litellm"]),
        ]
        in_scope, _ = partition_secrets_for_session(secrets, ["BerriAI/litellm"])
        assert in_scope == ["Z_SECRET", "A_SECRET", "M_SECRET"]

    def test_empty_secrets_returns_empty(self):
        assert partition_secrets_for_session([], ["BerriAI/litellm"]) == (
            [],
            [],
        )


# Make pytest pick the file up without an explicit `pytest_plugins`.
@pytest.fixture(autouse=True)
def _no_op_fixture():
    yield
