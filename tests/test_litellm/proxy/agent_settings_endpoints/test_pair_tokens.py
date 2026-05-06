"""
Unit tests for the pair-token + worker-JWT helpers (LIT-2891 validation #5).

These helpers underpin the "Add Machine" install flow. The invariants we
care about are:

* The raw token is high-entropy and unique per call.
* Only the sha256 is intended to be persisted; the helper returns both so
  the caller can do the right thing, but the digest is what lands in the DB.
* Hashing is deterministic — same raw token always hashes to the same
  digest, so the consume-side lookup works.
* Expiry is honored, with naive datetimes treated as UTC (matches the
  proxy's existing convention for DB-stored timestamps).
* The install command renders exactly the format the docs/UX promise
  (`curl ... | sh -s -- --proxy ... --token ...`). Tests pin this so
  changing the install host doesn't silently break the on-box install.
"""

from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.agent_settings_endpoints.pair_tokens import (
    build_install_command,
    hash_pair_token,
    hash_worker_jwt,
    is_expired,
    issue_pair_token,
)


class TestIssuePairToken:
    def test_returns_distinct_raw_tokens_per_call(self):
        a = issue_pair_token()
        b = issue_pair_token()
        assert a.raw_token != b.raw_token
        assert a.token_hash != b.token_hash

    def test_token_hash_matches_raw(self):
        issued = issue_pair_token()
        assert issued.token_hash == hash_pair_token(issued.raw_token)

    def test_default_expiry_is_in_the_future(self):
        issued = issue_pair_token()
        now = datetime.now(timezone.utc)
        assert issued.expires_at > now
        # Should be ~15 minutes; allow generous slack so test isn't flaky.
        assert issued.expires_at - now <= timedelta(minutes=20)

    def test_custom_ttl_respected(self):
        issued = issue_pair_token(ttl_minutes=1)
        delta = issued.expires_at - datetime.now(timezone.utc)
        assert delta <= timedelta(minutes=2)
        assert delta >= timedelta(seconds=30)

    def test_raw_token_has_meaningful_entropy(self):
        # 32 bytes urlsafe-base64 → 43 chars unpadded; we just need to
        # confirm the helper isn't accidentally returning a short string.
        issued = issue_pair_token()
        assert len(issued.raw_token) >= 32


class TestHashHelpers:
    def test_pair_token_hash_is_deterministic(self):
        token = "abc123"
        assert hash_pair_token(token) == hash_pair_token(token)

    def test_pair_token_hash_distinguishes_inputs(self):
        assert hash_pair_token("a") != hash_pair_token("b")

    def test_worker_jwt_hash_uses_same_algo(self):
        # Both helpers should be sha256 hex; identical input → identical
        # output. (Not strictly required by the spec but a useful sanity
        # check that we didn't accidentally swap the algorithm in one.)
        assert hash_pair_token("zzz") == hash_worker_jwt("zzz")

    def test_hashes_are_64_hex_chars(self):
        digest = hash_pair_token("anything")
        assert len(digest) == 64
        int(digest, 16)  # valid hex; raises if not


class TestIsExpired:
    def test_future_is_not_expired(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert is_expired(future) is False

    def test_past_is_expired(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert is_expired(past) is True

    def test_none_is_treated_as_expired(self):
        # Defensive default — missing expiry must never be treated as
        # "valid forever".
        assert is_expired(None) is True

    def test_naive_datetime_treated_as_utc(self):
        # Matches the proxy DB convention; naive timestamps come back from
        # SQLite without tzinfo.
        future_naive = datetime.utcnow() + timedelta(
            minutes=5
        )  # noqa: DTZ003 — intentionally naive for the test
        assert is_expired(future_naive) is False

    def test_explicit_now_argument_used(self):
        expires = datetime.now(timezone.utc) + timedelta(minutes=1)
        far_future = expires + timedelta(hours=1)
        assert is_expired(expires, now=far_future) is True


class TestBuildInstallCommand:
    def test_default_renders_expected_one_liner(self):
        cmd = build_install_command(
            proxy_url="https://proxy.example.com", raw_token="TOK"
        )
        # All values pass through shlex.quote — simple alnum/`:/.-` strings
        # come back unquoted, matching the documented one-liner.
        assert cmd == (
            "curl -fsS https://litellm.ai/install-worker | sh -s -- "
            "--proxy https://proxy.example.com --token TOK"
        )

    def test_install_url_override(self):
        cmd = build_install_command(
            proxy_url="https://p",
            raw_token="T",
            install_script_url="https://internal.example/install.sh",
        )
        assert "https://internal.example/install.sh" in cmd
        assert "--proxy https://p" in cmd
        assert "--token T" in cmd

    def test_proxy_url_with_metacharacters_is_shell_quoted(self):
        # Defense against header-injection / config bugs that might land a
        # space, semicolon, or quote in the proxy URL. shlex.quote wraps the
        # value in single quotes so it can't break out of the install line.
        cmd = build_install_command(proxy_url="https://h; rm -rf /", raw_token="TOK")
        assert "'https://h; rm -rf /'" in cmd
        # Sanity: the dangerous payload must NOT appear unquoted.
        assert "--proxy https://h; rm -rf /" not in cmd

    def test_raw_token_with_metacharacters_is_shell_quoted(self):
        cmd = build_install_command(proxy_url="https://p", raw_token="abc def$(whoami)")
        assert "'abc def$(whoami)'" in cmd

    def test_install_script_url_is_shell_quoted(self):
        cmd = build_install_command(
            proxy_url="https://p",
            raw_token="T",
            install_script_url="https://hosts space.example/install.sh",
        )
        assert "'https://hosts space.example/install.sh'" in cmd


@pytest.fixture(autouse=True)
def _no_op_fixture():
    yield
