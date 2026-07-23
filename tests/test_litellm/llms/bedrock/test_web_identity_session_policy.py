"""
Regression for #30200.

``_auth_with_web_identity_token`` passes an inline ``Policy`` to
``sts.assume_role_with_web_identity``. In AWS IAM an STS session policy
acts as a PERMISSION CEILING — effective permissions are the
intersection of the role's identity policies and this policy, so any
action not listed here 403s on OIDC-auth requests only (static creds
and IRSA flow through different paths).

The original policy only granted ``bedrock:*`` actions. When
``#27678`` added the ``bedrock/claude_platform/<model>`` route, the
service-side action namespace was ``aws-external-anthropic:*``, not
``bedrock:*``, so every claude_platform call via OIDC silently denied
with::

    User: arn:aws:sts::ACCOUNT:assumed-role/...
    is not authorized to perform: aws-external-anthropic:CreateInference
    on resource: arn:aws:aws-external-anthropic:...
    because no session policy allows the
    aws-external-anthropic:CreateInference action

— even with a fully permissive identity policy.

Tests below intercept the kwargs handed to
``assume_role_with_web_identity``, parse the embedded ``Policy`` JSON,
and assert that both the original bedrock statement and the new
claude_platform statement are present and cover every documented
action.
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Actions the Claude Platform on AWS service is documented to call.
# Source: AWS IAM action reference + the #27678 surface area.
_CLAUDE_PLATFORM_ACTIONS = {
    "aws-external-anthropic:CreateInference",
    "aws-external-anthropic:CreateBatchInference",
    "aws-external-anthropic:CancelBatchInference",
    "aws-external-anthropic:DeleteBatchInference",
    "aws-external-anthropic:CountTokens",
    "aws-external-anthropic:Get*",
    "aws-external-anthropic:List*",
}


def _captured_policy() -> dict:
    """Run _auth_with_web_identity_token under mocks + return the parsed
    Policy dict that was actually sent to STS."""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    base = BaseAWSLLM()

    mock_sts = MagicMock()
    mock_sts.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": "k",
            "SecretAccessKey": "s",
            "SessionToken": "t",
            "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "PackedPolicySize": 0,
    }

    with (
        patch("boto3.client", return_value=mock_sts),
        patch(
            "litellm.llms.bedrock.base_aws_llm.get_secret",
            return_value="oidc-jwt-token",
        ),
    ):
        base._auth_with_web_identity_token(
            aws_web_identity_token="/path/to/token",
            aws_role_name="arn:aws:iam::123456789012:role/litellm-bedrock-role",
            aws_session_name="test-session",
            aws_region_name="us-east-1",
            aws_sts_endpoint=None,
        )

    mock_sts.assume_role_with_web_identity.assert_called_once()
    kwargs = mock_sts.assume_role_with_web_identity.call_args.kwargs
    policy_str = kwargs["Policy"]
    return json.loads(policy_str)


def _statement_by_sid(policy: dict, sid: str) -> dict:
    for stmt in policy["Statement"]:
        if stmt.get("Sid") == sid:
            return stmt
    raise AssertionError(
        f"Sid={sid!r} not found in session policy; "
        f"saw {[s.get('Sid') for s in policy['Statement']]}"
    )


class TestWebIdentitySessionPolicyShape:
    def test_policy_parses_as_valid_iam_document(self):
        policy = _captured_policy()
        assert policy["Version"] == "2012-10-17"
        assert isinstance(policy["Statement"], list)
        assert len(policy["Statement"]) >= 2

    def test_bedrock_statement_actions_preserved(self):
        """The original bedrock action set must still be granted —
        regression for the pre-existing bedrock/* routes."""
        policy = _captured_policy()
        bedrock_stmt = _statement_by_sid(policy, "BedrockLiteLLM")
        actions = set(bedrock_stmt["Action"])
        for required in (
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
        ):
            assert required in actions, f"{required} missing from BedrockLiteLLM"


class TestClaudePlatformActionsCovered:
    """The #30200 bug: every action in the claude_platform service
    namespace must appear in the session policy or OIDC requests 403."""

    @pytest.mark.parametrize("action", sorted(_CLAUDE_PLATFORM_ACTIONS))
    def test_claude_platform_action_present(self, action: str):
        policy = _captured_policy()
        # Action may live in any Statement — search across all.
        all_actions: set = set()
        for stmt in policy["Statement"]:
            stmt_actions = stmt.get("Action")
            if isinstance(stmt_actions, str):
                all_actions.add(stmt_actions)
            elif isinstance(stmt_actions, list):
                all_actions.update(stmt_actions)
        assert action in all_actions, (
            f"{action} missing from session policy — "
            f"bedrock/claude_platform/* requests will 403 on OIDC auth"
        )

    def test_claude_platform_statement_allows(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "ClaudePlatformLiteLLM")
        assert stmt["Effect"] == "Allow"
        assert stmt["Resource"] == "*"

    def test_no_aws_external_anthropic_statement_collision(self):
        """Don't accidentally grant a `*` action that would broaden the
        ceiling beyond what the documented actions require."""
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "ClaudePlatformLiteLLM")
        actions = stmt["Action"]
        if isinstance(actions, str):
            actions = [actions]
        assert "aws-external-anthropic:*" not in actions, (
            "session policy must not grant aws-external-anthropic:* — "
            "the ceiling should match the documented action set"
        )


class TestBedrockMantleActionsCovered:
    """LIT-3859: bedrock_mantle inference authorizes against the
    ``bedrock-mantle`` action namespace, so the session-policy ceiling
    must include it or every Mantle request via OIDC/WIF auth denies
    with "no session policy allows the bedrock-mantle:CreateInference
    action" even when the role's identity policy grants it."""

    def test_bedrock_mantle_create_inference_present(self):
        policy = _captured_policy()
        all_actions: set = set()
        for stmt in policy["Statement"]:
            stmt_actions = stmt.get("Action")
            if isinstance(stmt_actions, str):
                all_actions.add(stmt_actions)
            elif isinstance(stmt_actions, list):
                all_actions.update(stmt_actions)
        assert "bedrock-mantle:CreateInference" in all_actions, (
            "bedrock-mantle:CreateInference missing from session policy — "
            "bedrock_mantle/* requests will 403 on OIDC/WIF auth"
        )

    def test_bedrock_mantle_statement_allows(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockMantleLiteLLM")
        assert stmt["Effect"] == "Allow"
        assert stmt["Resource"] == "*"

    def test_no_bedrock_mantle_wildcard(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockMantleLiteLLM")
        actions = stmt["Action"]
        if isinstance(actions, str):
            actions = [actions]
        assert "bedrock-mantle:*" not in actions, (
            "session policy must not grant bedrock-mantle:* — "
            "the ceiling should match the documented action set"
        )

    def test_bedrock_mantle_statement_carries_secure_transport_condition(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockMantleLiteLLM")
        cond = stmt.get("Condition") or {}
        assert cond.get("Bool", {}).get("aws:SecureTransport") == "true", (
            "BedrockMantleLiteLLM must require aws:SecureTransport=true "
            "to keep parity with the bedrock statement"
        )


class TestBedrockCountTokensCovered:
    """#33142: ``BaseAWSLLM._auth_with_web_identity_token`` omits
    ``bedrock:CountTokens`` from the inline STS session policy, so OIDC
    auth paths 403 on the Bedrock count-tokens handler even when the
    assumed role's identity policy explicitly allows the action. STS
    session policies are a permission ceiling; missing entries silently
    down-grade effective permissions.

    Source-level mirror of the ``TestBedrockMantleActionsCovered``
    block above."""

    def test_bedrock_count_tokens_present_in_session_policy(self):
        policy = _captured_policy()
        all_actions: set = set()
        for stmt in policy["Statement"]:
            stmt_actions = stmt.get("Action")
            if isinstance(stmt_actions, str):
                all_actions.add(stmt_actions)
            elif isinstance(stmt_actions, list):
                all_actions.update(stmt_actions)
        assert "bedrock:CountTokens" in all_actions, (
            "bedrock:CountTokens missing from session policy; "
            "bedrock/count_tokens/* requests via OIDC auth will 403 "
            "even when the IAM role allows the action (#33142)"
        )

    def test_bedrock_count_tokens_lives_on_bedrock_statement(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockLiteLLM")
        actions = stmt["Action"]
        if isinstance(actions, str):
            actions = [actions]
        assert "bedrock:CountTokens" in actions, (
            "bedrock:CountTokens should be granted by the existing "
            "BedrockLiteLLM statement, not a separate ad-hoc one"
        )

    def test_no_bedrock_wildcard_added(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockLiteLLM")
        actions = stmt["Action"]
        if isinstance(actions, str):
            actions = [actions]
        assert "bedrock:*" not in actions, (
            "session policy must not grant bedrock:*; the ceiling "
            "should match the documented action set"
        )


def _make_jwt(payload: dict) -> str:
    def _segment(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{_segment({'alg': 'RS256', 'typ': 'JWT'})}.{_segment(payload)}.signature"


class TestInvalidIdentityTokenSurfacesAudience:
    """LIT-4026: when STS rejects the web identity token with
    ``InvalidIdentityToken`` (the "Incorrect token audience" case), the raised
    error must name the ``aud``/``iss`` the token actually carries so an
    operator can diagnose the mismatch without enabling LITELLM_LOG=DEBUG on a
    prod instance."""

    _AUD = "https://guidepoint.litellm-prod.ai"
    _ISS = "https://accounts.google.com"
    _STS_MESSAGE = (
        "An error occurred (InvalidIdentityToken) when calling the "
        "AssumeRoleWithWebIdentity operation: Incorrect token audience"
    )

    def _raise_invalid_identity_token(self) -> Exception:
        from litellm.llms.bedrock.base_aws_llm import AwsAuthError, BaseAWSLLM

        token = _make_jwt({"aud": self._AUD, "iss": self._ISS, "sub": "svc-account"})

        mock_sts = MagicMock()

        class _InvalidIdentityTokenException(Exception):
            pass

        mock_sts.exceptions.InvalidIdentityTokenException = (
            _InvalidIdentityTokenException
        )
        mock_sts.assume_role_with_web_identity.side_effect = (
            _InvalidIdentityTokenException(self._STS_MESSAGE)
        )

        with (
            patch("boto3.client", return_value=mock_sts),
            patch(
                "litellm.llms.bedrock.base_aws_llm.get_secret",
                return_value=token,
            ),
            pytest.raises(AwsAuthError) as exc_info,
        ):
            BaseAWSLLM()._auth_with_web_identity_token(
                aws_web_identity_token="oidc/google/" + self._AUD,
                aws_role_name="arn:aws:iam::123456789012:role/litellm-bedrock-role",
                aws_session_name="test-session",
                aws_region_name="us-east-1",
                aws_sts_endpoint=None,
            )
        return exc_info.value

    def test_error_names_token_audience(self):
        err = self._raise_invalid_identity_token()
        assert self._AUD in str(err)

    def test_error_names_token_issuer(self):
        err = self._raise_invalid_identity_token()
        assert self._ISS in str(err)

    def test_error_preserves_original_sts_reason(self):
        err = self._raise_invalid_identity_token()
        assert "Incorrect token audience" in str(err)

    def test_error_is_401(self):
        err = self._raise_invalid_identity_token()
        assert err.status_code == 401


class TestPolicyTransportConditions:
    def test_bedrock_statement_keeps_secure_transport_condition(self):
        policy = _captured_policy()
        bedrock_stmt = _statement_by_sid(policy, "BedrockLiteLLM")
        cond = bedrock_stmt.get("Condition") or {}
        assert cond.get("Bool", {}).get("aws:SecureTransport") == "true"

    def test_claude_platform_statement_carries_secure_transport_condition(self):
        """The new statement should match the existing one's hardening
        posture — TLS-only, same as bedrock."""
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "ClaudePlatformLiteLLM")
        cond = stmt.get("Condition") or {}
        assert cond.get("Bool", {}).get("aws:SecureTransport") == "true", (
            "ClaudePlatformLiteLLM must require aws:SecureTransport=true "
            "to keep parity with the bedrock statement"
        )
