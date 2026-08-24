"""Live e2e: Bedrock routes reached over AssumeRoleWithWebIdentity.

When the proxy authenticates to Bedrock with a web identity token it attaches
an inline STS session policy, and that policy is a permission ceiling:
effective permissions are the intersection with the assumed role's identity
policy, so a route whose IAM action the ceiling omits returns 403 no matter how
permissive the role is. Static keys and ambient IRSA resolve credentials
through other code paths that never attach the policy, which is why every other
Bedrock test in this suite stays green while these routes are broken.

That omission has shipped repeatedly: #30200 (claude_platform), Bedrock Mantle,
and #33142 / #37336 (`/count-tokens`). Each test below drives one route that
403s when its action is missing from the ceiling.

Deselected unless E2E_BEDROCK_OIDC is set, because reaching this path needs an
IAM OIDC provider trusting the cluster's issuer and a role granting the Bedrock
actions. See the PR's QA runbook for the provisioning steps.
"""

from __future__ import annotations

import os

import pytest
from e2e_config import (
    BEDROCK_OIDC_DEFAULT_TOKEN,
    BEDROCK_OIDC_ROLE_ARN_ENV,
    BEDROCK_OIDC_TOKEN_ENV,
    unique_marker,
)
from e2e_http import require_successful_call, unwrap
from endpoints_client import EndpointsClient, RerankResult
from lifecycle import ResourceManager
from models import CountTokensBody, ChatMessage, LiteLLMParamsBody

pytestmark = [pytest.mark.e2e, pytest.mark.bedrock_oidc]

CHAT_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
RERANK_MODEL = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"

DOCUMENTS = [
    "Carson City is the capital city of the American state of Nevada.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since before it was a country.",
]
QUERY = "What is the capital of the United States?"


def _require_role_arn() -> str:
    """Hard-fail rather than skip: the opt-in env is set, so the operator
    asserted this infrastructure exists."""
    role_arn = os.environ.get(BEDROCK_OIDC_ROLE_ARN_ENV)
    assert role_arn, (
        f"{BEDROCK_OIDC_ROLE_ARN_ENV} must name the IAM role to assume when "
        f"the {BEDROCK_OIDC_TOKEN_ENV} web identity token is presented."
    )
    return role_arn


def _web_identity_params(model: str) -> LiteLLMParamsBody:
    """A deployment the proxy must authenticate via AssumeRoleWithWebIdentity.

    All three of token, role, and session name are required together; drop any
    one and credential resolution silently falls through to a path that never
    applies the session policy, so the test would pass without proving anything.
    """
    return LiteLLMParamsBody(
        model=model,
        aws_web_identity_token=os.environ.get(BEDROCK_OIDC_TOKEN_ENV, BEDROCK_OIDC_DEFAULT_TOKEN),
        aws_role_name=_require_role_arn(),
        aws_session_name=f"e2e-oidc-{unique_marker()}",
        aws_region_name="os.environ/AWS_REGION",
    )


class TestBedrockWebIdentitySessionPolicy:
    """Each route here 403s when the session policy omits its IAM action."""

    def test_count_tokens_is_authorized(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        """Regression for #33142 / #37336: needs bedrock:CountTokens."""
        model = f"e2e-oidc-count-tokens-{unique_marker()}"
        model_id = endpoints_client.create_model(model, _web_identity_params(CHAT_MODEL))
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.proxy.count_tokens(
            key,
            CountTokensBody(model=model, messages=[ChatMessage(role="user", content="hello")]),
        )
        counted = unwrap(result)
        assert counted.input_tokens > 0, f"count_tokens returned {counted.input_tokens} tokens"

    def test_rerank_is_authorized(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        """Needs bedrock:Rerank, which the ceiling omitted entirely."""
        model = f"e2e-oidc-rerank-{unique_marker()}"
        model_id = endpoints_client.create_model(model, _web_identity_params(RERANK_MODEL))
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.rerank(key, model, QUERY, DOCUMENTS, top_n=2)
        require_successful_call(result)
        parsed = RerankResult.model_validate_json(result.body)
        assert parsed.results, f"/rerank returned no results: {result.body[:300]}"
