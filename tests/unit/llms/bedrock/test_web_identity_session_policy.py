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
import itertools
import json
from collections.abc import Callable
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Final, TypeAlias
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.session import get_session
from pydantic import TypeAdapter

from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.sqs import SQSLogger
from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
from litellm.llms.bedrock.model_listing import BedrockModelLister
from litellm.llms.bedrock.passthrough.transformation import BedrockAgentRuntimeAWS
from litellm.llms.bedrock.realtime.handler import BedrockRealtime
from litellm.llms.bedrock.web_identity_session_policy import WebIdentitySessionPolicy, _SessionPolicyStatement
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.opensearch.vector_stores.transformation import OpenSearchVectorStoreConfig
from litellm.llms.s3_vectors.vector_stores.transformation import S3VectorsVectorStoreConfig
from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig
from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.proxy.common_utils.load_config_utils import s3_object_reader
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import _is_bedrock_agent_runtime_route
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.comprehend_medical_passthrough_logging_handler import (
    COMPREHEND_MEDICAL_SUPPORTED_OPERATIONS,
    ComprehendMedicalAWS,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.transcribe_passthrough_logging_handler import (
    TranscribeAWS,
    transcribe_supported_operations,
)
from litellm.rag.ingestion.bedrock_ingestion import BedrockRAGIngestion
from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.router import GenericLiteLLMParams

_STS_SESSION_EXPIRATION: Final = datetime(2099, 1, 1, tzinfo=timezone.utc)

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


def _captured_policy_document() -> str:
    """Run _auth_with_web_identity_token under mocks + return the Policy
    JSON document that was actually sent to STS."""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    base = BaseAWSLLM()

    mock_sts = MagicMock()
    mock_sts.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": "k",
            "SecretAccessKey": "s",
            "SessionToken": "t",
            "Expiration": _STS_SESSION_EXPIRATION,
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
    return kwargs["Policy"]


_SESSION_POLICY_ADAPTER: Final = TypeAdapter(WebIdentitySessionPolicy)


def _captured_policy() -> WebIdentitySessionPolicy:
    return _SESSION_POLICY_ADAPTER.validate_python(json.loads(_captured_policy_document()))


def _granted_actions(policy: WebIdentitySessionPolicy) -> frozenset[str]:
    return frozenset(itertools.chain.from_iterable(stmt["Action"] for stmt in policy["Statement"]))


def _invoke_model_grants_over_any_transport(policy: WebIdentitySessionPolicy) -> tuple[_SessionPolicyStatement, ...]:
    return tuple(
        stmt for stmt in policy["Statement"] if "bedrock:InvokeModel" in stmt["Action"] and "Condition" not in stmt
    )


def _statement_by_sid(policy: WebIdentitySessionPolicy, sid: str) -> _SessionPolicyStatement:
    for stmt in policy["Statement"]:
        if stmt.get("Sid") == sid:
            return stmt
    raise AssertionError(f"Sid={sid!r} not found in session policy; saw {[s.get('Sid') for s in policy['Statement']]}")


class TestWebIdentitySessionPolicyShape:
    def test_policy_parses_as_valid_iam_document(self):
        policy = _captured_policy()
        assert policy["Version"] == "2012-10-17"
        assert len(policy["Statement"]) >= 2

    def test_bedrock_statement_actions_preserved(self):
        """The original bedrock action set must still be granted —
        regression for the pre-existing bedrock/* routes."""
        granted: Final = _granted_actions(_captured_policy())
        for required in (
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
        ):
            assert required in granted, f"{required} missing from the Bedrock session policy"

    def test_bedrock_count_tokens_action_present(self):
        """Regression for #33142: the CountTokens handler authorizes
        against ``bedrock:CountTokens``, so the session-policy ceiling
        must grant it or every count-tokens request via OIDC auth 403s
        even when the role's identity policy allows it."""
        policy = _captured_policy()
        bedrock_stmt = _statement_by_sid(policy, "BedrockLiteLLM")
        actions = set(bedrock_stmt["Action"])
        assert "bedrock:CountTokens" in actions, (
            "bedrock:CountTokens missing from BedrockLiteLLM — count-tokens requests will 403 on OIDC auth"
        )


class TestClaudePlatformActionsCovered:
    """The #30200 bug: every action in the claude_platform service
    namespace must appear in the session policy or OIDC requests 403."""

    @pytest.mark.parametrize("action", sorted(_CLAUDE_PLATFORM_ACTIONS))
    def test_claude_platform_action_present(self, action: str):
        assert action in _granted_actions(_captured_policy()), (
            f"{action} missing from session policy — bedrock/claude_platform/* requests will 403 on OIDC auth"
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


class TestClaudePlatformOutboundIdentityCovered:
    """Claude Platform on AWS calls ``sts:GetWebIdentityToken`` with the
    caller's session before serving inference, and tags the token it
    requests (``sts:TagGetWebIdentityToken``). Without either in the ceiling,
    every claude_platform request via OIDC auth denies with "no session
    policy allows the sts:<action> action" even when the role's identity
    policy grants it."""

    @pytest.mark.parametrize("action", ["sts:GetWebIdentityToken", "sts:TagGetWebIdentityToken"])
    def test_outbound_identity_action_present(self, action: str):
        assert action in _granted_actions(_captured_policy()), (
            f"{action} missing from session policy, so bedrock/claude_platform/* requests will 403 on OIDC auth"
        )

    @pytest.mark.parametrize("action", ["sts:GetWebIdentityToken", "sts:TagGetWebIdentityToken"])
    def test_outbound_identity_action_is_granted_only_via_claude_platform(self, action: str):
        granting: Final = tuple(stmt for stmt in _captured_policy()["Statement"] if action in stmt["Action"])
        assert granting, f"{action} missing from session policy"
        assert all(
            stmt["Condition"].get("StringEquals") == {"aws:CalledViaLast": "aws-external-anthropic.amazonaws.com"}
            and stmt["Condition"]["Bool"] == {"aws:SecureTransport": "true"}
            for stmt in granting
        ), f"{action} must only be usable over TLS when Claude Platform on AWS calls STS for the session"

    def test_outbound_identity_statement_is_scoped(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "ClaudePlatformOutboundIdentityLiteLLM")
        assert stmt["Effect"] == "Allow"
        assert tuple(stmt["Action"]) == ("sts:GetWebIdentityToken", "sts:TagGetWebIdentityToken"), (
            "the outbound identity statement must grant only the two outbound identity actions"
        )

    def test_no_sts_wildcard(self):
        assert "sts:*" not in _granted_actions(_captured_policy()), (
            "session policy must not grant sts:*, only the two outbound identity actions are needed"
        )


class TestBedrockMantleActionsCovered:
    """LIT-3859: bedrock_mantle inference authorizes against the
    ``bedrock-mantle`` action namespace, so the session-policy ceiling
    must include it or every Mantle request via OIDC/WIF auth denies
    with "no session policy allows the bedrock-mantle:CreateInference
    action" even when the role's identity policy grants it."""

    def test_bedrock_mantle_create_inference_present(self):
        assert "bedrock-mantle:CreateInference" in _granted_actions(_captured_policy()), (
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
            "session policy must not grant bedrock-mantle:* — the ceiling should match the documented action set"
        )

    def test_bedrock_mantle_statement_carries_secure_transport_condition(self):
        policy = _captured_policy()
        stmt = _statement_by_sid(policy, "BedrockMantleLiteLLM")
        cond = stmt.get("Condition") or {}
        assert cond.get("Bool", {}).get("aws:SecureTransport") == "true", (
            "BedrockMantleLiteLLM must require aws:SecureTransport=true to keep parity with the bedrock statement"
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

    _AUD = "https://gateway.example.com"
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

        mock_sts.exceptions.InvalidIdentityTokenException = _InvalidIdentityTokenException
        mock_sts.assume_role_with_web_identity.side_effect = _InvalidIdentityTokenException(self._STS_MESSAGE)

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
            "ClaudePlatformLiteLLM must require aws:SecureTransport=true to keep parity with the bedrock statement"
        )

    def test_statements_for_services_litellm_calls_directly_require_no_calling_service(self):
        directly_called_services: Final = frozenset(
            {"bedrock", "bedrock-agentcore", "bedrock-mantle", "aws-external-anthropic"}
        )
        directly_called: Final = tuple(
            stmt
            for stmt in _captured_policy()["Statement"]
            if all(_service(action) in directly_called_services for action in stmt["Action"])
        )
        assert directly_called
        assert all("StringEquals" not in (stmt.get("Condition") or {}) for stmt in directly_called), (
            "statements LiteLLM calls directly must not require aws:CalledViaLast, or OIDC auth 403s on those routes"
        )


_STS_SESSION_POLICY_PLAINTEXT_LIMIT: Final = 2048

_BEDROCK_ROUTE_ACTIONS: Final = MappingProxyType(
    {
        "model/{model_id}/invoke": "bedrock:InvokeModel",
        "model/{model_id}/invoke-with-response-stream": "bedrock:InvokeModelWithResponseStream",
        "model/{model_id}/converse": "bedrock:InvokeModel",
        "model/{model_id}/converse-stream": "bedrock:InvokeModelWithResponseStream",
        "model/{model_id}/count-tokens": "bedrock:CountTokens",
        "model/{model_id}/invoke-with-bidirectional-stream": "bedrock:InvokeModel",
        "bedrock-mantle anthropic/v1/messages": "bedrock-mantle:CreateInference",
        "bedrock-mantle anthropic/v1/messages/count_tokens": "bedrock-mantle:CountTokens",
        "guardrail/{guardrail_id}/version/{version}/apply": "bedrock:ApplyGuardrail",
        "rerank": "bedrock:Rerank",
        "knowledgebases/{knowledge_base_id}/retrieve": "bedrock:Retrieve",
        "knowledgebases": "bedrock:ListKnowledgeBases",
        "agents/{agent_id}/agentAliases/{alias_id}/sessions/{session_id}/text": "bedrock:InvokeAgent",
        "runtimes/{agent_runtime_arn}/invocations": "bedrock-agentcore:InvokeAgentRuntime",
        "runtimes/{agent_runtime_arn}/invocations with X-Amzn-Bedrock-AgentCore-Runtime-User-Id": (
            "bedrock-agentcore:InvokeAgentRuntimeForUser"
        ),
        "mcp": "bedrock-agentcore:InvokeGateway",
    }
)


class TestSessionPolicyGrantsEveryBedrockRoute:
    """LIT-7348: ``/rerank`` authorizes against ``bedrock:Rerank``, which the
    ceiling never granted, so rerank 403d on web identity auth while static
    credentials and IRSA worked. Each route the bedrock package signs with the
    web identity session maps to the IAM action it authorizes against, and the
    ceiling must grant every one of them."""

    @pytest.mark.parametrize(("route", "action"), sorted(_BEDROCK_ROUTE_ACTIONS.items()))
    def test_route_action_is_granted_by_the_ceiling(self, route: str, action: str):
        assert action in _granted_actions(_captured_policy()), (
            f"/{route} authorizes against {action}, which the session policy does not grant, "
            "so it 403s on web identity auth"
        )

    def test_policy_document_fits_the_sts_plaintext_limit(self):
        assert len(_captured_policy_document()) <= _STS_SESSION_POLICY_PLAINTEXT_LIMIT


def _s3_logger() -> S3Logger:
    with patch("asyncio.create_task"):
        return S3Logger(s3_bucket_name="litellm-logs", s3_region_name="us-east-1")


def _sqs_logger() -> SQSLogger:
    with patch("asyncio.create_task"), patch("litellm.aws_sqs_callback_params", {}):
        return SQSLogger(
            sqs_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/logs", sqs_region_name="us-east-1"
        )


def _bedrock_knowledge_base_ingestion() -> BedrockRAGIngestion:
    return BedrockRAGIngestion(
        ingest_options={
            "vector_store": {"custom_llm_provider": "bedrock", "aws_region_name": "us-east-1"},
        }
    )


def _bedrock_model_lister() -> BedrockModelLister:
    return BedrockModelLister(deployment={"aws_region_name": "us-east-1"}, client=HTTPHandler())


def _s3_vectors_ingestion() -> S3VectorsRAGIngestion:
    return S3VectorsRAGIngestion(
        ingest_options={
            "embedding": {"model": "bedrock/amazon.titan-embed-text-v2:0"},
            "vector_store": {
                "custom_llm_provider": "s3_vectors",
                "vector_bucket_name": "litellm-vectors",
                "aws_region_name": "us-east-1",
            },
        }
    )


_CallerAndActions: TypeAlias = tuple[Callable[[], BaseAWSLLM], tuple[str, ...]]

_CALLER_ROUTE_ACTIONS: Final[MappingProxyType[str, _CallerAndActions]] = MappingProxyType(
    {
        "bedrock batches": (
            BedrockBatchesConfig,
            (
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:StopModelInvocationJob",
                "bedrock:TagResource",
                "iam:PassRole",
            ),
        ),
        "bedrock files": (
            BedrockFilesConfig,
            ("s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket", "kms:GenerateDataKey"),
        ),
        "bedrock file download": (BedrockFilesHandler, ("s3:GetObject", "kms:Decrypt")),
        "bedrock embeddings": (
            BedrockEmbedding,
            (
                "bedrock:InvokeModel",
                "bedrock:ApplyGuardrail",
                "bedrock:GetAsyncInvoke",
                "s3:GetObject",
                "s3:PutObject",
                "kms:GenerateDataKey",
                "kms:Decrypt",
            ),
        ),
        "bedrock realtime": (BedrockRealtime, ("bedrock:InvokeModel", "bedrock:ApplyGuardrail")),
        "bedrock agent runtime passthrough": (
            BedrockAgentRuntimeAWS,
            (
                "bedrock:InvokeAgent",
                "bedrock:InvokeInlineAgent",
                "bedrock:InvokeFlow",
                "bedrock:StartFlowExecution",
                "bedrock:StopFlowExecution",
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate",
                "bedrock:GenerateQuery",
                "bedrock:GetKnowledgeBase",
                "bedrock:GetDocumentContent",
                "bedrock:CheckIngestedDocumentAcl",
                "bedrock:GetIngestedDocumentAcl",
                "bedrock:Rerank",
                "bedrock:OptimizePrompt",
                "bedrock:InvokeModel",
                "bedrock:ApplyGuardrail",
                "bedrock:GetInferenceProfile",
                "sqlworkbench:GetSqlRecommendations",
            ),
        ),
        "bedrock model listing": (
            _bedrock_model_lister,
            ("bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"),
        ),
        "bedrock knowledge base ingestion": (
            _bedrock_knowledge_base_ingestion,
            (
                "bedrock:ListDataSources",
                "bedrock:GetDataSource",
                "bedrock:CreateKnowledgeBase",
                "bedrock:GetKnowledgeBase",
                "bedrock:CreateDataSource",
                "bedrock:StartIngestionJob",
                "bedrock:GetIngestionJob",
                "s3:CreateBucket",
                "s3:PutObject",
                "kms:GenerateDataKey",
                "aoss:CreateSecurityPolicy",
                "aoss:CreateAccessPolicy",
                "aoss:CreateCollection",
                "aoss:BatchGetCollection",
                "aoss:APIAccessAll",
                "iam:CreateServiceLinkedRole",
                "iam:CreateRole",
                "iam:PutRolePolicy",
                "iam:PassRole",
            ),
        ),
        "sagemaker completion": (
            SagemakerLLM,
            ("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"),
        ),
        "sagemaker chat": (
            SagemakerChatConfig,
            ("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"),
        ),
        "sagemaker chat handler": (
            SagemakerChatHandler,
            ("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"),
        ),
        "polly speech": (AWSPollyTextToSpeechConfig, ("polly:SynthesizeSpeech",)),
        "s3 vectors search": (S3VectorsVectorStoreConfig, ("s3vectors:QueryVectors",)),
        "s3 vectors ingest": (
            _s3_vectors_ingestion,
            (
                "s3vectors:GetVectorBucket",
                "s3vectors:CreateVectorBucket",
                "s3vectors:GetIndex",
                "s3vectors:CreateIndex",
                "s3vectors:PutVectors",
            ),
        ),
        "s3 logging": (_s3_logger, ("s3:PutObject", "kms:GenerateDataKey")),
        "sqs logging": (_sqs_logger, ("sqs:SendMessage", "kms:GenerateDataKey")),
        "secrets manager": (
            AWSSecretsManagerV2,
            (
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret",
                "secretsmanager:CreateSecret",
                "secretsmanager:PutSecretValue",
                "secretsmanager:UpdateSecret",
                "secretsmanager:RestoreSecret",
                "secretsmanager:TagResource",
                "secretsmanager:ReplicateSecretToRegions",
                "secretsmanager:DeleteSecret",
                "kms:GenerateDataKey",
                "kms:Decrypt",
            ),
        ),
        "comprehend medical passthrough": (
            ComprehendMedicalAWS,
            tuple(f"comprehendmedical:{operation}" for operation in sorted(COMPREHEND_MEDICAL_SUPPORTED_OPERATIONS)),
        ),
        "transcribe passthrough": (
            TranscribeAWS,
            (
                "transcribe:StartTranscriptionJob",
                "transcribe:GetTranscriptionJob",
                "s3:GetObject",
                "s3:PutObject",
                "iam:PassRole",
                "kms:GenerateDataKey",
                "kms:Decrypt",
            ),
        ),
    }
)


_KMS_VIA_SERVICE_BY_ROUTE: Final = MappingProxyType(
    {
        "bedrock files": "s3.*",
        "bedrock file download": "s3.*",
        "bedrock embeddings": "s3.*",
        "s3 logging": "s3.*",
        "sqs logging": "sqs.*",
        "bedrock knowledge base ingestion": "s3.*",
        "secrets manager": "secretsmanager.*",
        "transcribe passthrough": "s3.*",
    }
)


_PASS_ROLE_TARGET_BY_ROUTE: Final = MappingProxyType(
    {
        "bedrock batches": "bedrock.amazonaws.com",
        "bedrock knowledge base ingestion": "bedrock.amazonaws.com",
        "transcribe passthrough": "transcribe.amazonaws.com",
    }
)


def _sts_mock() -> MagicMock:
    sts: Final = MagicMock()
    sts.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": "k",
            "SecretAccessKey": "s",
            "SessionToken": "t",
            "Expiration": _STS_SESSION_EXPIRATION,
        },
        "PackedPolicySize": 0,
    }
    return sts


def _policies_sent_to_sts(callers: tuple[BaseAWSLLM, ...], session_name: str) -> tuple[WebIdentitySessionPolicy, ...]:
    sts: Final = _sts_mock()
    with (
        patch("boto3.client", return_value=sts),
        patch("litellm.llms.bedrock.base_aws_llm.get_secret", return_value="oidc-jwt-token"),
    ):
        for caller in callers:
            caller.get_credentials(
                aws_web_identity_token="oidc/google/litellm",
                aws_role_name="arn:aws:iam::123456789012:role/litellm-role",
                aws_session_name=session_name,
                aws_region_name="us-east-1",
            )
    return tuple(
        _SESSION_POLICY_ADAPTER.validate_python(json.loads(call.kwargs["Policy"]))
        for call in sts.assume_role_with_web_identity.call_args_list
    )


_PROXY_ROLE_ARN: Final = "arn:aws:iam::123456789012:role/litellm-role"

_KNOWLEDGE_BASE_ROLE_ACTIONS: Final = frozenset({"iam:CreateRole", "iam:PutRolePolicy", "iam:PassRole"})


def _role_created_in_account(**create_role_request: str) -> dict[str, dict[str, str]]:
    return {"Role": {"Arn": f"arn:aws:iam::123456789012:role/{create_role_request['RoleName']}"}}


def _service(action: str) -> str:
    return action.split(":", 1)[0]


def _grants_kms(statement: _SessionPolicyStatement) -> bool:
    return any(_service(action) == "kms" for action in statement["Action"])


def _ungranted(actions: tuple[str, ...], policy: WebIdentitySessionPolicy) -> tuple[str, ...]:
    granted: Final = _granted_actions(policy)
    return tuple(action for action in actions if not any(fnmatchcase(action, pattern) for pattern in granted))


class TestEachCallerGetsItsOwnCeiling:
    """Every AWS caller besides Bedrock runtime (batches, files, model listing, knowledge base
    ingestion, SageMaker, Polly, S3 Vectors, the loggers, Secrets Manager, the config bucket, and the
    Comprehend Medical and Transcribe pass-throughs) authorizes against actions outside the Bedrock
    runtime ceiling, so on web identity auth it 403d with "no session policy allows" even when the
    role allowed it. Each caller's STS session must carry a ceiling that grants what its routes call
    and nothing from another caller's service."""

    @pytest.mark.parametrize(("route", "caller_and_actions"), sorted(_CALLER_ROUTE_ACTIONS.items()))
    def test_route_actions_are_granted_by_the_callers_ceiling(self, route: str, caller_and_actions: _CallerAndActions):
        make_caller, actions = caller_and_actions
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"grant-{route}")
        missing: Final = _ungranted(actions, policy)
        assert not missing, f"{route} calls {sorted(missing)}, which its session policy does not grant"

    @pytest.mark.parametrize(("route", "caller_and_actions"), sorted(_CALLER_ROUTE_ACTIONS.items()))
    def test_ceiling_grants_no_service_the_route_does_not_call(self, route: str, caller_and_actions: _CallerAndActions):
        make_caller, actions = caller_and_actions
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"deny-{route}")
        called_services: Final = frozenset(_service(action) for action in actions)
        extra: Final = sorted(a for a in _granted_actions(policy) if _service(a) not in called_services)
        assert not extra, f"{route}'s session policy also grants {extra}"

    @pytest.mark.parametrize(("route", "caller_and_actions"), sorted(_CALLER_ROUTE_ACTIONS.items()))
    def test_ceiling_fits_the_sts_plaintext_limit(self, route: str, caller_and_actions: _CallerAndActions):
        make_caller, _actions = caller_and_actions
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"size-{route}")
        assert len(json.dumps(policy, separators=(",", ":"))) <= _STS_SESSION_POLICY_PLAINTEXT_LIMIT

    def test_bedrock_chat_ceiling_reaches_no_other_callers_service(self):
        every_route_action: Final = itertools.chain.from_iterable(
            actions for _caller, actions in _CALLER_ROUTE_ACTIONS.values()
        )
        other_services: Final = frozenset(
            _service(action) for action in every_route_action if _service(action) != "bedrock"
        )
        (policy,) = _policies_sent_to_sts((BedrockConverseLLM(),), session_name="chat-reach")
        reached: Final = sorted(action for action in _granted_actions(policy) if _service(action) in other_services)
        assert not reached, f"a Bedrock chat session can also call {reached}"

    @pytest.mark.parametrize(("route", "target_service"), sorted(_PASS_ROLE_TARGET_BY_ROUTE.items()))
    def test_service_role_can_only_be_passed_to_the_routes_service(self, route: str, target_service: str):
        make_caller, _actions = _CALLER_ROUTE_ACTIONS[route]
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"pass-role-{route}")
        granting: Final = tuple(stmt for stmt in policy["Statement"] if "iam:PassRole" in stmt["Action"])
        assert granting
        assert all(
            stmt["Condition"].get("StringEquals") == {"iam:PassedToService": target_service} for stmt in granting
        )

    @pytest.mark.parametrize("action", ["s3:GetObject", "s3:PutObject"])
    def test_async_invoke_bucket_access_is_only_granted_through_bedrock(self, action: str):
        (policy,) = _policies_sent_to_sts((BedrockEmbedding(),), session_name=f"async-{action}")
        granting: Final = tuple(stmt for stmt in policy["Statement"] if action in stmt["Action"])
        assert granting
        assert all(
            stmt["Condition"].get("StringEquals") == {"aws:CalledViaLast": "bedrock.amazonaws.com"} for stmt in granting
        )

    def test_transcribe_ceiling_grants_every_operation_the_passthrough_serves(self):
        (policy,) = _policies_sent_to_sts((TranscribeAWS(),), session_name="transcribe-operations")
        operations: Final = tuple(f"transcribe:{operation}" for operation in sorted(transcribe_supported_operations()))
        assert not _ungranted(operations, policy)

    def test_transcript_output_write_is_only_granted_through_transcribe(self):
        (policy,) = _policies_sent_to_sts((TranscribeAWS(),), session_name="transcribe-output")
        granting: Final = tuple(stmt for stmt in policy["Statement"] if "s3:PutObject" in stmt["Action"])
        assert granting
        assert all(
            stmt["Condition"].get("StringEquals") == {"aws:CalledViaLast": "transcribe.amazonaws.com"}
            for stmt in granting
        )

    def test_knowledge_base_ingestion_can_only_create_the_opensearch_serverless_service_linked_role(self):
        (policy,) = _policies_sent_to_sts((_bedrock_knowledge_base_ingestion(),), session_name="kb-slr")
        granting: Final = tuple(stmt for stmt in policy["Statement"] if "iam:CreateServiceLinkedRole" in stmt["Action"])
        assert granting
        assert all(
            stmt["Condition"].get("StringEquals") == {"iam:AWSServiceName": "observability.aoss.amazonaws.com"}
            for stmt in granting
        )

    async def test_knowledge_base_ingestion_can_only_create_and_pass_the_roles_it_names(self):
        ingestion: Final = BedrockRAGIngestion(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "bedrock",
                    "aws_region_name": "us-east-1",
                    "aws_web_identity_token": "oidc/google/litellm",
                    "aws_role_name": _PROXY_ROLE_ARN,
                    "aws_session_name": "kb-role-scope",
                }
            }
        )
        sts: Final = _sts_mock()
        iam: Final = MagicMock()
        iam.create_role.side_effect = _role_created_in_account
        with (
            patch("boto3.client", return_value=sts),
            patch("boto3.Session", return_value=MagicMock(client=MagicMock(return_value=iam))),
            patch("litellm.llms.bedrock.base_aws_llm.get_secret", return_value="oidc-jwt-token"),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            role_arn: Final = await ingestion._create_bedrock_role(
                "abcd1234", "123456789012", "arn:aws:aoss:us-east-1:123456789012:collection/kb"
            )
        policy: Final = _SESSION_POLICY_ADAPTER.validate_python(
            json.loads(sts.assume_role_with_web_identity.call_args.kwargs["Policy"])
        )
        granting: Final = tuple(
            stmt for stmt in policy["Statement"] if _KNOWLEDGE_BASE_ROLE_ACTIONS & set(stmt["Action"])
        )
        assert _granted_actions(policy) >= _KNOWLEDGE_BASE_ROLE_ACTIONS
        assert all(fnmatchcase(role_arn, stmt["Resource"]) for stmt in granting)
        assert not any(fnmatchcase(_PROXY_ROLE_ARN, stmt["Resource"]) for stmt in granting)

    def test_config_bucket_reader_signs_with_a_read_only_ceiling(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN", "oidc/google/litellm")
        monkeypatch.setenv("AWS_ROLE_NAME", "arn:aws:iam::123456789012:role/litellm-role")
        monkeypatch.setenv("AWS_SESSION_NAME", "config-bucket")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        sts: Final = _sts_mock()
        with (
            patch("boto3.client", return_value=sts),
            patch("litellm.llms.bedrock.base_aws_llm.get_secret", return_value="oidc-jwt-token"),
        ):
            s3_object_reader("litellm-config")
        policy: Final = _SESSION_POLICY_ADAPTER.validate_python(
            json.loads(sts.assume_role_with_web_identity.call_args.kwargs["Policy"])
        )
        assert _granted_actions(policy) == frozenset({"s3:GetObject", "kms:Decrypt"})

    def test_file_download_ceiling_cannot_write_list_or_delete(self):
        (policy,) = _policies_sent_to_sts((BedrockFilesHandler(),), session_name="file-download-only")
        assert _granted_actions(policy) == frozenset({"s3:GetObject", "kms:Decrypt"})

    @pytest.mark.parametrize(("route", "via_service"), sorted(_KMS_VIA_SERVICE_BY_ROUTE.items()))
    def test_kms_is_only_granted_through_the_routes_storage_service(self, route: str, via_service: str):
        make_caller, _actions = _CALLER_ROUTE_ACTIONS[route]
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"kms-via-{route}")
        granting: Final = tuple(stmt for stmt in policy["Statement"] if _grants_kms(stmt))
        assert granting
        assert all(stmt["Condition"].get("StringLike") == {"kms:ViaService": via_service} for stmt in granting)


_OPENSEARCH_ENDPOINT_ACTIONS: Final = MappingProxyType(
    {
        "https://search-docs-abc123.us-east-1.es.amazonaws.com": frozenset({"es:ESHttpGet", "es:ESHttpPost"}),
        "https://abc123.us-east-1.aoss.amazonaws.com": frozenset({"aoss:APIAccessAll"}),
    }
)


def _opensearch_policy(api_base: str) -> WebIdentitySessionPolicy:
    sts: Final = _sts_mock()
    with (
        patch("boto3.client", return_value=sts),
        patch("litellm.llms.bedrock.base_aws_llm.get_secret", return_value="oidc-jwt-token"),
    ):
        OpenSearchVectorStoreConfig().validate_environment(
            {},
            GenericLiteLLMParams(
                api_base=api_base,
                aws_web_identity_token="oidc/google/litellm",
                aws_role_name="arn:aws:iam::123456789012:role/litellm-role",
                aws_session_name=f"opensearch-{api_base}",
                aws_region_name="us-east-1",
            ),
        )
    return _SESSION_POLICY_ADAPTER.validate_python(
        json.loads(sts.assume_role_with_web_identity.call_args.kwargs["Policy"])
    )


class TestOpenSearchSearchGetsItsOwnCeiling:
    @pytest.mark.parametrize(("api_base", "actions"), sorted(_OPENSEARCH_ENDPOINT_ACTIONS.items()))
    def test_search_and_mapping_read_actions_are_granted_by_the_opensearch_ceiling(
        self, api_base: str, actions: frozenset[str]
    ):
        assert actions <= _granted_actions(_opensearch_policy(api_base))

    @pytest.mark.parametrize("api_base", sorted(_OPENSEARCH_ENDPOINT_ACTIONS))
    def test_opensearch_ceiling_grants_only_opensearch_services(self, api_base: str):
        allowed: Final = frozenset(map(_service, frozenset[str]().union(*_OPENSEARCH_ENDPOINT_ACTIONS.values())))
        extra: Final = sorted(a for a in _granted_actions(_opensearch_policy(api_base)) if _service(a) not in allowed)
        assert not extra, f"the OpenSearch session policy also grants {extra}"


class TestInvokeModelTransport:
    def test_chat_ceiling_grants_invoke_model_only_over_tls(self):
        (policy,) = _policies_sent_to_sts((BedrockConverseLLM(),), session_name="chat-transport")
        assert "bedrock:InvokeModel" in _granted_actions(policy)
        assert not _invoke_model_grants_over_any_transport(policy)

    def test_realtime_ceiling_grants_invoke_model_over_any_transport_and_nothing_else(self):
        (policy,) = _policies_sent_to_sts((BedrockRealtime(),), session_name="realtime-transport")
        assert _invoke_model_grants_over_any_transport(policy), (
            "Bedrock authorizes the Nova Sonic bidirectional stream as bedrock:InvokeModel with aws:SecureTransport "
            "false (live probe on amazon.nova-2-sonic-v1:0, 2026-11-08), "
            "so a TLS-only grant denies every realtime session"
        )
        assert _granted_actions(policy) == frozenset({"bedrock:InvokeModel", "bedrock:ApplyGuardrail"})


_MODEL_INVOKING_CALLERS: Final[MappingProxyType[str, Callable[[], BaseAWSLLM]]] = MappingProxyType(
    {
        "bedrock chat": BedrockConverseLLM,
        **{
            route: make_caller
            for route, (make_caller, actions) in _CALLER_ROUTE_ACTIONS.items()
            if "bedrock:InvokeModel" in actions
        },
    }
)


class TestEnforcedGuardrails:
    @pytest.mark.parametrize(("route", "make_caller"), sorted(_MODEL_INVOKING_CALLERS.items()))
    def test_every_grant_to_invoke_models_also_grants_apply_guardrail(
        self, route: str, make_caller: Callable[[], BaseAWSLLM]
    ):
        (policy,) = _policies_sent_to_sts((make_caller(),), session_name=f"guardrail-{route}")
        invoking: Final = tuple(stmt for stmt in policy["Statement"] if "bedrock:InvokeModel" in stmt["Action"])
        assert invoking
        assert all("bedrock:ApplyGuardrail" in stmt["Action"] for stmt in invoking), (
            f"{route} can invoke models but not apply an account or organization enforced guardrail, "
            "which Bedrock answers with AccessDenied"
        )


def _agent_runtime_operations_the_passthrough_signs_and_forwards() -> tuple[str, ...]:
    service_model: Final = get_session().get_service_model("bedrock-agent-runtime")
    http_bindings: Final = tuple(
        (name, service_model.operation_model(name).http) for name in service_model.operation_names
    )
    return tuple(
        name
        for name, http in http_bindings
        if http["method"] == "POST" and _is_bedrock_agent_runtime_route(f"{http['requestUri'].lstrip('/')}/")
    )


class TestAgentRuntimePassThroughCeiling:
    def test_ceiling_grants_every_post_operation_the_passthrough_forwards(self):
        operations: Final = _agent_runtime_operations_the_passthrough_signs_and_forwards()
        (policy,) = _policies_sent_to_sts((BedrockAgentRuntimeAWS(),), session_name="agent-runtime-operations")
        assert operations
        missing: Final = _ungranted(tuple(f"bedrock:{operation}" for operation in operations), policy)
        assert not missing, (
            f"the agent runtime pass-through forwards {sorted(missing)}, which its ceiling does not grant"
        )


class TestWebIdentityCredentialCacheIsPerCeiling:
    def test_callers_with_different_ceilings_get_their_own_sts_session(self):
        bedrock_policy, sagemaker_policy = _policies_sent_to_sts(
            (BaseAWSLLM(), SagemakerLLM()), session_name="cache-split"
        )
        assert "sagemaker:InvokeEndpoint" not in _granted_actions(bedrock_policy)
        assert "sagemaker:InvokeEndpoint" in _granted_actions(sagemaker_policy)

    def test_callers_with_the_same_ceiling_share_one_sts_session(self):
        policies: Final = _policies_sent_to_sts((SagemakerLLM(), SagemakerLLM()), session_name="cache-share")
        assert len(policies) == 1


class TestBedrockBatchRequestSigning:
    def test_create_batch_job_is_signed_with_the_batch_ceiling(self):
        sts: Final = _sts_mock()
        with (
            patch("boto3.client", return_value=sts),
            patch("litellm.llms.bedrock.base_aws_llm.get_secret", return_value="oidc-jwt-token"),
        ):
            BedrockBatchesConfig().common_utils.sign_aws_request(
                service_name="bedrock",
                data={"jobName": "litellm-batch-job"},
                endpoint_url="https://bedrock.us-east-1.amazonaws.com/model-invocation-job",
                optional_params={
                    "aws_web_identity_token": "oidc/google/litellm",
                    "aws_role_name": "arn:aws:iam::123456789012:role/litellm-role",
                    "aws_session_name": "batch-sign",
                    "aws_region_name": "us-east-1",
                },
            )
        policy: Final = _SESSION_POLICY_ADAPTER.validate_python(
            json.loads(sts.assume_role_with_web_identity.call_args.kwargs["Policy"])
        )
        assert "bedrock:CreateModelInvocationJob" in _granted_actions(policy)
