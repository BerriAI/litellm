import json
from typing import Final, Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

_SecureTransportBool: Final = TypedDict("_SecureTransportBool", {"aws:SecureTransport": ReadOnly[Literal["true"]]})

_CalledViaLastEquals: Final = TypedDict(
    "_CalledViaLastEquals", {"aws:CalledViaLast": ReadOnly[Literal["aws-external-anthropic.amazonaws.com"]]}
)

_PassedToServiceEquals: Final = TypedDict(
    "_PassedToServiceEquals", {"iam:PassedToService": ReadOnly[Literal["bedrock.amazonaws.com"]]}
)


class _SessionPolicyCondition(TypedDict):
    Bool: ReadOnly[_SecureTransportBool]
    StringEquals: NotRequired[ReadOnly[_CalledViaLastEquals | _PassedToServiceEquals]]


class _SessionPolicyStatement(TypedDict):
    Sid: ReadOnly[str]
    Effect: ReadOnly[Literal["Allow"]]
    Action: ReadOnly[tuple[str, ...]]
    Resource: ReadOnly[Literal["*"]]
    Condition: ReadOnly[_SessionPolicyCondition]


class WebIdentitySessionPolicy(TypedDict):
    Version: ReadOnly[Literal["2012-10-17"]]
    Statement: ReadOnly[tuple[_SessionPolicyStatement, ...]]


_SECURE_TRANSPORT: Final = _SecureTransportBool({"aws:SecureTransport": "true"})

_SECURE_TRANSPORT_ONLY: Final = _SessionPolicyCondition(Bool=_SECURE_TRANSPORT)


def _allow(
    sid: str, actions: tuple[str, ...], condition: _SessionPolicyCondition = _SECURE_TRANSPORT_ONLY
) -> _SessionPolicyStatement:
    return _SessionPolicyStatement(Sid=sid, Effect="Allow", Action=actions, Resource="*", Condition=condition)


def _policy(*statements: _SessionPolicyStatement) -> WebIdentitySessionPolicy:
    return WebIdentitySessionPolicy(Version="2012-10-17", Statement=statements)


def session_policy_document(policy: WebIdentitySessionPolicy) -> str:
    return json.dumps(policy, separators=(",", ":"))


_OBJECT_ENCRYPTION: Final = _allow("KmsLiteLLM", ("kms:GenerateDataKey", "kms:Decrypt"))

BEDROCK_SESSION_POLICY: Final = _policy(
    _allow(
        "BedrockLiteLLM",
        (
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
            "bedrock:CountTokens",
            "bedrock:Rerank",
            "bedrock:Retrieve",
            "bedrock:ListKnowledgeBases",
            "bedrock:InvokeAgent",
            "bedrock:ApplyGuardrail",
            "bedrock:GetGuardrail",
            "bedrock:ListGuardrails",
        ),
    ),
    _allow(
        "BedrockAgentCoreLiteLLM",
        (
            "bedrock-agentcore:InvokeAgentRuntime",
            "bedrock-agentcore:InvokeAgentRuntimeForUser",
            "bedrock-agentcore:InvokeGateway",
        ),
    ),
    _allow(
        "ClaudePlatformLiteLLM",
        (
            "aws-external-anthropic:CreateInference",
            "aws-external-anthropic:CreateBatchInference",
            "aws-external-anthropic:CancelBatchInference",
            "aws-external-anthropic:DeleteBatchInference",
            "aws-external-anthropic:CountTokens",
            "aws-external-anthropic:Get*",
            "aws-external-anthropic:List*",
        ),
    ),
    _allow(
        "ClaudePlatformOutboundIdentityLiteLLM",
        ("sts:GetWebIdentityToken", "sts:TagGetWebIdentityToken"),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_CalledViaLastEquals({"aws:CalledViaLast": "aws-external-anthropic.amazonaws.com"}),
        ),
    ),
    _allow("BedrockMantleLiteLLM", ("bedrock-mantle:CreateInference",)),
)

BEDROCK_EMBEDDING_SESSION_POLICY: Final = _policy(
    _allow("BedrockEmbeddingLiteLLM", ("bedrock:InvokeModel", "bedrock:GetAsyncInvoke")),
)

BEDROCK_BATCHES_SESSION_POLICY: Final = _policy(
    _allow(
        "BedrockBatchLiteLLM",
        (
            "bedrock:CreateModelInvocationJob",
            "bedrock:GetModelInvocationJob",
            "bedrock:StopModelInvocationJob",
            "bedrock:TagResource",
        ),
    ),
    _allow(
        "BedrockBatchServiceRoleLiteLLM",
        ("iam:PassRole",),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_PassedToServiceEquals({"iam:PassedToService": "bedrock.amazonaws.com"}),
        ),
    ),
)

BEDROCK_FILES_SESSION_POLICY: Final = _policy(
    _allow("S3ObjectsLiteLLM", ("s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket")),
    _OBJECT_ENCRYPTION,
)

SAGEMAKER_SESSION_POLICY: Final = _policy(
    _allow("SageMakerLiteLLM", ("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream")),
)

POLLY_SESSION_POLICY: Final = _policy(_allow("PollyLiteLLM", ("polly:SynthesizeSpeech",)))

S3_VECTORS_QUERY_SESSION_POLICY: Final = _policy(
    _allow("S3VectorsLiteLLM", ("s3vectors:QueryVectors", "s3vectors:GetVectors")),
)

S3_VECTORS_INGESTION_SESSION_POLICY: Final = _policy(
    _allow(
        "S3VectorsLiteLLM",
        (
            "s3vectors:GetVectorBucket",
            "s3vectors:CreateVectorBucket",
            "s3vectors:GetIndex",
            "s3vectors:CreateIndex",
            "s3vectors:PutVectors",
        ),
    ),
)

S3_LOGGING_SESSION_POLICY: Final = _policy(
    _allow("S3ObjectsLiteLLM", ("s3:PutObject", "s3:GetObject")),
    _OBJECT_ENCRYPTION,
)

SQS_LOGGING_SESSION_POLICY: Final = _policy(
    _allow("SqsLiteLLM", ("sqs:SendMessage",)),
    _OBJECT_ENCRYPTION,
)
