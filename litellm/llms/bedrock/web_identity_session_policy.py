import json
from typing import Final, Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

_SecureTransportBool: Final = TypedDict("_SecureTransportBool", {"aws:SecureTransport": ReadOnly[Literal["true"]]})

_CalledViaLastEquals: Final = TypedDict(
    "_CalledViaLastEquals",
    {
        "aws:CalledViaLast": ReadOnly[
            Literal["aws-external-anthropic.amazonaws.com", "bedrock.amazonaws.com", "transcribe.amazonaws.com"]
        ]
    },
)

_PassedToServiceEquals: Final = TypedDict(
    "_PassedToServiceEquals",
    {"iam:PassedToService": ReadOnly[Literal["bedrock.amazonaws.com", "transcribe.amazonaws.com"]]},
)

_ServiceLinkedRoleEquals: Final = TypedDict(
    "_ServiceLinkedRoleEquals", {"iam:AWSServiceName": ReadOnly[Literal["observability.aoss.amazonaws.com"]]}
)

_KmsViaService: TypeAlias = Literal["s3.*", "sqs.*", "secretsmanager.*"]

_ViaServiceLike: Final = TypedDict("_ViaServiceLike", {"kms:ViaService": ReadOnly[_KmsViaService]})

_StatementResource: TypeAlias = Literal["*", "arn:*:iam::*:role/litellm-bedrock-kb-*"]


class _SessionPolicyCondition(TypedDict):
    Bool: ReadOnly[_SecureTransportBool]
    StringEquals: NotRequired[ReadOnly[_CalledViaLastEquals | _PassedToServiceEquals | _ServiceLinkedRoleEquals]]
    StringLike: NotRequired[ReadOnly[_ViaServiceLike]]


class _SessionPolicyStatement(TypedDict):
    Sid: ReadOnly[str]
    Effect: ReadOnly[Literal["Allow"]]
    Action: ReadOnly[tuple[str, ...]]
    Resource: ReadOnly[_StatementResource]
    Condition: NotRequired[ReadOnly[_SessionPolicyCondition]]


class WebIdentitySessionPolicy(TypedDict):
    Version: ReadOnly[Literal["2012-10-17"]]
    Statement: ReadOnly[tuple[_SessionPolicyStatement, ...]]


_SECURE_TRANSPORT: Final = _SecureTransportBool({"aws:SecureTransport": "true"})

_SECURE_TRANSPORT_ONLY: Final = _SessionPolicyCondition(Bool=_SECURE_TRANSPORT)


def _allow(
    sid: str,
    actions: tuple[str, ...],
    condition: _SessionPolicyCondition = _SECURE_TRANSPORT_ONLY,
    resource: _StatementResource = "*",
) -> _SessionPolicyStatement:
    return _SessionPolicyStatement(Sid=sid, Effect="Allow", Action=actions, Resource=resource, Condition=condition)


def _allow_over_any_transport(sid: str, actions: tuple[str, ...]) -> _SessionPolicyStatement:
    return _SessionPolicyStatement(Sid=sid, Effect="Allow", Action=actions, Resource="*")


def _policy(*statements: _SessionPolicyStatement) -> WebIdentitySessionPolicy:
    return WebIdentitySessionPolicy(Version="2012-10-17", Statement=statements)


def session_policy_document(policy: WebIdentitySessionPolicy) -> str:
    return json.dumps(policy, separators=(",", ":"))


def _encryption_through(service: _KmsViaService, actions: tuple[str, ...]) -> _SessionPolicyStatement:
    return _allow(
        "KmsLiteLLM",
        actions,
        _SessionPolicyCondition(Bool=_SECURE_TRANSPORT, StringLike=_ViaServiceLike({"kms:ViaService": service})),
    )


_OBJECT_ENCRYPTION: Final = ("kms:GenerateDataKey", "kms:Decrypt")

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
    _allow("BedrockMantleLiteLLM", ("bedrock-mantle:CreateInference", "bedrock-mantle:CountTokens")),
)

BEDROCK_REALTIME_SESSION_POLICY: Final = _policy(
    _allow_over_any_transport("BedrockRealtimeLiteLLM", ("bedrock:InvokeModel", "bedrock:ApplyGuardrail")),
)

BEDROCK_EMBEDDING_SESSION_POLICY: Final = _policy(
    _allow("BedrockEmbeddingLiteLLM", ("bedrock:InvokeModel", "bedrock:ApplyGuardrail", "bedrock:GetAsyncInvoke")),
    _allow(
        "BedrockS3LiteLLM",
        ("s3:GetObject", "s3:PutObject"),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_CalledViaLastEquals({"aws:CalledViaLast": "bedrock.amazonaws.com"}),
        ),
    ),
    _encryption_through("s3.*", _OBJECT_ENCRYPTION),
)

BEDROCK_AGENT_RUNTIME_SESSION_POLICY: Final = _policy(
    _allow(
        "BedrockAgentRuntimeLiteLLM",
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
        ),
    ),
    _allow("SqlWorkbenchLiteLLM", ("sqlworkbench:GetSqlRecommendations",)),
)

BEDROCK_MODEL_LISTING_SESSION_POLICY: Final = _policy(
    _allow("BedrockModelListingLiteLLM", ("bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles")),
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
    _encryption_through("s3.*", _OBJECT_ENCRYPTION),
)

S3_OBJECT_READ_SESSION_POLICY: Final = _policy(
    _allow("S3ObjectsLiteLLM", ("s3:GetObject",)),
    _encryption_through("s3.*", ("kms:Decrypt",)),
)

BEDROCK_KNOWLEDGE_BASE_INGESTION_SESSION_POLICY: Final = _policy(
    _allow(
        "BedrockKnowledgeBaseLiteLLM",
        (
            "bedrock:ListDataSources",
            "bedrock:GetDataSource",
            "bedrock:CreateKnowledgeBase",
            "bedrock:GetKnowledgeBase",
            "bedrock:CreateDataSource",
            "bedrock:StartIngestionJob",
            "bedrock:GetIngestionJob",
        ),
    ),
    _allow("S3ObjectsLiteLLM", ("s3:CreateBucket", "s3:PutObject")),
    _encryption_through("s3.*", _OBJECT_ENCRYPTION),
    _allow(
        "OpenSearchServerlessLiteLLM",
        (
            "aoss:CreateSecurityPolicy",
            "aoss:CreateAccessPolicy",
            "aoss:CreateCollection",
            "aoss:BatchGetCollection",
            "aoss:APIAccessAll",
        ),
    ),
    _allow(
        "OpenSearchServerlessServiceLinkedRoleLiteLLM",
        ("iam:CreateServiceLinkedRole",),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_ServiceLinkedRoleEquals({"iam:AWSServiceName": "observability.aoss.amazonaws.com"}),
        ),
    ),
    _allow(
        "KnowledgeBaseRoleLiteLLM",
        ("iam:CreateRole", "iam:PutRolePolicy"),
        resource="arn:*:iam::*:role/litellm-bedrock-kb-*",
    ),
    _allow(
        "KnowledgeBaseServiceRoleLiteLLM",
        ("iam:PassRole",),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_PassedToServiceEquals({"iam:PassedToService": "bedrock.amazonaws.com"}),
        ),
        resource="arn:*:iam::*:role/litellm-bedrock-kb-*",
    ),
)

SAGEMAKER_SESSION_POLICY: Final = _policy(
    _allow("SageMakerLiteLLM", ("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream")),
)

POLLY_SESSION_POLICY: Final = _policy(_allow("PollyLiteLLM", ("polly:SynthesizeSpeech",)))

OPENSEARCH_SESSION_POLICY: Final = _policy(_allow("OpenSearchLiteLLM", ("es:ESHttpPost", "aoss:APIAccessAll")))

OPENSEARCH_INDEX_SESSION_POLICY: Final = _policy(
    _allow(
        "OpenSearchIndexLiteLLM",
        (
            "es:ESHttpGet",
            "es:ESHttpHead",
            "es:ESHttpPost",
            "es:ESHttpPut",
            "es:ESHttpDelete",
            "es:ESHttpPatch",
            "aoss:APIAccessAll",
        ),
    ),
)

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
    _encryption_through("s3.*", _OBJECT_ENCRYPTION),
)

SQS_LOGGING_SESSION_POLICY: Final = _policy(
    _allow("SqsLiteLLM", ("sqs:SendMessage",)),
    _encryption_through("sqs.*", _OBJECT_ENCRYPTION),
)

SECRETS_MANAGER_SESSION_POLICY: Final = _policy(
    _allow(
        "SecretsManagerLiteLLM",
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
        ),
    ),
    _encryption_through("secretsmanager.*", _OBJECT_ENCRYPTION),
)

COMPREHEND_MEDICAL_SESSION_POLICY: Final = _policy(
    _allow(
        "ComprehendMedicalLiteLLM",
        (
            "comprehendmedical:DetectEntitiesV2",
            "comprehendmedical:DetectPHI",
            "comprehendmedical:InferICD10CM",
            "comprehendmedical:InferRxNorm",
            "comprehendmedical:InferSNOMEDCT",
        ),
    ),
)

TRANSCRIBE_SESSION_POLICY: Final = _policy(
    _allow("TranscribeLiteLLM", ("transcribe:*",)),
    _allow("TranscribeMediaLiteLLM", ("s3:GetObject",)),
    _allow(
        "TranscribeOutputLiteLLM",
        ("s3:PutObject",),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_CalledViaLastEquals({"aws:CalledViaLast": "transcribe.amazonaws.com"}),
        ),
    ),
    _allow(
        "TranscribeDataAccessRoleLiteLLM",
        ("iam:PassRole",),
        _SessionPolicyCondition(
            Bool=_SECURE_TRANSPORT,
            StringEquals=_PassedToServiceEquals({"iam:PassedToService": "transcribe.amazonaws.com"}),
        ),
    ),
    _encryption_through("s3.*", _OBJECT_ENCRYPTION),
)
