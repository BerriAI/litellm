import ast
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote, urlparse

import pytest
from botocore.credentials import Credentials

import litellm
from litellm.integrations.s3_v2 import S3Logger
from litellm.litellm_core_utils.aws_partition import (
    AwsPartition,
    contains_aws_arn,
    contains_bedrock_arn,
    get_aws_arn_prefix,
    get_aws_dns_suffix,
    get_aws_partition,
    is_bedrock_arn,
)
from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
from litellm.llms.bedrock.chat.invoke_agent.transformation import AmazonInvokeAgentConfig
from litellm.llms.bedrock.common_utils import init_bedrock_client
from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
from litellm.llms.bedrock.rerank.handler import BedrockRerankHandler
from litellm.llms.bedrock.vector_stores.transformation import BedrockVectorStoreConfig
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig
from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.proxy.auth.rds_iam_token import init_rds_client
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

STATIC_AWS_CREDENTIALS: Final = MappingProxyType(
    {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}
)


@pytest.mark.parametrize(
    "region,partition,dns_suffix",
    [
        ("us-east-1", "aws", "amazonaws.com"),
        ("eu-central-1", "aws", "amazonaws.com"),
        ("ap-southeast-1", "aws", "amazonaws.com"),
        ("sa-east-1", "aws", "amazonaws.com"),
        ("cn-north-1", "aws-cn", "amazonaws.com.cn"),
        ("cn-northwest-1", "aws-cn", "amazonaws.com.cn"),
        ("us-gov-west-1", "aws-us-gov", "amazonaws.com"),
        ("us-gov-east-1", "aws-us-gov", "amazonaws.com"),
        ("us-iso-east-1", "aws-iso", "c2s.ic.gov"),
        ("us-isob-east-1", "aws-iso-b", "sc2s.sgov.gov"),
        ("us-isof-south-1", "aws-iso-f", "csp.hci.ic.gov"),
        ("eu-isoe-west-1", "aws-iso-e", "cloud.adc-e.uk"),
        (None, "aws", "amazonaws.com"),
        ("", "aws", "amazonaws.com"),
    ],
)
def test_partition_lookup(region: str | None, partition: str, dns_suffix: str) -> None:
    assert get_aws_partition(region) == AwsPartition(partition=partition, dns_suffix=dns_suffix)
    assert get_aws_dns_suffix(region) == dns_suffix
    assert get_aws_arn_prefix(region) == f"arn:{partition}:"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("arn:aws:bedrock:us-west-2:123456789012:foundation-model/anthropic.claude-3", True),
        ("arn:aws-cn:bedrock:cn-north-1:123456789012:inference-profile/p", True),
        ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:foundation-model/m", True),
        ("bedrock/arn:aws-cn:bedrock:cn-north-1:123456789012:application-inference-profile/p", True),
        ("arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/r", True),
        ("anthropic.claude-3", False),
        ("arn:aws:iam::123456789012:role/foo", False),
    ],
)
def test_contains_bedrock_arn(value: str, expected: bool) -> None:
    assert contains_bedrock_arn(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/j", True),
        ("arn:aws-cn:bedrock:cn-north-1:123456789012:model-invocation-job/j", True),
        ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:model-invocation-job/j", True),
        ("abc1234567", False),
        ("bedrock/arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/j", False),
        ("arn:aws:iam::123456789012:role/foo", False),
    ],
)
def test_is_bedrock_arn(value: str, expected: bool) -> None:
    assert is_bedrock_arn(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("model/arn:aws:bedrock:us-east-1:123456789012:foundation-model/m/converse", True),
        ("model/arn:aws-cn:bedrock:cn-north-1:123456789012:foundation-model/m/converse", True),
        ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/p", True),
        ("model/anthropic.claude-3/converse", False),
        ("arnaws:bedrock", False),
    ],
)
def test_contains_aws_arn(value: str, expected: bool) -> None:
    assert contains_aws_arn(value) is expected


def _agentcore_model(region: str) -> str:
    return f"agentcore/{get_aws_arn_prefix(region)}bedrock-agentcore:{region}:111122223333:runtime/my-agent"


def _s3_object_url(region: str) -> str:
    logger = S3Logger.__new__(S3Logger)
    logger.s3_endpoint_url = None
    logger.s3_bucket_name = "audit-bucket"
    logger.s3_region_name = region
    return logger._build_object_url("2025-01-01/key.json")


def _bedrock_job_arn(region: str) -> str:
    return f"{get_aws_arn_prefix(region)}bedrock:{region}:111122223333:model-invocation-job/abc1234567"


def _bedrock_files_upload_url(region: str) -> str:
    return BedrockFilesConfig().get_complete_file_url(
        api_base=None,
        api_key=None,
        model="amazon.nova-pro-v1:0",
        optional_params={},
        litellm_params={"s3_bucket_name": "batch-bucket", "s3_region_name": region},
        data={"file": ("batch.jsonl", b"{}", "application/jsonl"), "purpose": "batch"},
    )


def _bedrock_files_download_url(region: str) -> str:
    return (
        BedrockFilesConfig()
        ._s3_request_target(optional_params={}, litellm_params={"s3_region_name": region})
        .endpoint_url
    )


def _bedrock_guardrail_url(region: str) -> str:
    guardrail = BedrockGuardrail(guardrailIdentifier="guardrail-id", guardrailVersion="1")
    return guardrail._prepare_request(
        credentials=Credentials("test-key", "test-secret"),
        data={"source": "INPUT", "content": []},
        optional_params={},
        aws_region_name=region,
    ).url


def _secrets_manager_url(region: str) -> str:
    endpoint_url, _headers, _body = AWSSecretsManagerV2(aws_region_name=region)._prepare_request(
        action="GetSecretValue",
        secret_name="my-secret",
        optional_params=dict(STATIC_AWS_CREDENTIALS),
    )
    return endpoint_url


ENDPOINT_BUILDERS: Final = {
    "bedrock_runtime_default": lambda region: BaseAWSLLM()._select_default_endpoint_url("runtime", region),
    "bedrock_agent_default": lambda region: BaseAWSLLM()._select_default_endpoint_url("agent", region),
    "bedrock_agentcore_default": lambda region: BaseAWSLLM()._select_default_endpoint_url("agentcore", region),
    "bedrock_get_runtime_endpoint": lambda region: BaseAWSLLM().get_runtime_endpoint(None, None, region)[0],
    "bedrock_legacy_client": lambda region: init_bedrock_client(
        region_name=region,
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    ).meta.endpoint_url,
    "bedrock_batches": lambda region: BedrockBatchesConfig().get_complete_batch_url(
        api_base=None,
        api_key=None,
        model="anthropic.claude-3",
        optional_params={"aws_region_name": region},
        litellm_params={},
        data={"input_file_id": "s3://bucket/key.jsonl"},
    ),
    "bedrock_batches_retrieve": lambda region: BedrockBatchesConfig().transform_retrieve_batch_request(
        batch_id=_bedrock_job_arn(region),
        optional_params=dict(STATIC_AWS_CREDENTIALS),
        litellm_params={},
    )["url"],
    "bedrock_files_upload": _bedrock_files_upload_url,
    "bedrock_files_download": _bedrock_files_download_url,
    "bedrock_agentcore_invoke": lambda region: AmazonAgentCoreConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model=_agentcore_model(region),
        optional_params={},
        litellm_params={},
    ),
    "bedrock_invoke_agent": lambda region: AmazonInvokeAgentConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="agent/AGENT123/ALIAS456",
        optional_params={"aws_region_name": region},
        litellm_params={},
    ),
    "bedrock_guardrail_apply": _bedrock_guardrail_url,
    "bedrock_rerank": lambda region: BedrockRerankHandler()._prepare_request(
        model="amazon.rerank-v1:0",
        api_base=None,
        extra_headers=None,
        data={"queries": [], "sources": []},
        optional_params={"aws_region_name": region, **STATIC_AWS_CREDENTIALS},
    )["endpoint_url"],
    "bedrock_knowledgebase_search": lambda region: BedrockVectorStoreConfig().get_complete_url(
        api_base=None, litellm_params={"aws_region_name": region}
    ),
    "secrets_manager": _secrets_manager_url,
    "rds_iam_client": lambda region: (
        init_rds_client(
            aws_region_name=region,
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        ).meta.endpoint_url
    ),
    "polly": lambda region: AWSPollyTextToSpeechConfig().get_complete_url(
        model="polly/neural",
        api_base=None,
        litellm_params={"aws_region_name": region},
    ),
    "sagemaker_chat": lambda region: SagemakerChatConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="my-endpoint",
        optional_params={"aws_region_name": region},
        litellm_params={},
        stream=False,
    ),
    "sagemaker_chat_stream": lambda region: SagemakerChatConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="my-endpoint",
        optional_params={"aws_region_name": region},
        litellm_params={},
        stream=True,
    ),
    "sagemaker_completion": lambda region: (
        SagemakerLLM()
        ._prepare_request(
            credentials=Credentials("test-key", "test-secret"),
            model="my-endpoint",
            data={},
            messages=[],
            litellm_params={},
            optional_params={},
            aws_region_name=region,
        )
        .url
    ),
    "s3_object_url": _s3_object_url,
}


@pytest.fixture(autouse=True)
def _clear_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in ("AWS_BEDROCK_RUNTIME_ENDPOINT", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_REGION_NAME"):
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize("region", ["cn-north-1", "cn-northwest-1"])
@pytest.mark.parametrize("builder_name", sorted(ENDPOINT_BUILDERS))
def test_every_endpoint_builder_respects_cn_partition(builder_name: str, region: str) -> None:
    url = ENDPOINT_BUILDERS[builder_name](region)
    hostname = urlparse(url).hostname
    assert hostname is not None
    assert hostname.endswith(".amazonaws.com.cn"), url
    assert not hostname.endswith("amazonaws.com"), url
    assert "arn:aws:" not in url, url


@pytest.mark.parametrize("region", ["us-east-1", "us-gov-west-1"])
@pytest.mark.parametrize("builder_name", sorted(ENDPOINT_BUILDERS))
def test_every_endpoint_builder_keeps_amazonaws_com_outside_cn(builder_name: str, region: str) -> None:
    url = ENDPOINT_BUILDERS[builder_name](region)
    hostname = urlparse(url).hostname
    assert hostname is not None
    assert hostname.endswith(".amazonaws.com"), url


@pytest.mark.parametrize("region", ["us-gov-west-1", "us-gov-east-1"])
@pytest.mark.parametrize("builder_name", sorted(ENDPOINT_BUILDERS))
def test_every_endpoint_builder_respects_us_gov_partition(builder_name: str, region: str) -> None:
    url = unquote(ENDPOINT_BUILDERS[builder_name](region))
    hostname = urlparse(url).hostname
    assert hostname is not None
    assert hostname.endswith(f".{region}.amazonaws.com"), url
    assert "arn:aws:" not in url, url
    if "arn:" in url:
        assert "arn:aws-us-gov:" in url, url


def _fstring_literal_offenders(needle: str) -> list[str]:
    litellm_root = Path(litellm.__file__).parent
    return [
        f"{path.relative_to(litellm_root)}: {part.value!r}"
        for path in sorted(litellm_root.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.JoinedStr)
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str) and needle in part.value
    ]


def test_no_fstring_hardcodes_the_commercial_dns_suffix() -> None:
    assert _fstring_literal_offenders("amazonaws.com") == []


def test_no_fstring_hardcodes_the_commercial_arn_prefix() -> None:
    assert _fstring_literal_offenders("arn:aws:") == []
