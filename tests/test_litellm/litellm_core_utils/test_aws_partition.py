import ast
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import pytest

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
from litellm.llms.bedrock.common_utils import init_bedrock_client
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


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
    "bedrock_agentcore_invoke": lambda region: AmazonAgentCoreConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model=_agentcore_model(region),
        optional_params={},
        litellm_params={},
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
