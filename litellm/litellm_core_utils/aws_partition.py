import re
from types import MappingProxyType
from typing import Final, NamedTuple


class AwsPartition(NamedTuple):
    partition: str
    dns_suffix: str


_COMMERCIAL_PARTITION: Final = AwsPartition(partition="aws", dns_suffix="amazonaws.com")

_PARTITIONS_BY_REGION_PREFIX: Final = MappingProxyType(
    {
        "cn-": AwsPartition(partition="aws-cn", dns_suffix="amazonaws.com.cn"),
        "us-gov-": AwsPartition(partition="aws-us-gov", dns_suffix="amazonaws.com"),
        "us-isob-": AwsPartition(partition="aws-iso-b", dns_suffix="sc2s.sgov.gov"),
        "us-isof-": AwsPartition(partition="aws-iso-f", dns_suffix="csp.hci.ic.gov"),
        "us-iso-": AwsPartition(partition="aws-iso", dns_suffix="c2s.ic.gov"),
        "eu-isoe-": AwsPartition(partition="aws-iso-e", dns_suffix="cloud.adc-e.uk"),
    }
)

_BEDROCK_ARN_PATTERN: Final = re.compile(r"arn:aws(?:-[a-z0-9-]+)?:bedrock")
_BEDROCK_ARN_PREFIX_PATTERN: Final = re.compile(r"\Aarn:aws(?:-[a-z0-9-]+)?:bedrock:")
_AWS_ARN_PATTERN: Final = re.compile(r"arn:aws(?:-[a-z0-9-]+)?:")


def get_aws_partition(aws_region_name: str | None) -> AwsPartition:
    if not aws_region_name:
        return _COMMERCIAL_PARTITION
    return next(
        (partition for prefix, partition in _PARTITIONS_BY_REGION_PREFIX.items() if aws_region_name.startswith(prefix)),
        _COMMERCIAL_PARTITION,
    )


def get_aws_dns_suffix(aws_region_name: str | None) -> str:
    return get_aws_partition(aws_region_name).dns_suffix


def get_aws_arn_prefix(aws_region_name: str | None) -> str:
    return f"arn:{get_aws_partition(aws_region_name).partition}:"


def contains_bedrock_arn(value: str) -> bool:
    return _BEDROCK_ARN_PATTERN.search(value) is not None


def is_bedrock_arn(value: str) -> bool:
    return _BEDROCK_ARN_PREFIX_PATTERN.match(value) is not None


def contains_aws_arn(value: str) -> bool:
    return _AWS_ARN_PATTERN.search(value) is not None
