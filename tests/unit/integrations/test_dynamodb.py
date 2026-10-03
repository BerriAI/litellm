import sys
from typing import Final

import litellm
import pytest

from litellm.integrations.dynamodb import DyanmoDBLogger


def test_missing_aws_extra_explains_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ImportError, match=r"litellm\[aws\]"):
        DyanmoDBLogger()


def test_dynamodb_logger_initializes_with_aws_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(litellm, "dynamodb_table_name", "sdk-test-logs")
    logger: Final = DyanmoDBLogger()
    assert logger.table_name == "sdk-test-logs"
