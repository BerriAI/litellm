import logging
import os
from typing import Final, cast

import pytest
from dotenv import load_dotenv

load_dotenv()

from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


def test_encrypt_decrypt_with_master_key():
    setattr(proxy_server, "master_key", "sk-1234")
    assert decrypt_value_helper(encrypt_value_helper("test"), key="test_key") == "test"
    assert decrypt_value_helper(encrypt_value_helper(10), key="test_key") == 10
    assert decrypt_value_helper(encrypt_value_helper(True), key="test_key") is True
    assert decrypt_value_helper(encrypt_value_helper(None), key="test_key") is None
    assert decrypt_value_helper(encrypt_value_helper({"rpm": 10}), key="test_key") == {"rpm": 10}

    # encryption should actually occur for strings
    assert encrypt_value_helper("test") != "test"


def test_encrypt_decrypt_with_salt_key():
    os.environ["LITELLM_SALT_KEY"] = "sk-salt-key2222"
    assert decrypt_value_helper(encrypt_value_helper("test"), key="test_key") == "test"
    assert decrypt_value_helper(encrypt_value_helper(10), key="test_key") == 10
    assert decrypt_value_helper(encrypt_value_helper(True), key="test_key") is True
    assert decrypt_value_helper(encrypt_value_helper(None), key="test_key") is None
    assert decrypt_value_helper(encrypt_value_helper({"rpm": 10}), key="test_key") == {"rpm": 10}

    # encryption should actually occur for strings
    assert encrypt_value_helper("test") != "test"

    os.environ.pop("LITELLM_SALT_KEY", None)


def test_encrypt_value_helper_does_not_log_invalid_value(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    secret: Final[str] = "must-not-be-logged"
    value: Final[dict[str, str]] = {"token": secret}

    encrypt_value_helper(cast(str, value))

    messages: Final = tuple(record.getMessage() for record in caplog.records)
    assert any("Invalid value type passed to encrypt_value" in message for message in messages)
    assert all(secret not in message for message in messages)
