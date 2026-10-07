import asyncio
import importlib
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.proxy._types import KeyManagementSystem
from litellm.secret_managers.main import get_secret
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class MockSecretClient:
    def get_secret(self, secret_name):
        return Mock(value="mocked_secret_value")


@pytest.mark.asyncio
async def test_azure_kms():
    """
    Basic asserts that the value from get secret is from Azure Key Vault when Key Management System is Azure Key Vault
    """
    with patch("litellm.secret_manager_client", new=MockSecretClient()):
        litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
        secret = get_secret(secret_name="ishaan-test-key")
        assert secret == "mocked_secret_value"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)
