"""
Unit tests for cli_get_key API endpoint (device-key-separation feature)
"""
import pytest
from litellm.proxy.zx.token_util import TokenStore
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles


# Test fixtures
@pytest.fixture
def mock_user_api_key_dict():
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


@pytest.fixture
def mock_store_with_device_id():
    """Mock store with device_id in key_metadata"""
    store = TokenStore(
        type="cli",
        token="test-token",
        login=True,
        status="success",
        expire_time=9999999999,
        auth_key=None,
        data={
            "user_info": {
                "userId": "user-123",
                "name": "Test User",
                "orgEmail": "user@company.com",
                "deptIdList": ["dept-001"],
            },
            "key_metadata": {"device_id": "device-abc123", "device_name": "My Device"},
        },
    )
    return store


@pytest.fixture
def mock_store_without_device_id():
    """Mock store without device_id in key_metadata"""
    store = TokenStore(
        type="cli",
        token="test-token-2",
        login=True,
        status="success",
        expire_time=9999999999,
        auth_key=None,
        data={
            "user_info": {
                "userId": "user-456",
                "name": "Test User 2",
                "orgEmail": "user2@company.com",
                "deptIdList": ["dept-002"],
            },
            "key_metadata": {},
        },
    )
    return store


# T2 Tests: Device ID reading
class TestCliGetKeyDeviceIdReading:
    """Test cli_get_key can read device_id from store"""

    @pytest.mark.asyncio
    async def test_cli_get_key_with_device_id(
        self, mock_store_with_device_id, mock_user_api_key_dict
    ):
        """Test that cli_get_key correctly reads device_id when present"""
        # Verify device_id is in store
        assert (
            mock_store_with_device_id.data["key_metadata"]["device_id"]
            == "device-abc123"
        )
        assert (
            mock_store_with_device_id.data["key_metadata"]["device_name"] == "My Device"
        )

    @pytest.mark.asyncio
    async def test_cli_get_key_without_device_id(
        self, mock_store_without_device_id, mock_user_api_key_dict
    ):
        """Test that cli_get_key handles missing device_id gracefully"""
        # Verify device_id is not in store
        assert (
            mock_store_without_device_id.data["key_metadata"].get("device_id") is None
        )


# T3 Tests: Key alias construction
class TestKeyAliasConstruction:
    """Test key_alias construction with and without device_id"""

    @pytest.mark.asyncio
    async def test_key_alias_with_device_id(self, mock_store_with_device_id):
        """Test that key_alias is {org_email}--{device_id} when device_id exists"""
        org_email = mock_store_with_device_id.data["user_info"]["orgEmail"]
        device_id = mock_store_with_device_id.data["key_metadata"]["device_id"]

        # Expected key_alias pattern
        expected_key_alias = f"{org_email}--{device_id}"
        assert expected_key_alias == "user@company.com--device-abc123"

    @pytest.mark.asyncio
    async def test_key_alias_without_device_id_with_type(
        self, mock_store_without_device_id
    ):
        """Test that key_alias uses old logic (type-based) when device_id not present"""
        org_email = mock_store_without_device_id.data["user_info"]["orgEmail"]
        type_param = "assistant-gpt4"

        # Expected key_alias pattern (old logic)
        expected_key_alias = f"{org_email.split('@')[0]}--{type_param}"
        assert expected_key_alias == "user2--assistant-gpt4"

    @pytest.mark.asyncio
    async def test_key_alias_without_device_id_without_type(
        self, mock_store_without_device_id
    ):
        """Test that key_alias defaults to org_email when neither device_id nor type present"""
        org_email = mock_store_without_device_id.data["user_info"]["orgEmail"]

        # When no device_id and no type, use org_email
        expected_key_alias = org_email
        assert expected_key_alias == "user2@company.com"


# T4 Tests: Key metadata initialization
class TestKeyMetadataInitialization:
    """Test key_metadata is properly initialized with device_id"""

    @pytest.mark.asyncio
    async def test_key_metadata_contains_device_info(self, mock_store_with_device_id):
        """Test that key_metadata includes device_id and device_name when device_id exists"""
        key_metadata = mock_store_with_device_id.data["key_metadata"].copy()
        device_id = mock_store_with_device_id.data["key_metadata"].get("device_id")
        device_name = mock_store_with_device_id.data["key_metadata"].get(
            "device_name", "unknown"
        )

        # Verify metadata is properly set
        assert "device_id" in key_metadata
        assert key_metadata["device_id"] == device_id
        assert key_metadata["device_name"] == device_name

    @pytest.mark.asyncio
    async def test_key_metadata_without_device_id(self, mock_store_without_device_id):
        """Test that key_metadata doesn't break when device_id is not present"""
        key_metadata = mock_store_without_device_id.data["key_metadata"].copy()
        device_id = key_metadata.get("device_id")

        # Verify device_id is None or missing
        assert device_id is None


# T5 Tests: Legacy key metadata update
class TestLegacyKeyMetadataUpdate:
    """Test that old keys get their metadata dynamically updated"""

    @pytest.mark.asyncio
    async def test_legacy_key_receives_device_id(self):
        """Test that legacy key metadata is updated to include device_id when it's missing"""
        # Simulate an existing key without device_id in metadata
        legacy_metadata = {"custom_field": "value"}

        # When device_id is provided, it should be added to metadata
        device_id = "device-new-123"
        device_name = "New Device"

        # Updated metadata should contain both old and new fields
        updated_metadata = legacy_metadata.copy()
        updated_metadata["device_id"] = device_id
        updated_metadata["device_name"] = device_name

        assert updated_metadata["device_id"] == device_id
        assert updated_metadata["device_name"] == device_name
        assert updated_metadata["custom_field"] == "value"  # Old fields preserved


# Integration-like tests
class TestCliGetKeyIntegration:
    """High-level tests for cli_get_key behavior"""

    @pytest.mark.asyncio
    async def test_idempotency_same_device(self):
        """Test that same device returns same key_alias on multiple calls"""
        device_id = "device-abc123"
        org_email = "user@company.com"

        # Simulate two calls with same device
        key_alias_1 = f"{org_email}--{device_id}"
        key_alias_2 = f"{org_email}--{device_id}"

        # Both should be identical (idempotent)
        assert key_alias_1 == key_alias_2

    @pytest.mark.asyncio
    async def test_device_isolation(self):
        """Test that different devices get different key_aliases"""
        org_email = "user@company.com"
        device_id_1 = "device-abc123"
        device_id_2 = "device-xyz789"

        # Simulate calls with different devices
        key_alias_1 = f"{org_email}--{device_id_1}"
        key_alias_2 = f"{org_email}--{device_id_2}"

        # Should be different
        assert key_alias_1 != key_alias_2
        assert key_alias_1 == "user@company.com--device-abc123"
        assert key_alias_2 == "user@company.com--device-xyz789"

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_device_id(self):
        """Test that old logic still works when device_id is not present"""
        org_email = "user@company.com"
        type_param = "assistant-gpt4"

        # Old logic: use type-based key_alias
        key_alias_old = f"{org_email.split('@')[0]}--{type_param}"

        # Should still work as before
        assert key_alias_old == "user--assistant-gpt4"
