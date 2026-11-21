"""
Test raise_if_unsafe_secret_name, the shared guard applied before secret_name
reaches a secret manager backend.
"""

import httpx
import pytest

from litellm.secret_managers.base_secret_manager import BaseSecretManager, raise_if_unsafe_secret_name


@pytest.mark.parametrize(
    "secret_name",
    [
        "..",
        "../../../other-app/creds",
        "litellm/../../secret",
        "foo/../bar",
        "foo/..",
        "../foo",
        r"..\secret",
        r"secret\..",
        r"foo\..\bar",
        r"foo\../bar",
        r"foo/..\bar",
        r"..\..\other-app\creds",
        r"litellm\..\..\secret",
        "foo\nbar",
        "foo\rbar",
        "foo\x00bar",
        "foo\x7fbar",
        "foo\x85bar",
        "foo bar",
        "foo bar",
    ],
)
def test_raise_if_unsafe_secret_name_rejects_traversal_and_line_breaks(secret_name):
    with pytest.raises(ValueError, match='Invalid secret_name'):
        raise_if_unsafe_secret_name(secret_name)


@pytest.mark.parametrize(
    "secret_name",
    [
        "plain-alias",
        "my-key-123",
        "prod/my-service-key",
        "team/user@example.com",
        "foo: bar",
        "foo # bar",
        "foo?evil=1",
        "foo#bar",
        "a" * 500,
        "release-1.0..2",
        "my..key",
        "..foo",
        "foo..",
        "v2.0..1-beta",
    ],
)
def test_raise_if_unsafe_secret_name_allows_legitimate_aliases(secret_name):
    raise_if_unsafe_secret_name(secret_name)


class DummySecretManager(BaseSecretManager):
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return "secret-val"

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return "secret-val"

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: dict | list | None = None,
    ) -> dict[str, object]:
        return {"status": "success"}

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = 7,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict:
        return {"status": "deleted"}


@pytest.mark.asyncio
async def test_async_rotate_secret_validates_secret_names():
    manager = DummySecretManager()

    # Reject unsafe current_secret_name with forward and backward slashes
    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret("../bad_current", "valid_new", "value")

    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret(r"..\bad_current", "valid_new", "value")

    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret(r"foo\..\bad_current", "valid_new", "value")

    # Reject unsafe new_secret_name with forward and backward slashes
    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret("valid_current", "../bad_new", "value")

    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret("valid_current", r"..\bad_new", "value")

    with pytest.raises(ValueError, match="Invalid secret_name"):
        await manager.async_rotate_secret("valid_current", r"foo\..\bad_new", "value")

    # Valid secret names succeed
    result = await manager.async_rotate_secret("valid_current", "valid_new", "value")
    assert result == {"status": "success"}
