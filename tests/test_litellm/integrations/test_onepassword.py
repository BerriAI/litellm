import sys
import types
from typing import List, Optional

import pytest

from litellm.integrations.onepassword import (
    OnePasswordClient,
    OnePasswordConfig,
    OnePasswordShareError,
    OnePasswordShareResult,
    get_cached_onepassword_client,
    get_onepassword_config,
)

CONFIG = OnePasswordConfig(service_account_token="ops_token", vault_id="vault123")


class RecordingBackend:
    def __init__(self, result: OnePasswordShareResult):
        self._result = result
        self.calls: List[tuple] = []

    async def __call__(
        self,
        config: OnePasswordConfig,
        title: str,
        secret_value: str,
        recipients: Optional[List[str]],
        expire_after: Optional[str],
        one_time_only: bool,
    ) -> OnePasswordShareResult:
        self.calls.append((config, title, secret_value, recipients, expire_after, one_time_only))
        return self._result


@pytest.mark.asyncio
async def test_share_secret_passes_arguments_to_backend():
    result = OnePasswordShareResult(
        share_link="https://share.1password.com/abc",
        item_id="item1",
        item_title="vendor-key",
    )
    backend = RecordingBackend(result)
    client = OnePasswordClient(CONFIG, backend=backend)

    got = await client.share_secret(
        title="vendor-key",
        secret_value="sk-1234",
        recipients=["vendor@example.com"],
        expire_after="SevenDays",
        one_time_only=True,
    )

    assert got is result
    assert backend.calls == [
        (CONFIG, "vendor-key", "sk-1234", ["vendor@example.com"], "SevenDays", True)
    ]


@pytest.mark.asyncio
async def test_share_secret_rejects_invalid_expiry_before_calling_backend():
    backend = RecordingBackend(
        OnePasswordShareResult(share_link="x", item_id="i", item_title="t")
    )
    client = OnePasswordClient(CONFIG, backend=backend)

    with pytest.raises(OnePasswordShareError, match="Invalid expire_after"):
        await client.share_secret(title="t", secret_value="sk-1", expire_after="TwoDays")

    assert backend.calls == []


def test_get_onepassword_config_reads_env(monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_tok")
    monkeypatch.setenv("OP_VAULT_ID", "vault-xyz")

    config = get_onepassword_config()

    assert config.service_account_token == "ops_tok"
    assert config.vault_id == "vault-xyz"


def test_get_onepassword_config_falls_back_to_op_vault(monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_tok")
    monkeypatch.delenv("OP_VAULT_ID", raising=False)
    monkeypatch.setenv("OP_VAULT", "legacy-vault")

    assert get_onepassword_config().vault_id == "legacy-vault"


def test_get_onepassword_config_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.delenv("OP_VAULT_ID", raising=False)
    monkeypatch.delenv("OP_VAULT", raising=False)

    with pytest.raises(OnePasswordShareError) as exc:
        get_onepassword_config()

    message = str(exc.value)
    assert "OP_SERVICE_ACCOUNT_TOKEN" in message
    assert "OP_VAULT_ID" in message


def test_get_cached_client_reuses_and_rebuilds(monkeypatch):
    from litellm.integrations import onepassword as op_module

    op_module._cached_client_for.cache_clear()

    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "tok-a")
    monkeypatch.setenv("OP_VAULT_ID", "vault-a")
    first = get_cached_onepassword_client()
    second = get_cached_onepassword_client()
    assert first is second

    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "tok-b")
    third = get_cached_onepassword_client()
    assert third is not first


class _FakeItem:
    def __init__(self, item_id: str):
        self.id = item_id


class _FakeShares:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    async def get_account_policy(self, vault_id, item_id):
        self._recorder["policy_args"] = (vault_id, item_id)
        return "policy-object"

    async def validate_recipients(self, policy, recipients):
        self._recorder["validate_args"] = (policy, recipients)
        return recipients

    async def create(self, item, policy, params):
        self._recorder["create_share_args"] = (item, policy, params)
        return "https://share.1password.com/xyz"


class _FakeItems:
    def __init__(self, recorder: dict):
        self._recorder = recorder
        self.shares = _FakeShares(recorder)

    async def create(self, params):
        self._recorder["create_item_params"] = params
        return _FakeItem("item-42")


class _FakeClient:
    def __init__(self, recorder: dict):
        self.items = _FakeItems(recorder)


def _install_fake_sdk(monkeypatch):
    onepassword_pkg = types.ModuleType("onepassword")
    types_mod = types.ModuleType("onepassword.types")

    class ItemCategory:
        APICREDENTIALS = "API_CREDENTIAL"

    class ItemFieldType:
        CONCEALED = "CONCEALED"

    class ItemField:
        def __init__(self, id, title, field_type, value):
            self.id = id
            self.title = title
            self.field_type = field_type
            self.value = value

    class ItemCreateParams:
        def __init__(self, category, vault_id, title, fields):
            self.category = category
            self.vault_id = vault_id
            self.title = title
            self.fields = fields

    class ItemShareParams:
        def __init__(self, recipients, expire_after, one_time_only):
            self.recipients = recipients
            self.expire_after = expire_after
            self.one_time_only = one_time_only

    types_mod.ItemCategory = ItemCategory
    types_mod.ItemFieldType = ItemFieldType
    types_mod.ItemField = ItemField
    types_mod.ItemCreateParams = ItemCreateParams
    types_mod.ItemShareParams = ItemShareParams
    onepassword_pkg.types = types_mod

    monkeypatch.setitem(sys.modules, "onepassword", onepassword_pkg)
    monkeypatch.setitem(sys.modules, "onepassword.types", types_mod)
    return types_mod


@pytest.mark.asyncio
async def test_share_via_sdk_creates_item_and_share_link(monkeypatch):
    from litellm.integrations import _onepassword_sdk_adapter as adapter

    types_mod = _install_fake_sdk(monkeypatch)
    recorder: dict = {}

    async def fake_authenticate(config):
        assert config is CONFIG
        return _FakeClient(recorder)

    result = await adapter.share_via_sdk(
        CONFIG,
        "vendor-key",
        "sk-secret",
        ["vendor@example.com"],
        "OneDay",
        True,
        authenticate=fake_authenticate,
    )

    assert result.share_link == "https://share.1password.com/xyz"
    assert result.item_id == "item-42"
    assert result.item_title == "vendor-key"
    assert result.expire_after == "OneDay"
    assert result.one_time_only is True

    created = recorder["create_item_params"]
    assert created.vault_id == "vault123"
    assert created.category == types_mod.ItemCategory.APICREDENTIALS
    assert created.fields[0].id == "credential"
    assert created.fields[0].value == "sk-secret"

    assert recorder["policy_args"] == ("vault123", "item-42")
    assert recorder["validate_args"] == ("policy-object", ["vendor@example.com"])
    share_params = recorder["create_share_args"][2]
    assert share_params.recipients == ["vendor@example.com"]
    assert share_params.one_time_only is True


@pytest.mark.asyncio
async def test_share_via_sdk_skips_recipient_validation_when_none(monkeypatch):
    from litellm.integrations import _onepassword_sdk_adapter as adapter

    _install_fake_sdk(monkeypatch)
    recorder: dict = {}

    async def fake_authenticate(config):
        return _FakeClient(recorder)

    await adapter.share_via_sdk(
        CONFIG, "t", "sk-secret", None, None, False, authenticate=fake_authenticate
    )

    assert "validate_args" not in recorder
    assert recorder["create_share_args"][2].recipients is None


@pytest.mark.asyncio
async def test_share_via_sdk_wraps_sdk_errors(monkeypatch):
    from litellm.integrations import _onepassword_sdk_adapter as adapter

    _install_fake_sdk(monkeypatch)

    async def fake_authenticate(config):
        class _Boom:
            class items:
                @staticmethod
                async def create(params):
                    raise RuntimeError("sdk exploded")

        return _Boom()

    with pytest.raises(OnePasswordShareError, match="sdk exploded"):
        await adapter.share_via_sdk(
            CONFIG, "t", "sk-secret", None, None, False, authenticate=fake_authenticate
        )


@pytest.mark.asyncio
async def test_authenticate_raises_clear_error_when_sdk_missing(monkeypatch):
    from litellm.integrations import _onepassword_sdk_adapter as adapter

    monkeypatch.setitem(sys.modules, "onepassword", None)
    monkeypatch.setitem(sys.modules, "onepassword.client", None)

    with pytest.raises(OnePasswordShareError, match="onepassword-sdk"):
        await adapter._authenticate(CONFIG)
