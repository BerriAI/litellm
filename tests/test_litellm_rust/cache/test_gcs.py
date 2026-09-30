from types import SimpleNamespace
from typing import Final

import pytest

from litellm.caching.caching import Cache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import CacheTestResolver, activate_native, native_runtime
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    (("bucket_name", "other"), ("key_prefix", "other/"), ("path_service_account", "other.json")),
)
def test_selected_gcs_runtime_declines_backend_configuration_changes(
    monkeypatch: pytest.MonkeyPatch, attribute: str, replacement: str
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    facade: Final = activate_native(Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/"))
    selected: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    assert selected.resolve().kind == "native"
    with rebound(facade.cache, attribute, replacement):
        with pytest.raises(_native.RustBridgeDeclined):
            selected.resolve()
    assert selected.resolve().kind == "native"


async def test_gcs_runtime_flush_is_a_no_op_and_ping_is_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    runtime: Final = native_runtime(Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket"))
    await runtime.async_flush()
    with pytest.raises(NotImplementedError):
        await runtime.ping()


def test_gcs_runtime_declines_missing_bucket_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    with pytest.raises(_native.RustBridgeDeclined, match="requires a configured bucket name"):
        native_runtime(Cache(type=LiteLLMCacheType.GCS))
