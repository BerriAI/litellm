from typing import Final

from pydantic import SecretStr

from litellm.proxy.control_plane.settings import DataplaneSpec
from litellm.proxy.control_plane.store.crypto import SecretCodec
from litellm.proxy.control_plane.store.models import ENTITY


def test_secrets_roundtrip_with_the_configured_key_not_the_global_environment(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "different-global-key")
    codec: Final = SecretCodec(SecretStr("explicit-store-key"))
    spec: Final = DataplaneSpec.model_validate(
        {
            "url": "https://runtime.test",
            "service_token": {"token": "private-service-credential"},
            "oidc_exchange": {
                "token_endpoint": "https://idp.test/token",
                "client_id": "cp",
                "client_secret": "private-client-credential",
            },
        }
    )
    encrypted: Final = codec.encode(spec)
    wire: Final = ENTITY.dump_json(encrypted).decode()
    assert "private-service-credential" not in wire
    assert "private-client-credential" not in wire
    decoded: Final = codec.decode(encrypted)
    assert decoded.service_token is not None
    assert decoded.oidc_exchange is not None
    assert decoded.service_token.token.get_secret_value() == spec.service_token.token.get_secret_value()
    assert decoded.oidc_exchange.client_secret.get_secret_value() == spec.oidc_exchange.client_secret.get_secret_value()
    assert decoded.url == spec.url


def test_specs_without_secrets_roundtrip_unchanged() -> None:
    codec: Final = SecretCodec(SecretStr("explicit-store-key"))
    spec: Final = DataplaneSpec(url="https://runtime.test")
    assert codec.decode(codec.encode(spec)) == spec
