import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.dataplanes.registry import InvalidDataplaneConfig, build_registry, visible_to
from litellm.proxy.control_plane.settings import DataplaneSpec
from tests.unit.proxy.control_plane.dataplanes.conftest import principal


async def _exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    return subject_token


def _spec(**fields: object) -> DataplaneSpec:
    return DataplaneSpec.model_validate({"url": "https://eu.litellm.test/", **fields})


OIDC = {"token_endpoint": "https://idp.test/token", "client_id": "cp", "client_secret": "s", "audience": "dp"}


def test_registry_exposes_credential_order_and_normalizes_urls() -> None:
    registry = build_registry(
        {"eu": _spec(oidc_exchange=OIDC, service_token={"token": "sk"}), "us": _spec(service_token={"token": "sk"})},
        _exchange,
    )
    eu = registry.get("eu")
    assert eu is not None
    assert (eu.name, eu.url, eu.credential_kinds) == (
        "eu",
        "https://eu.litellm.test",
        ("oidc_exchange", "service_token"),
    )
    assert tuple(dataplane.id for dataplane in registry.all()) == ("eu", "us")
    assert registry.get("apac") is None


def test_dataplane_without_any_credential_fails_at_startup() -> None:
    with pytest.raises(InvalidDataplaneConfig, match="eu"):
        build_registry({"eu": _spec()}, _exchange)


def test_plain_http_token_endpoint_fails_at_startup() -> None:
    with pytest.raises(InvalidDataplaneConfig, match="HTTPS"):
        build_registry({"eu": _spec(oidc_exchange={**OIDC, "token_endpoint": "http://idp.test/token"})}, _exchange)


def test_visible_dataplanes_follow_the_principals_binding() -> None:
    registry = build_registry({name: _spec(service_token={"token": "sk"}) for name in ("eu", "us", "apac")}, _exchange)
    assert tuple(d.id for d in visible_to(registry, principal("session"))) == ("eu", "us", "apac")
    assert tuple(d.id for d in visible_to(registry, principal("session", "a", None, "us", "apac"))) == ("us", "apac")


def test_plain_http_dataplane_needs_an_explicit_opt_in() -> None:
    with pytest.raises(ValidationError, match="unencrypted"):
        DataplaneSpec.model_validate({"url": "http://dp.internal:4000"})
    assert DataplaneSpec.model_validate({"url": "http://dp.internal:4000", "allow_insecure_http": True}).url
    assert DataplaneSpec.model_validate({"url": "https://dp.example.test"}).allow_insecure_http is False
