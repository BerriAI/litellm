import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
    PassthroughEndpointRouter,
)
from litellm.types.utils import CredentialItem


@pytest.fixture(autouse=True)
def isolated_credential_list(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)


def _credential(name: str, api_key: str) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={"api_key": api_key},
        credential_info={},
    )


def _flagged_deployment(model: str, **litellm_params) -> dict:
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, "use_in_pass_through": True, **litellm_params},
    }


def _passthrough_router(llm_router: litellm.Router | None) -> PassthroughEndpointRouter:
    return PassthroughEndpointRouter(llm_router_getter=lambda: llm_router)


def test_credential_loaded_after_deployment_registration_still_resolves():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_openai")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) is None

    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-loaded-after-boot")])

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None)
        == "sk-loaded-after-boot"
    )


def test_credential_rotation_is_reflected_without_deployment_update():
    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-before-rotation")])
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_openai")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None)
        == "sk-before-rotation"
    )

    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-after-rotation")])

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None)
        == "sk-after-rotation"
    )


def test_deleted_deployment_stops_serving_its_key(monkeypatch):
    llm_router = litellm.Router(model_list=[_flagged_deployment("openai/gpt-4o", api_key="sk-inline")])
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) == "sk-inline"

    llm_router.set_model_list([])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) == "sk-from-env"
    )


def test_inline_api_key_resolves_without_credential_name():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("anthropic/claude-sonnet-4-5", api_key="sk-ant-inline")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None)
        == "sk-ant-inline"
    )


def test_missing_credential_and_no_inline_key_falls_back_to_env(monkeypatch):
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_deleted")]
    )
    passthrough_router = _passthrough_router(llm_router)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) == "sk-from-env"
    )


def test_deployment_for_other_provider_does_not_match():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("anthropic/claude-sonnet-4-5", api_key="sk-ant-inline")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) is None


def test_unflagged_deployment_does_not_match():
    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-not-flagged"},
            }
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) is None


def test_first_matching_deployment_wins():
    llm_router = litellm.Router(
        model_list=[
            _flagged_deployment("openai/gpt-4o", api_key="sk-first"),
            _flagged_deployment("openai/gpt-4o-mini", api_key="sk-second"),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) == "sk-first"


def test_assemblyai_region_matching():
    llm_router = litellm.Router(
        model_list=[
            _flagged_deployment(
                "assemblyai/best", api_key="sk-eu", api_base="https://api.eu.assemblyai.com"
            ),
            _flagged_deployment("assemblyai/best", api_key="sk-us", api_base="https://api.assemblyai.com"),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="assemblyai", region_name="eu") == "sk-eu"
    assert passthrough_router.get_credentials(custom_llm_provider="assemblyai", region_name=None) == "sk-us"


def test_env_fallback_when_no_router(monkeypatch):
    passthrough_router = _passthrough_router(None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) == "sk-from-env"
    )


def test_returns_none_when_no_router_and_no_env():
    passthrough_router = _passthrough_router(None)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None) is None
