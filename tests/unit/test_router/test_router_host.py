import subprocess
import sys

import pytest

import litellm.router_host as router_host_module
from litellm.router_host import RouterHost, router_host


def test_router_uses_the_proxy_host_without_importing_the_proxy_first():
    # A fresh interpreter, so no earlier test has imported litellm.proxy.
    script = (
        "import sys\n"
        "from litellm.router import Router\n"
        "from litellm.router_host import router_host\n"
        "assert 'litellm.proxy' not in sys.modules, 'importing the router imported the proxy'\n"
        "print(type(router_host()).__name__)\n"
    )
    result = subprocess.run([sys.executable, "-P", "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ProxyRouterHost"


@pytest.fixture(autouse=True)
def nothing_installed():
    saved = router_host_module._installed[0]
    router_host_module._installed[0] = None
    yield
    router_host_module._installed[0] = saved


def test_router_falls_back_to_the_default_host_when_the_proxy_cannot_be_imported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "litellm.proxy._router_host", None)

    assert type(router_host()) is RouterHost


def test_router_uses_the_proxy_host_when_the_proxy_was_imported_before():
    from litellm.proxy._router_host import ProxyRouterHost

    assert type(router_host()) is ProxyRouterHost


def test_an_installed_host_is_kept():
    host = RouterHost()
    router_host_module.install_router_host(host)

    assert router_host() is host


def test_default_host_refuses_member_auto_routers():
    import asyncio

    from litellm.exceptions import PermissionDeniedError

    host = RouterHost()
    member = {"model_name": "team-auto", "model_info": {"member_auto_router": True}}
    with pytest.raises(PermissionDeniedError, match="team access checks"):
        asyncio.run(host.authorize_strategy_inference(member, {}, None))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "deployment", [None, {"model_name": "auto", "model_info": {}}, {"model_name": "auto", "model_info": None}]
)
def test_default_host_allows_other_strategies(deployment):
    import asyncio

    assert asyncio.run(RouterHost().authorize_strategy_inference(deployment, {}, None)) is None  # pyright: ignore[reportArgumentType]


def test_default_host_keeps_nothing_and_compresses_nothing():
    import asyncio

    host = RouterHost()
    assert host.add_vertex_pass_through_credentials("p", "l", None) is None
    assert host.compression_policy(None, "auto", {}, ()) is None
    assert host.model_hop_compression_armed() is False
    assert asyncio.run(host.messages_for_routing(None, [{"role": "user", "content": "hi"}], {})) is None


_TIER_KEYWORD = "zzsimpletier"


def _complexity_router_reply(metadata: dict[str, object] | None) -> str:
    import asyncio

    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": tier, "litellm_params": {"model": f"openai/{tier}", "api_key": "k", "mock_response": tier}}
            for tier in ("cheap", "smart")
        ]
        + [
            {
                "model_name": "auto",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "heuristic",
                        "keyword_tier_rules": [{"keywords": [_TIER_KEYWORD], "tier": "SIMPLE"}],
                        "tiers": {"SIMPLE": "cheap", "MEDIUM": "smart", "COMPLEX": "smart", "REASONING": "smart"},
                    },
                },
            }
        ]
    )
    extra = {} if metadata is None else {"metadata": metadata}
    response = asyncio.run(
        router.acompletion(model="auto", messages=[{"role": "user", "content": _TIER_KEYWORD}], **extra)
    )
    return response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming reply


@pytest.mark.parametrize("disallowed_models", [None, [], ["smart"]], ids=["no-key", "no-deny-list", "other-tier-denied"])
def test_a_complexity_router_serves_its_selected_tier_when_the_key_does_not_disallow_it(
    disallowed_models: list[str] | None,
):
    from litellm.proxy._types import UserAPIKeyAuth

    metadata = (
        None
        if disallowed_models is None
        else {"user_api_key_auth": UserAPIKeyAuth(api_key="sk", disallowed_models=disallowed_models)}
    )

    assert _complexity_router_reply(metadata) == "cheap"


@pytest.mark.parametrize("deny_entry", ["cheap", "che*"])
def test_a_complexity_router_cannot_send_a_key_to_a_tier_it_disallows(deny_entry: str):
    from litellm.proxy._types import ModelAccessDeniedProxyException, ProxyErrorTypes, UserAPIKeyAuth

    key = UserAPIKeyAuth(api_key="sk", disallowed_models=[deny_entry])

    with pytest.raises(ModelAccessDeniedProxyException) as denied:
        _complexity_router_reply({"user_api_key_auth": key})

    assert denied.value.type == ProxyErrorTypes.key_model_access_denied
    assert denied.value.code == "403"
