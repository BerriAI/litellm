import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from fastapi import FastAPI
from openai import AsyncOpenAI
from starlette.requests import Request

import litellm
from litellm.caching.caching import Cache, DualCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.model_offerings import ModelOfferingsManager
from litellm.proxy.offering_router import OfferingAccessGuard, OfferingSnapshotMiddleware
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.utils import CallTypesLiteral


@pytest.mark.parametrize("provider", ("openrouter", "vercel_ai_gateway"))
async def test_native_base_changes_invalidate_connection_inventory_and_preserve_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    variable: Final = "OPENROUTER" if provider == "openrouter" else "VERCEL_AI_GATEWAY"
    first_base: Final = "https://first.test/custom/v1"
    second_base: Final = "https://second.test/custom/v1"
    monkeypatch.setenv(f"{variable}_API_BASE", first_base)
    monkeypatch.setenv(f"{variable}_API_KEY", "ambient-provider-key")
    monkeypatch.setattr(litellm, "api_base", None)
    path: Final = tmp_path / "offerings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {"supplier": {"provider": provider, "api_key": "configured-key"}},
                "offerings": [
                    {
                        "model_name": "selected",
                        "source": "auto",
                        "provider": "supplier",
                        "upstream_model": "org/backend",
                    }
                ],
            }
        )
    )
    responses: Final = iter(
        (
            (first_base, 200),
            (second_base, 401),
            (second_base, 200),
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        expected_base, status = next(responses)
        assert str(request.url) == f"{expected_base}/models"
        assert request.headers["authorization"] == "Bearer configured-key"
        return httpx.Response(status, json={"data": [{"id": "org/backend"}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        original_ids: Final = manager.router.get_model_ids()
        with manager.router.pin_snapshot():
            monkeypatch.setenv(f"{variable}_API_BASE", second_base)
            assert await manager.reload()
            assert manager.router.get_model_ids() == original_ids
            assert manager.router.latest_snapshot().available_models == set()
        assert manager.router.get_model_names() == []
        assert await manager.reload(force_inventory=True)
        assert manager.router.latest_snapshot().available_models == {"selected"}
        assert manager.router.get_model_ids() != original_ids


@pytest.mark.parametrize("catalog_state", ("missing", "known", "different_route_provider"))
async def test_wildcard_listing_tracks_selected_offerings_and_pinned_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, catalog_state: str
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    path: Final = tmp_path / "offerings.yaml"
    selected: Final = {
        "model_name": "openrouter/org/new-model",
        "source": "auto",
        "provider": "supplier",
        "upstream_model": "org/new-model",
        "model_info": {"mode": "chat", "context_window": 4000},
        "provider_policy": {"provider": "openrouter", "only": ["synthetic/fp8"]},
    }
    existing: Final = {
        "model_name": "chatgpt/existing",
        "source": "manual",
        "provider": "secondary",
        "upstream_model": "org/existing",
    }
    other_existing: Final = tuple(
        {**existing, "model_name": name} for name in ("chatgpt/second", "chatgpt/third", "chatgpt/fourth")
    )
    existing_names: Final = frozenset((existing["model_name"], *(row["model_name"] for row in other_existing)))
    denied: Final = {**existing, "model_name": "restricted/other"}
    disabled: Final = {**selected, "model_name": "openrouter/org/disabled", "enabled": False}
    absent: Final = {**selected, "model_name": "openrouter/org/absent", "upstream_model": "org/absent"}
    config: Final = {
        "version": 1,
        "providers": {
            "supplier": {
                "provider": "openrouter",
                "api_base": "https://supplier.test/v1",
                "api_key": "private-fixture-key",
            },
            "secondary": {"provider": "openai", "api_base": "https://secondary.test/v1"},
        },
        "offerings": [existing, *other_existing, denied, disabled, absent],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": "org/new-model"}, {"id": "org/unselected"}]})
        )
    ) as client:
        handler.client = client
        template: Final = Router(
            model_list=[],
            model_group_alias={"openrouter/*": existing["model_name"]}
            if catalog_state == "different_route_provider"
            else {},
        )
        manager: Final = ModelOfferingsManager(path=path, template=template, client=handler, clock=lambda: 0.0)
        assert await manager.reload(initial=True)
        monkeypatch.setattr(proxy_server, "llm_router", manager.router)
        monkeypatch.setattr(proxy_server, "llm_model_list", manager.router.model_list)
        monkeypatch.setattr(proxy_server, "prisma_client", None)
        monkeypatch.setattr(proxy_server, "general_settings", {})
        monkeypatch.setattr(proxy_server, "user_model", None)
        monkeypatch.setattr(litellm, "callbacks", [OfferingAccessGuard(manager.router)])
        monkeypatch.setattr(
            litellm,
            "models_by_provider",
            {
                "openrouter": ["org/catalog-only"] if catalog_state == "missing" else [selected["model_name"]],
                "chatgpt": sorted(existing_names),
                "openai": ["fixture/catalog-only"],
            },
        )
        monkeypatch.setattr(litellm, "check_provider_endpoint", False)
        assert (selected["model_name"] in get_known_models_from_wildcard("openrouter/*")) == (
            catalog_state != "missing"
        )
        wildcard: Final = UserAPIKeyAuth(
            api_key="wildcard", user_role=LitellmUserRoles.INTERNAL_USER, models=["openrouter/*", "chatgpt/*"]
        )
        exact: Final = UserAPIKeyAuth(
            api_key="exact", user_role=LitellmUserRoles.INTERNAL_USER, models=sorted(existing_names)
        )
        denied_key: Final = UserAPIKeyAuth(
            api_key="denied", user_role=LitellmUserRoles.INTERNAL_USER, models=["denied/*"]
        )
        team_key: Final = UserAPIKeyAuth(
            api_key="team", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-one", team_models=["openrouter/*"]
        )
        alias_key: Final = UserAPIKeyAuth(
            api_key="alias",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["shortcut"],
            aliases={"shortcut": selected["model_name"]},
        )
        admin: Final = UserAPIKeyAuth(api_key="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        credentials: Final = MappingProxyType(
            {auth.api_key: auth for auth in (wildcard, exact, denied_key, team_key, alias_key, admin)}
        )
        application: Final = FastAPI()
        application.include_router(proxy_server.router)
        application.add_middleware(OfferingSnapshotMiddleware, router_getter=lambda: manager.router)

        async def scoped_auth(request: Request) -> UserAPIKeyAuth:
            if edit := request.headers.get("x-fixture-offerings"):
                path.write_text(yaml.safe_dump(json.loads(edit)))
                assert await manager.reload()
            return credentials[request.headers["authorization"].removeprefix("Bearer ")]

        application.dependency_overrides[proxy_server.user_api_key_auth] = scoped_auth

        async def listed(auth: UserAPIKeyAuth, edit: str | None = None) -> frozenset[str]:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://gateway.test"
            ) as gateway:
                response: Final = await gateway.get(
                    "/v1/models",
                    headers={
                        "Authorization": f"Bearer {auth.api_key}",
                        **({"x-fixture-offerings": edit} if edit is not None else {}),
                    },
                )
            assert response.status_code == 200, response.text
            assert "private-fixture-key" not in response.text
            assert "supplier.test" not in response.text
            assert "secondary.test" not in response.text
            assert "org/existing" not in response.text
            assert "synthetic/fp8" not in response.text
            assert "offering-" not in response.text
            return frozenset(row["id"] for row in response.json()["data"])

        assert await listed(exact) == existing_names
        assert await listed(admin) == existing_names | {"restricted/other"}
        assert await listed(admin, json.dumps({**config, "offerings": [*config["offerings"], selected]})) == (
            existing_names | {"restricted/other"}
        )
        assert await listed(admin) == existing_names | {"restricted/other", selected["model_name"]}
        assert await listed(wildcard) == existing_names | {selected["model_name"]}
        assert await listed(exact) == existing_names
        assert await listed(denied_key) == set()
        assert await listed(team_key) == {selected["model_name"]}
        assert await listed(alias_key) == {"shortcut"}
        assert await listed(wildcard, json.dumps(config)) == existing_names | {selected["model_name"]}
        assert await listed(wildcard) == existing_names
        assert await listed(alias_key) == set()
        assert await listed(team_key) == set()
        assert await listed(admin) == existing_names | {"restricted/other"}


async def test_authoritative_inventory_failure_recovery_and_atomic_config_reload(tmp_path: Path) -> None:
    path: Final = tmp_path / "offerings.yaml"
    config: Final = {
        "version": 1,
        "providers": {"supplier": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
        "offerings": [
            {"model_name": "automatic", "source": "auto", "provider": "supplier", "upstream_model": "backend"},
            {
                "model_name": "manual",
                "source": "manual",
                "provider": "supplier",
                "upstream_model": "handwritten",
                "model_info": {"context_window": 12345},
            },
        ],
    }
    path.write_text(yaml.safe_dump(config))
    responses: Final = iter(
        (
            httpx.Response(200, json={"data": [{"id": "backend", "context_window": 8000}, {"id": "unselected"}]}),
            httpx.Response(401, json={"error": "expired"}),
            httpx.Response(200, json={"missing": "data"}),
            httpx.ReadTimeout("supplier timed out"),
            httpx.Response(200, json={"data": [], "error": {"message": "failed"}}),
            httpx.Response(200, json={"data": [{"id": "backend", "context_window": "bad"}]}),
            httpx.Response(200, json={"data": [{"id": "backend", "supports_function_calling": "bad"}]}),
            httpx.Response(200, json={"data": [{"id": "backend", "supported_endpoints": 123}]}),
            httpx.Response(200, json={"data": [{"id": " backend "}]}),
            httpx.Response(200, json={"data": []}),
            httpx.Response(200, json={"data": [{"id": "backend", "context_window": 16000}]}),
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        response: Final = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        assert manager.router.model_names == {"automatic", "manual"}
        assert manager.router.latest_snapshot().available_models == {"automatic", "manual"}
        first: Final = manager.router.latest_snapshot()
        for _ in range(8):
            assert await manager.reload(force_inventory=True)
            assert manager.router.latest_snapshot().available_models == first.available_models
            assert (
                manager.router.get_model_listing_info("automatic").deployments[0].model_info["context_window"] == 8000
            )
        pinned: Final = manager.router.latest_snapshot()
        with manager.router.pin_snapshot():
            assert await manager.reload(force_inventory=True)
            assert manager.router.serving_snapshot() is pinned
            assert manager.router.latest_snapshot().available_models == {"manual"}
        assert manager.router.serving_snapshot().available_models == {"manual"}
        assert manager.router.get_model_group_info("automatic") is None
        assert await manager.reload(force_inventory=True)
        assert manager.router.latest_snapshot().available_models == {"automatic", "manual"}
        assert manager.router.get_model_listing_info("automatic").deployments[0].model_info["context_window"] == 16000
        valid: Final = manager.router.latest_snapshot()
        path.write_text("version: 1\nofferings: broken\n")
        assert not await manager.reload()
        assert manager.router.latest_snapshot() is valid
        path.write_text(yaml.safe_dump({**config, "offerings": [config["offerings"][1]]}))
        assert await manager.reload()
        assert manager.router.latest_snapshot().available_models == {"manual"}
        assert manager.router.model_names == {"manual"}


async def test_disabled_manual_offering_and_invalid_secrets_keep_authority_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.offering_router import OfferingAccessGuard

    path: Final = tmp_path / "offerings.yaml"
    monkeypatch.setenv("FIXTURE_OFFERING_KEY", "fixture-secret")
    config: Final = {
        "version": 1,
        "providers": {
            "supplier": {
                "provider": "openai",
                "api_base": "https://supplier.test/v1",
                "api_key": "os.environ/FIXTURE_OFFERING_KEY",
                "headers": {"X-Session": "os.environ/FIXTURE_OFFERING_KEY"},
            }
        },
        "offerings": [
            {"model_name": "manual", "source": "manual", "provider": "supplier", "upstream_model": "backend"}
        ],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []}))
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        active: Final = manager.router.latest_snapshot()
        assert active.available_models == {"manual"}
        assert await manager.reload()
        assert manager.router.latest_snapshot() is active
        monkeypatch.delenv("FIXTURE_OFFERING_KEY")
        assert not await manager.reload()
        assert manager.router.latest_snapshot() is active
        assert manager.last_error == "ValueError"
        path.write_bytes(b" " * (1024 * 1024 + 1))
        assert not await manager.reload()
        assert manager.router.latest_snapshot() is active
        assert manager.last_error == "ConfigTooLarge"
        monkeypatch.setenv("FIXTURE_OFFERING_KEY", "fixture-secret")
        path.write_text(yaml.safe_dump({**config, "offerings": [{**config["offerings"][0], "enabled": False}]}))
        assert await manager.reload()
        assert not manager.router.latest_snapshot().available_models
        assert "disabled" in manager.router.latest_snapshot().unavailable_models["manual"]
        rejected: Final = await OfferingAccessGuard(manager.router).async_pre_call_hook(
            UserAPIKeyAuth(), DualCache(), {"model": "manual"}, "acompletion"
        )
        assert isinstance(rejected, litellm.ServiceUnavailableError)
        path.write_text(yaml.safe_dump(config))
        assert await manager.reload()
        assert manager.router.latest_snapshot().available_models == {"manual"}


async def test_new_connection_cannot_reuse_another_credentials_inventory(tmp_path: Path) -> None:
    path: Final = tmp_path / "offerings.yaml"
    first_config: Final = {
        "version": 1,
        "providers": {
            "supplier": {"provider": "openai", "api_base": "https://supplier.test/v1", "api_key": "first-private-key"}
        },
        "offerings": [
            {"model_name": "selected", "source": "auto", "provider": "supplier", "upstream_model": "backend"}
        ],
    }
    path.write_text(yaml.safe_dump(first_config))
    responses: Final = iter(
        (
            httpx.Response(200, json={"data": [{"id": "backend"}]}),
            httpx.Response(401),
            httpx.Response(200, json={"data": [{"id": "backend"}]}),
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses))) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        assert manager.router.latest_snapshot().available_models == {"selected"}
        changed: Final = {
            **first_config,
            "providers": {"supplier": {**first_config["providers"]["supplier"], "api_key": "second-private-key"}},
        }
        path.write_text(yaml.safe_dump(changed))
        assert await manager.reload()
        assert manager.router.latest_snapshot().available_models == set()
        assert manager.router.get_model_group_info("selected") is None
        assert await manager.reload(force_inventory=True)
        assert manager.router.latest_snapshot().available_models == {"selected"}


async def test_pinned_metadata_does_not_read_new_snapshot_shared_catalog(tmp_path: Path) -> None:
    from litellm.proxy.utils import create_model_info_response

    path: Final = tmp_path / "offerings.yaml"
    initial: Final = {
        "version": 1,
        "providers": {"supplier": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
        "offerings": [
            {
                "model_name": "snapshot-only-model",
                "source": "manual",
                "provider": "supplier",
                "upstream_model": "snapshot-only-backend",
                "model_info": {"context_window": 8000},
            }
        ],
    }
    path.write_text(yaml.safe_dump(initial))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        with manager.router.pin_snapshot():
            before: Final = create_model_info_response("snapshot-only-model", "openai", llm_router=manager.router)
            changed: Final = {
                **initial,
                "offerings": [
                    {**initial["offerings"][0], "model_info": {"context_window": 16000, "max_output_tokens": 5001}}
                ],
            }
            path.write_text(yaml.safe_dump(changed))
            assert await manager.reload()
            assert create_model_info_response("snapshot-only-model", "openai", llm_router=manager.router) == before
        after: Final = create_model_info_response("snapshot-only-model", "openai", llm_router=manager.router)
        assert before["context_window"] == 8000
        assert "max_output_tokens" not in before
        assert after["context_window"] == 16000
        assert after["max_output_tokens"] == 5001


async def test_snapshot_factory_dispatch_and_reload_resources_are_isolated() -> None:
    import gc
    import weakref

    import litellm

    template: Final = Router(model_list=[], num_retries=0)
    callback_counts: Final = tuple(
        len(callbacks)
        for callbacks in (
            litellm.success_callback,
            litellm.failure_callback,
            litellm._async_success_callback,
            litellm._async_failure_callback,
        )
    )
    models: Final = (
        {"model_name": "selected", "litellm_params": {"model": "openai/fixture", "api_key": "local-only"}},
    )
    first: Final = template.snapshot_with_model_list(models)
    response: Final = await first.aresponses(model="selected", input="fixture", mock_response="local-only")
    assert response.output[0].content[0].text == "local-only"
    assert template.get_model_names() == []
    assert template.deployment_names == []
    assert first.deployment_names == ["openai/fixture"]
    references: Final = tuple(weakref.ref(template.snapshot_with_model_list(models)) for _ in range(30))
    gc.collect()
    assert all(reference() is None for reference in references)
    assert template.deployment_names == []
    assert first.deployment_names == ["openai/fixture"]
    assert callback_counts == tuple(
        len(callbacks)
        for callbacks in (
            litellm.success_callback,
            litellm.failure_callback,
            litellm._async_success_callback,
            litellm._async_failure_callback,
        )
    )


async def test_auto_catalog_fallback_is_frozen_and_manual_metadata_stays_handwritten(tmp_path: Path) -> None:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    path: Final = tmp_path / "offerings.yaml"
    upstream: Final = "text-embedding-3-large"
    bundled: Final = GetModelCostMap.loaded_model_cost_map()[upstream]
    assert bundled["max_input_tokens"] > 0
    config: Final = {
        "version": 1,
        "providers": {"supplier": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
        "offerings": [
            {"model_name": "automatic", "source": "auto", "provider": "supplier", "upstream_model": upstream},
            {"model_name": "manual", "source": "manual", "provider": "supplier", "upstream_model": upstream},
        ],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": upstream, "context_window": 12345}]})
        )
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[]), client=handler)
        assert await manager.reload(initial=True)
        auto: Final = manager.router.get_model_listing_info("automatic").deployments[0].model_info
        manual: Final = manager.router.get_model_listing_info("manual").deployments[0].model_info
        assert auto["context_window"] == 12345
        assert auto["max_input_tokens"] == bundled["max_input_tokens"]
        assert auto["mode"] == "embedding"
        assert manual.get("context_window") is None
        assert manual.get("max_input_tokens") is None


async def test_changed_inventory_interval_refreshes_without_retaining_old_deadline(tmp_path: Path) -> None:
    path: Final = tmp_path / "offerings.yaml"
    config: Final = {
        "version": 1,
        "inventory_poll_seconds": 86400,
        "providers": {"supplier": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
        "offerings": [
            {"model_name": "automatic", "source": "auto", "provider": "supplier", "upstream_model": "backend"}
        ],
    }
    path.write_text(yaml.safe_dump(config))
    responses: Final = iter(
        (httpx.Response(200, json={"data": [{"id": "backend"}]}), httpx.Response(200, json={"data": []}))
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses))) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        assert manager.router.latest_snapshot().available_models == {"automatic"}
        path.write_text(yaml.safe_dump({**config, "inventory_poll_seconds": 1}))
        assert await manager.reload()
        assert manager.router.latest_snapshot().available_models == set()
        assert manager.next_inventory_refresh == 1.0


@pytest.mark.parametrize("authorization_name", (None, "authorization", "AUTHORIZATION"))
async def test_native_client_forwards_one_supplier_authorization_value(
    tmp_path: Path, authorization_name: str | None
) -> None:
    path: Final = tmp_path / "offerings.yaml"
    api_key: Final = "supplier-fixture-key.with-symbols="
    expected_authorization: Final = (
        "Bearer explicit-fixture-override" if authorization_name is not None else f"Bearer {api_key}"
    )
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "supplier": {
                        "provider": "openai",
                        "api_base": "https://supplier.test/v1",
                        "api_key": api_key,
                        "headers": (
                            {authorization_name: expected_authorization} if authorization_name is not None else {}
                        ),
                    }
                },
                "offerings": [
                    {
                        "model_name": "selected",
                        "source": "manual",
                        "provider": "supplier",
                        "upstream_model": "fixture-backend",
                    }
                ],
            }
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers.get_list("authorization") == [expected_authorization]
        return httpx.Response(
            200,
            json={
                "id": "fixture-response",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-backend",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "accepted"}, "finish_reason": "stop"}
                ],
            },
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler.client = http_client
        manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[], num_retries=0), client=handler)
        assert await manager.reload(initial=True)
        async with AsyncOpenAI(api_key=api_key, http_client=http_client, max_retries=0) as sdk_client:
            response: Final = await manager.router.acompletion(
                model="selected", messages=[{"role": "user", "content": "authenticate"}], client=sdk_client
            )
        assert response.choices[0].message.content == "accepted"


async def test_global_affinity_preserves_encrypted_content_across_switch_and_pinned_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm
    from litellm.responses.utils import ResponsesAPIRequestUtils

    monkeypatch.setattr(litellm, "callbacks", [])
    path: Final = tmp_path / "offerings.yaml"
    config: Final = {
        "version": 1,
        "providers": {
            "source": {"provider": "openai", "api_base": "https://supplier.test/v1", "api_key": "fixture-key"}
        },
        "offerings": [
            {"model_name": name, "source": "manual", "provider": "source", "upstream_model": "gpt-4o-mini"}
            for name in ("parent", "child")
        ],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    ) as client:
        handler.client = client
        template: Final = Router(model_list=[], optional_pre_call_checks=["encrypted_content_affinity"])
        manager: Final = ModelOfferingsManager(path=path, template=template, client=handler)
        assert await manager.reload(initial=True)
        old: Final = manager.router.latest_snapshot()
        origin: Final = old.router.get_model_ids("parent")[0]
        wrapped: Final = ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id("opaque-fixture", origin)
        with manager.router.pin_snapshot():
            config["offerings"] = config["offerings"][1:]
            path.write_text(yaml.safe_dump(config))
            assert await manager.reload()
            request: Final = {
                "input": [{"type": "reasoning", "encrypted_content": wrapped}, {"role": "user", "content": "fixture"}]
            }
            child: Final = [row for row in old.router.get_model_list() or [] if row["model_name"] == "child"]
            result: Final = await manager.router.async_callback_filter_deployments(
                model="child",
                healthy_deployments=child,
                messages=None,
                parent_otel_span=None,
                request_kwargs=request,
            )
            assert result == child
            assert request["input"][0]["encrypted_content"] == wrapped
        next_request: Final = {
            "input": [{"type": "reasoning", "encrypted_content": wrapped}, {"role": "user", "content": "fixture"}]
        }
        await manager.router.async_callback_filter_deployments(
            model="child",
            healthy_deployments=child,
            messages=None,
            parent_otel_span=None,
            request_kwargs=next_request,
        )
        assert next_request["input"] == [{"role": "user", "content": "fixture"}]
        assert template.deployment_names == []


async def test_private_connection_identity_is_stable_and_changes_on_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "fixture-server-owned-key")
    path: Final = tmp_path / "offerings.yaml"
    config: Final = {
        "version": 1,
        "providers": {
            "source": {
                "provider": "openai",
                "api_base": "https://supplier.test/v1",
                "headers": {"Authorization": "Basic dXNlcjpwYXNz"},
            }
        },
        "offerings": [
            {"model_name": "selected", "source": "manual", "provider": "source", "upstream_model": "gpt-4o-mini"}
        ],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[]), client=handler)
        assert await manager.reload(initial=True)
        before: Final = manager.router.get_model_ids()
        assert await manager.reload(force_inventory=True)
        assert manager.router.get_model_ids() == before
        config["providers"]["source"]["headers"]["Authorization"] = "Basic dXNlcjpuZXctcGFzcw=="
        path.write_text(yaml.safe_dump(config))
        assert await manager.reload()
        assert manager.router.get_model_ids() != before


async def test_native_account_change_changes_private_deployment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "fixture-server-owned-key")
    monkeypatch.setenv("CHATGPT_API_BASE", "https://native.test/backend-api/codex")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    auth_file: Final = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"access_token": "fixture-oauth", "account_id": "first", "expires_at": 10**30}))
    path: Final = tmp_path / "offerings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {"native": {"provider": "chatgpt"}},
                "offerings": [
                    {"model_name": "selected", "source": "auto", "provider": "native", "upstream_model": "fixture"}
                ],
            }
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    responses: Final = iter((200, 401, 200))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(next(responses), json={"models": [{"slug": "fixture"}]})
        )
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[]), client=handler)
        assert await manager.reload(initial=True)
        first: Final = manager.router.get_model_ids()
        assert len(first) == 1
        auth_file.write_text(json.dumps({"access_token": "second-oauth", "account_id": "second", "expires_at": 10**30}))
        assert await manager.reload(force_inventory=True)
        assert not manager.router.latest_snapshot().available_models
        assert await manager.reload(force_inventory=True)
        assert manager.router.get_model_ids() != first
        assert manager.router.latest_snapshot().available_models == {"selected"}


async def test_shared_inventory_client_is_reacquired_after_cache_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    path: Final = tmp_path / "offerings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {"source": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
                "offerings": [
                    {"model_name": "selected", "source": "auto", "provider": "source", "upstream_model": "backend"}
                ],
            }
        )
    )
    cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", cache)
    first: Final = get_async_httpx_client(httpxSpecialProvider.ModelInventory)
    await first.client.aclose()
    first.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "backend"}]}))
    )
    manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[]), clock=lambda: 0.0)
    assert await manager.reload(initial=True)
    assert manager.router.latest_snapshot().available_models == {"selected"}
    cache.delete_cache(cache.update_cache_key_with_event_loop("async_httpx_clientmodel_inventory"))
    await first.close()
    second: Final = get_async_httpx_client(httpxSpecialProvider.ModelInventory)
    assert second is not first
    await second.client.aclose()
    second.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    )
    try:
        assert await manager.reload(force_inventory=True)
        assert manager.router.latest_snapshot().available_models == set()
        assert not second.client.is_closed
    finally:
        await second.close()


@pytest.mark.parametrize("surface", ("chat", "responses", "responses_chat", "messages"))
@pytest.mark.parametrize("stream", (False, True))
async def test_provider_policy_reaches_upstream_body_on_every_chat_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, stream: bool
) -> None:
    path: Final = tmp_path / "offerings.yaml"
    policy: Final = {"only": ["synthetic/fp8"], "allow_fallbacks": False}
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "supplier": {
                        "provider": "openrouter",
                        "api_base": "https://supplier.test/v1",
                        "api_key": "fixture-key",
                    }
                },
                "offerings": [
                    {
                        "model_name": "selected",
                        "source": "manual",
                        "provider": "supplier",
                        "upstream_model": "org/backend",
                        "model_info": {"mode": "chat"},
                        "provider_policy": {"provider": "openrouter", **policy},
                    }
                ],
            }
        )
    )
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        requests.append(request)
        body: Final = json.loads(request.content)
        assert body["model"] == "org/backend"
        assert body["provider"] == policy
        assert body["temperature"] == 0.5
        if request.url.path.endswith("/responses"):
            response: Final = {
                "id": "resp_fixture",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "org/backend",
                "output": [
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "accepted", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        "event: response.completed\ndata: "
                        + json.dumps(
                            {
                                "type": "response.completed",
                                "sequence_number": 1,
                                "response": response,
                            }
                        )
                        + "\n\n"
                    ),
                )
            return httpx.Response(200, json=response)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    "data: "
                    + json.dumps(
                        {
                            "id": "fixture",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "org/backend",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "accepted"},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )
                    + "\n\ndata: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "org/backend",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "accepted"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler.client = http_client
        cache: Final = LLMClientCache()
        cache.set_cache("async_httpx_clientssl_verify_Noneopenrouter", handler)
        monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", cache)
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[], num_retries=0), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        guard: Final = OfferingAccessGuard(manager.router)
        monkeypatch.setattr(litellm, "callbacks", [guard])
        call_type: Final[CallTypesLiteral] = (
            "acompletion" if surface == "chat" else "anthropic_messages" if surface == "messages" else "aresponses"
        )
        response_cache: Final = Cache(
            type="local", supported_call_types=["acompletion", "aresponses", "anthropic_messages"]
        )
        monkeypatch.setattr(litellm, "cache", response_cache)
        response_cache.cache.set_cache("shared-fixture-key", {"response": "unrelated endpoint output"})
        shaped: Final = await ProxyLogging(UserApiKeyCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            data={
                "model": "selected",
                "stream": stream,
                "extra_body": {"temperature": 0.5},
                "caching": True,
                "preset_cache_key": "shared-fixture-key",
                "cache": {"no-cache": False, "no-store": False},
                **({"use_chat_completions_api": True} if surface == "responses_chat" else {}),
            },
            call_type=call_type,
        )
        assert isinstance(shaped, dict)
        result: Final = (
            await manager.router.aresponses(input="fixture", client=handler, **shaped)
            if surface in ("responses", "responses_chat")
            else await manager.router.anthropic_messages(
                messages=[{"role": "user", "content": "fixture"}], max_tokens=10, client=handler, **shaped
            )
            if surface == "messages"
            else await manager.router.acompletion(
                messages=[{"role": "user", "content": "fixture"}], client=handler, **shaped
            )
        )
        if stream:
            chunks: Final = tuple([chunk async for chunk in result])
            assert chunks
        assert len(requests) == 1
        assert requests[0].url.path.endswith("/responses" if surface == "responses" else "/chat/completions")
        assert response_cache.cache.get_cache("shared-fixture-key") == {"response": "unrelated endpoint output"}
        for control in (
            {"additional_drop_params": ["provider"]},
            {"additional_drop_params": ["extra_body"]},
            {"additional_drop_params": ["extra_body.provider"]},
            {"additional_drop_params": ["provider.only"]},
            {"drop_params": True},
            {"allowed_openai_params": ["provider"]},
            {"extra_body": {"provider": policy, "temperature": 0.5, "additional_drop_params": ["provider"]}},
        ):
            with pytest.raises(litellm.BadRequestError, match="configured offering provider policy"):
                await (
                    manager.router.aresponses(input="fixture", client=handler, **{**shaped, **control})
                    if surface in ("responses", "responses_chat")
                    else manager.router.anthropic_messages(
                        messages=[{"role": "user", "content": "fixture"}],
                        max_tokens=10,
                        client=handler,
                        **{**shaped, **control},
                    )
                    if surface == "messages"
                    else manager.router.acompletion(
                        messages=[{"role": "user", "content": "fixture"}], client=handler, **{**shaped, **control}
                    )
                )
        assert len(requests) == 1


async def test_endpoint_policy_reload_isolates_aliases_identity_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.utils import create_model_info_response

    monkeypatch.setenv("LITELLM_SALT_KEY", "fixture-identity-key")
    path: Final = tmp_path / "offerings.yaml"
    selected: Final = {
        "model_name": "selected",
        "source": "auto",
        "provider": "supplier",
        "upstream_model": "org/backend",
        "model_info": {"mode": "chat", "context_window": 4000},
        "provider_policy": {"provider": "openrouter", "only": ["synthetic/fp8"]},
    }
    config: Final = {
        "version": 1,
        "providers": {"supplier": {"provider": "openrouter", "api_base": "https://supplier.test/v1"}},
        "offerings": [
            selected,
            {
                "model_name": "ordinary",
                "source": "auto",
                "provider": "supplier",
                "upstream_model": "org/backend",
            },
        ],
    }
    path.write_text(yaml.safe_dump(config))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "org/backend",
                            "context_length": 16000,
                            "top_provider": {"max_completion_tokens": 8000},
                        }
                    ]
                },
            )
        )
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path, template=Router(model_list=[]), client=handler, clock=lambda: 0.0
        )
        assert await manager.reload(initial=True)
        selected_deployment: Final = manager.router.get_deployment_by_model_group_name("selected")
        assert selected_deployment is not None
        assert selected_deployment.provider_policy.only == ("synthetic/fp8",)
        assert "provider_policy" not in selected_deployment.model_dump()
        selected_id: Final = manager.router.get_model_ids("selected")
        ordinary_id: Final = manager.router.get_model_ids("ordinary")
        pinned_metadata: Final = create_model_info_response("selected", "openrouter", llm_router=manager.router)
        ordinary_metadata: Final = create_model_info_response("ordinary", "openrouter", llm_router=manager.router)
        assert pinned_metadata["context_window"] == 4000
        assert "max_output_tokens" not in pinned_metadata
        assert ordinary_metadata["context_window"] == 16000
        assert ordinary_metadata["max_output_tokens"] == 8000
        assert await manager.reload(force_inventory=True)
        assert manager.router.get_model_ids("selected") == selected_id
        with manager.router.pin_snapshot():
            path.write_text(
                yaml.safe_dump(
                    {
                        **config,
                        "offerings": [
                            {
                                **selected,
                                "provider_policy": {"provider": "openrouter", "only": ["synthetic/bf16"]},
                            },
                            config["offerings"][1],
                        ],
                    }
                )
            )
            assert await manager.reload()
            assert (
                manager.router.get_deployment_by_model_group_name("selected").provider_policy
                == selected_deployment.provider_policy
            )
            assert manager.router.get_model_ids("selected") == selected_id
        assert manager.router.get_model_ids("selected") != selected_id
        assert manager.router.get_model_ids("ordinary") == ordinary_id
        assert manager.router.get_deployment_by_model_group_name("selected").provider_policy.only == ("synthetic/bf16",)
        assert manager.router.get_deployment_by_model_group_name("ordinary").provider_policy is None
        valid: Final = manager.router.latest_snapshot()
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "providers": {
                        "supplier": {
                            "provider": "openai",
                            "api_base": "https://supplier.test/v1",
                        }
                    },
                }
            )
        )
        assert not await manager.reload()
        assert manager.router.latest_snapshot() is valid


async def test_endpoint_failure_never_uses_gateway_fallback_and_tampering_never_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path: Final = tmp_path / "offerings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "providers": {
                    "supplier": {
                        "provider": "openrouter",
                        "api_base": "https://supplier.test/v1",
                        "api_key": "fixture-key",
                    }
                },
                "offerings": [
                    {
                        "model_name": "selected",
                        "source": "manual",
                        "provider": "supplier",
                        "upstream_model": "org/pinned",
                        "model_info": {"mode": "chat"},
                        "provider_policy": {"provider": "openrouter", "only": ["synthetic/fp8"]},
                    },
                    {
                        "model_name": "ordinary",
                        "source": "manual",
                        "provider": "supplier",
                        "upstream_model": "org/other",
                    },
                ],
            }
        )
    )
    forwarded: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        forwarded.append(json.loads(request.content))
        return httpx.Response(503, json={"error": {"message": "synthetic endpoint unavailable", "code": 503}})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(
            path=path,
            client=handler,
            template=Router(
                model_list=[],
                num_retries=0,
                fallbacks=[{"selected": ["ordinary"]}],
            ),
        )
        assert await manager.reload(initial=True)
        guard: Final = OfferingAccessGuard(manager.router)
        monkeypatch.setattr(litellm, "callbacks", [guard])
        shaped: Final = await guard.async_pre_call_hook(
            UserAPIKeyAuth(), DualCache(), {"model": "selected"}, "acompletion"
        )
        assert isinstance(shaped, dict)
        with pytest.raises(litellm.ServiceUnavailableError, match="synthetic endpoint unavailable"):
            await manager.router.acompletion(
                messages=[{"role": "user", "content": "fixture"}], client=handler, **shaped
            )
        assert tuple(body["model"] for body in forwarded) == ("org/pinned",)
        for extra_body in ({"provider": {"only": ["other"]}}, {"model": "org/other"}, {"models": ["org/other"]}):
            with pytest.raises(litellm.BadRequestError, match="configured offering provider policy"):
                await manager.router.acompletion(
                    model="selected",
                    messages=[{"role": "user", "content": "fixture"}],
                    client=handler,
                    extra_body=extra_body,
                    disable_fallbacks=True,
                )
        assert len(forwarded) == 1
