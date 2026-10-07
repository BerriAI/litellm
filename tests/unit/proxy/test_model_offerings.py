from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from openai import AsyncOpenAI

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.model_offerings import ModelOfferingsManager
from litellm.router import Router


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
    from litellm.llms.chatgpt.authenticator import Authenticator

    monkeypatch.setenv("LITELLM_SALT_KEY", "fixture-server-owned-key")
    monkeypatch.setenv("CHATGPT_API_BASE", "https://native.test/backend-api/codex")
    account: Final = {"id": "first"}
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: account["id"])
    monkeypatch.setattr(Authenticator, "get_access_token", lambda self, **kwargs: "fixture-oauth")
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
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"slug": "fixture"}]}))
    ) as client:
        handler.client = client
        manager: Final = ModelOfferingsManager(path=path, template=Router(model_list=[]), client=handler)
        assert await manager.reload(initial=True)
        first: Final = manager.router.get_model_ids()
        assert len(first) == 1
        account["id"] = "second"
        assert await manager.reload(force_inventory=True)
        assert manager.router.get_model_ids() != first
        assert manager.router.latest_snapshot().available_models == {"selected"}
