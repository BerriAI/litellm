from pathlib import Path
from typing import Final

import httpx
import yaml

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
        for _ in range(4):
            assert await manager.reload(force_inventory=True)
            assert manager.router.latest_snapshot().available_models == first.available_models
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
