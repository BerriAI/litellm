"""
Test that _update_llm_router and _delete_deployment are resilient to
config loading failures (e.g. database timeouts).

This addresses a bug where httpcore.ReadTimeout from the Prisma client
during get_config() would prevent ALL DB models from loading into the
router, because the exception propagated up and was caught by the
catch-all handler in _update_llm_router.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.proxy._types import ModelRoutesReload, UnloadedDbModel
from litellm.proxy.proxy_server import ProxyConfig
from tests._master_key import MASTER_KEY
from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs


def _make_db_model(model_name: str, model_id: str):
    """Helper to create a mock DB model record."""
    record = MagicMock()
    record.model_id = model_id
    record.model_name = model_name
    record.litellm_params = {"model": model_name}
    record.model_info = {"id": model_id}
    record.created_by = "default_user_id"
    record.created_at = None
    record.updated_at = None
    record.updated_by = None
    return record


class TestUpdateLlmRouterResilience:
    """Test _update_llm_router handles get_config failures gracefully."""

    @pytest.mark.asyncio
    async def test_models_loaded_when_get_config_times_out(self):
        """DB models should still be added to the router when get_config() raises a timeout."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = route_entry_points_to_stubs(MagicMock())
        mock_router.get_model_list.return_value = []
        mock_router.get_model_ids.return_value = []

        mock_proxy_logging = MagicMock()

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                side_effect=Exception("httpcore.ReadTimeout"),
            ),
            patch.object(proxy_config, "_add_deployment", return_value=1) as mock_add,
            patch.object(
                proxy_config,
                "_delete_deployment",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.llm_model_list", []),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            await proxy_config._update_llm_router(
                new_models=db_models,
                proxy_logging_obj=mock_proxy_logging,
            )

            # _add_deployment should still have been called despite get_config failure
            mock_add.assert_called_once_with(db_models=db_models)

    @pytest.mark.asyncio
    async def test_get_config_success_still_works(self):
        """Normal flow should still work when get_config succeeds."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = route_entry_points_to_stubs(MagicMock())
        mock_router.get_model_list.return_value = []
        mock_router.get_model_ids.return_value = []

        mock_proxy_logging = MagicMock()

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                return_value={"model_list": []},
            ),
            patch.object(proxy_config, "_add_deployment", return_value=1) as mock_add,
            patch.object(
                proxy_config,
                "_delete_deployment",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.llm_model_list", []),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            await proxy_config._update_llm_router(
                new_models=db_models,
                proxy_logging_obj=mock_proxy_logging,
            )

            mock_add.assert_called_once_with(db_models=db_models)


_OLD_ROUTE: Final = "runwayml/gen4.5"
_NEW_ROUTE: Final = "vertex_ai/gemini-3.8-flash"


def _db_row(
    model_id: str, route: str, api_key: str | None = None, team_info: Mapping[str, str] | None = None
) -> LiteLLM_ProxyModelTable:
    return LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name=model_id,
        litellm_params={"model": route} if api_key is None else {"model": route, "api_key": api_key},
        model_info={"id": model_id, "db_model": True, **(team_info or {})},
    )


def _router_with(*deployments: tuple[str, str, bool]) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": model_id,
                "litellm_params": {"model": route},
                "model_info": {"id": model_id, "db_model": db_model},
            }
            for model_id, route, db_model in deployments
        ]
    )


class TestModelRoutesReloadForQuotaChecks:
    async def _reload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        router: litellm.Router | None,
        db_models: Sequence[LiteLLM_ProxyModelTable],
    ) -> ModelRoutesReload:
        monkeypatch.delenv("QA_UNSET_OIDC_TOKEN", raising=False)
        proxy_config: Final = ProxyConfig()
        monkeypatch.setattr("litellm.proxy.proxy_server.DB_RECONCILE_LOCK", asyncio.Lock())
        monkeypatch.setattr(proxy_config, "get_models_from_db", AsyncMock(return_value=list(db_models)))
        monkeypatch.setattr(proxy_config, "_upsert_db_credentials", AsyncMock(return_value=True))
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.master_key", MASTER_KEY),
            patch("litellm.proxy.proxy_server.llm_model_list", []),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            return await _reconcile(proxy_config)

    @pytest.mark.asyncio
    async def test_a_model_whose_secret_cannot_load_is_reported_while_the_other_models_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        router: Final = _router_with(("clip-token", _OLD_ROUTE, True), ("clip-moved", _OLD_ROUTE, True))
        broken: Final = _db_row(
            "clip-token",
            _OLD_ROUTE,
            api_key="os.environ/oidc/env/QA_UNSET_OIDC_TOKEN",
            team_info={"team_id": "media-team", "team_public_model_name": "clip-studio"},
        )

        outcome: Final = await self._reload(monkeypatch, router, [broken, _db_row("clip-moved", _NEW_ROUTE)])

        assert outcome == ModelRoutesReload(
            routes_current=True,
            unloaded=frozenset(
                {
                    UnloadedDbModel(
                        model_id="clip-token", names=frozenset({"clip-token", "clip-studio"}), team_id="media-team"
                    )
                }
            ),
        )
        moved: Final = router.registry().deployment("clip-moved")
        assert moved is not None
        assert moved.litellm_params.model == _NEW_ROUTE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("db_models", "routes_current"),
        [
            ((), True),
            ((_db_row("clip-moved", _NEW_ROUTE),), True),
            (
                (
                    _db_row("clip-moved", _NEW_ROUTE),
                    _db_row("clip-token", _OLD_ROUTE, api_key="os.environ/oidc/env/QA_UNSET_OIDC_TOKEN"),
                ),
                False,
            ),
        ],
        ids=["no-db-models", "router-built", "router-build-failed"],
    )
    async def test_a_worker_with_no_router_confirms_routes_only_once_its_first_router_builds(
        self, monkeypatch: pytest.MonkeyPatch, db_models: Sequence[LiteLLM_ProxyModelTable], routes_current: bool
    ):
        monkeypatch.setattr(ProxyConfig, "get_config", AsyncMock(return_value={}))

        assert await self._reload(monkeypatch, None, db_models) == ModelRoutesReload(routes_current=routes_current)

    @pytest.mark.asyncio
    async def test_reload_drops_models_deleted_from_the_db_and_keeps_config_models(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        router: Final = _router_with(
            ("config-model", _OLD_ROUTE, False), ("db-deleted", _OLD_ROUTE, True), ("db-kept", _OLD_ROUTE, True)
        )

        outcome: Final = await self._reload(monkeypatch, router, [_db_row("db-kept", _NEW_ROUTE)])

        assert outcome == ModelRoutesReload(routes_current=True)
        assert sorted(router.registry().names(kind="ids")) == ["config-model", "db-kept"]


class TestDeleteDeploymentResilience:
    """Test _delete_deployment handles get_config failures gracefully."""

    @pytest.mark.asyncio
    async def test_returns_none_when_get_config_times_out(self):
        """Should return None (no reconcile ran, desired set unknown) when get_config
        fails, not raise. A caller judging its own reload must not read that as "the db
        wants nothing" and blame the reload for every model it serves."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = route_entry_points_to_stubs(MagicMock())
        mock_router.get_model_ids.return_value = ["db-id-1", "config-id-1"]

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                side_effect=Exception("httpcore.ReadTimeout"),
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=db_models)

            # Should safely return None instead of raising
            assert result is None
            # Should NOT have deleted any deployments
            mock_router.delete_deployment.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_delete_still_works(self):
        """Normal deletion should work when get_config succeeds."""
        proxy_config = ProxyConfig()

        db_models = [_make_db_model("gpt-5.1", "db-id-1")]

        mock_router = route_entry_points_to_stubs(MagicMock())
        # Router has a model ID that's not in DB or config -> should be deleted
        mock_router.get_model_ids.return_value = ["db-id-1", "stale-id"]
        mock_router.delete_deployment.return_value = True
        mock_router.generate_model_id = MagicMock(return_value="config-id-1")

        with (
            patch.object(
                proxy_config,
                "get_config",
                new_callable=AsyncMock,
                return_value={
                    "model_list": [
                        {
                            "model_name": "gpt-4",
                            "litellm_params": {"model": "gpt-4"},
                            "model_info": {"id": "config-id-1"},
                        }
                    ]
                },
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=db_models)

            # "stale-id" should have been deleted (not in db_models or config)
            mock_router.delete_deployment.assert_called_once_with(id="stale-id")
            assert result == frozenset({"db-id-1", "config-id-1"}), (
                "the returned set must be what the db + config still want, so a caller can "
                f"tell that eviction apart from a deployment that went missing; got {result}"
            )


class TestDeleteDeploymentKeepsPluginConfigModels:
    """Regression: _delete_deployment re-reads the raw config and hashes litellm_params to
    compute the ids the config wants served. The Router used to derive plugin-bearing
    deployment ids from the RESOLVED params (dotted paths swapped for live instances), so
    the reconcile computed different ids and evicted every plugin-bearing auto-router one
    sync after startup. load_config now pins model_info.id from the raw params before
    resolution, so both sides hash the same input and the reconcile needs no resolution."""

    @staticmethod
    def _write_plugin_module(tmp_path):
        (tmp_path / "rig_classifier.py").write_text(
            "class _Classifier:\n"
            "    async def classify(self, context):\n"
            "        return 'SIMPLE'\n"
            "\n"
            "class _Narrower:\n"
            "    async def run(self, context):\n"
            "        return context\n"
            "\n"
            "classifier_instance = _Classifier()\n"
            "narrower_instance = _Narrower()\n"
        )

    @staticmethod
    def _raw_model_entry():
        return {
            "model_name": "smart-router",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_default_model": "gpt-4o-mini",
                "complexity_router_config": {
                    "classifier_type": "custom",
                    "classifier_plugin": "rig_classifier.classifier_instance",
                    "plugins": ["rig_classifier.narrower_instance"],
                    "tiers": {"SIMPLE": "gpt-4o-mini"},
                },
            },
        }

    @pytest.mark.asyncio
    async def test_plugin_bearing_config_model_survives_reconcile_and_stale_ids_still_evict(self, tmp_path):
        import copy

        from litellm import Router
        from litellm.proxy.proxy_server import (
            pin_complexity_router_model_id,
            resolve_complexity_router_plugins,
        )

        self._write_plugin_module(tmp_path)
        config_file_path = str(tmp_path / "config.yaml")

        resolved_entry = copy.deepcopy(self._raw_model_entry())
        pin_complexity_router_model_id(resolved_entry)
        resolve_complexity_router_plugins(
            model_name="smart-router",
            complexity_router_config=resolved_entry["litellm_params"]["complexity_router_config"],
            config_file_path=config_file_path,
        )
        router = Router(
            model_list=[
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                resolved_entry,
                {
                    "model_name": "stale-model",
                    "litellm_params": {"model": "gpt-4o"},
                    "model_info": {"id": "stale-id"},
                },
            ]
        )
        assert "smart-router" in router.model_names
        assert "stale-model" in router.model_names

        raw_config = {
            "model_list": [
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                self._raw_model_entry(),
            ]
        }
        proxy_config = ProxyConfig()
        with (
            patch.object(proxy_config, "get_config", new_callable=AsyncMock, return_value=raw_config),
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.user_config_file_path", config_file_path),
            patch("litellm.proxy.proxy_server.premium_user", False),
        ):
            result = await proxy_config._delete_deployment(db_models=[])

        assert result is not None
        assert "smart-router" in router.model_names
        assert "stale-model" not in router.model_names

    def test_pin_respects_an_explicit_model_id(self):
        from litellm.proxy.proxy_server import pin_complexity_router_model_id

        entry = self._raw_model_entry()
        entry["model_info"] = {"id": "operator-pinned"}
        pin_complexity_router_model_id(entry)
        assert entry["model_info"]["id"] == "operator-pinned"

    def test_pin_is_a_noop_without_a_complexity_router_config(self):
        from litellm.proxy.proxy_server import pin_complexity_router_model_id

        entry = {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}
        pin_complexity_router_model_id(entry)
        assert "model_info" not in entry


class TestDeleteDeploymentKeepsConfigModelsOnEmptyConfigRead:
    """Regression: a config read that succeeds but returns no model_list (e.g. a
    partially written file) must not evict config-sourced deployments, because
    nothing re-adds config models at runtime. DB-sourced deployments missing from
    db_models must still be evicted."""

    @staticmethod
    def _router(model_list):
        from litellm import Router
        from litellm.types.router import RouterGeneralSettings

        return Router(
            model_list=model_list,
            router_general_settings=RouterGeneralSettings(async_only_mode=True),
        )

    @pytest.mark.asyncio
    async def test_delete_deployment_keeps_config_models_when_config_read_has_no_model_list(self, tmp_path):
        config_file_path = str(tmp_path / "config.yaml")
        (tmp_path / "config.yaml").write_text(f"general_settings:\n  master_key: {MASTER_KEY}\n")

        router = self._router(
            [
                {
                    "model_name": "config-model",
                    "litellm_params": {"model": "gpt-4o-mini"},
                    "model_info": {"id": "config-model-1"},
                },
                {
                    "model_name": "db-model",
                    "litellm_params": {"model": "gpt-4o-mini"},
                    "model_info": {"id": "db-model-1", "db_model": True},
                },
            ]
        )
        proxy_config = ProxyConfig()
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),  # test-quality-ok: reads module global
            patch(  # test-quality-ok: reads module global
                "litellm.proxy.proxy_server.user_config_file_path",
                config_file_path,
            ),
        ):
            result = await proxy_config._delete_deployment(db_models=[])

        model_ids = router.get_model_ids()
        assert "config-model-1" in model_ids
        assert "db-model-1" not in model_ids
        assert result is not None
        assert "config-model-1" in result

    @pytest.mark.asyncio
    async def test_delete_deployment_still_evicts_config_model_removed_from_non_empty_model_list(self, tmp_path):
        config_file_path = str(tmp_path / "config.yaml")
        (tmp_path / "config.yaml").write_text(
            "model_list:\n"
            "  - model_name: model-a\n"
            "    litellm_params:\n"
            "      model: gpt-4o-mini\n"
            "    model_info:\n"
            "      id: model-a-id\n"
        )

        router = self._router(
            [
                {
                    "model_name": "model-a",
                    "litellm_params": {"model": "gpt-4o-mini"},
                    "model_info": {"id": "model-a-id"},
                },
                {
                    "model_name": "model-b",
                    "litellm_params": {"model": "gpt-4o-mini"},
                    "model_info": {"id": "model-b-id"},
                },
            ]
        )
        proxy_config = ProxyConfig()
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),  # test-quality-ok: reads module global
            patch(  # test-quality-ok: reads module global
                "litellm.proxy.proxy_server.user_config_file_path",
                config_file_path,
            ),
        ):
            result = await proxy_config._delete_deployment(db_models=[])

        model_ids = router.get_model_ids()
        assert "model-a-id" in model_ids
        assert "model-b-id" not in model_ids
        assert result == frozenset({"model-a-id"})

    @pytest.mark.asyncio
    async def test_delete_deployment_evicts_config_models_on_explicit_empty_model_list(self, tmp_path):
        config_file_path = str(tmp_path / "config.yaml")
        (tmp_path / "config.yaml").write_text("model_list: []\n")

        router = self._router(
            [
                {
                    "model_name": "config-model",
                    "litellm_params": {"model": "gpt-4o-mini"},
                    "model_info": {"id": "config-model-1"},
                },
            ]
        )
        proxy_config = ProxyConfig()
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),  # test-quality-ok: reads module global
            patch(  # test-quality-ok: reads module global
                "litellm.proxy.proxy_server.user_config_file_path",
                config_file_path,
            ),
        ):
            result = await proxy_config._delete_deployment(db_models=[])

        assert router.get_model_ids() == []
        assert result == frozenset()


def _count_model_reads(
    monkeypatch: pytest.MonkeyPatch, proxy_config: ProxyConfig, first_read_held_until: asyncio.Event | None
) -> AsyncMock:
    first_read_started: Final = asyncio.Event()

    async def read_models(prisma_client: object) -> list[LiteLLM_ProxyModelTable]:
        if first_read_held_until is not None and not first_read_started.is_set():
            first_read_started.set()
            await first_read_held_until.wait()
        return []

    read_models_mock: Final = AsyncMock(side_effect=read_models)
    monkeypatch.setattr("litellm.proxy.proxy_server.DB_RECONCILE_LOCK", asyncio.Lock())
    monkeypatch.setattr(proxy_config, "get_models_from_db", read_models_mock)
    monkeypatch.setattr(proxy_config, "_upsert_db_credentials", AsyncMock(return_value=True))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", litellm.Router(model_list=[]))
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
    return read_models_mock


async def _reconcile(proxy_config: ProxyConfig) -> ModelRoutesReload:
    return await proxy_config.reconcile_db_models(
        prisma_client=MagicMock(), proxy_logging_obj=MagicMock(), requested_at=time.monotonic()
    )


@pytest.mark.asyncio
async def test_quota_reload_installs_a_new_db_credential_while_the_config_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_config: Final = ProxyConfig()
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_credentialstable.find_many = AsyncMock(
        return_value=[
            {"credential_name": "media-cred", "credential_values": {"api_key": "sk-media"}, "credential_info": {}}
        ]
    )
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.setattr("litellm.proxy.proxy_server.DB_RECONCILE_LOCK", asyncio.Lock())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", litellm.Router(model_list=[]))
    monkeypatch.setattr(proxy_config, "get_models_from_db", AsyncMock(return_value=[]))
    monkeypatch.setattr(proxy_config, "get_config", AsyncMock(side_effect=TimeoutError("config bucket unreachable")))

    outcome: Final = await proxy_config.reconcile_db_models(
        prisma_client=prisma_client, proxy_logging_obj=MagicMock(), requested_at=time.monotonic()
    )

    assert outcome == ModelRoutesReload(routes_current=True)
    assert CredentialAccessor.get_credential_values("media-cred") == {"api_key": "sk-media"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_credentials",
    [
        pytest.param(AsyncMock(side_effect=httpx.ConnectError("db unreachable")), id="db-unreachable"),
        pytest.param(
            AsyncMock(
                return_value=[
                    {"credential_name": "media-cred", "credential_values": {"api_key": "sk"}, "credential_info": None}
                ]
            ),
            id="a-row-cannot-be-parsed",
        ),
    ],
)
async def test_quota_reload_cannot_confirm_routes_while_the_credentials_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, read_credentials: AsyncMock
) -> None:
    proxy_config: Final = ProxyConfig()
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_credentialstable.find_many = read_credentials
    router: Final = litellm.Router(model_list=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.DB_RECONCILE_LOCK", asyncio.Lock())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(proxy_config, "get_models_from_db", AsyncMock(return_value=[_db_row("db-clip", _NEW_ROUTE)]))

    outcome: Final = await proxy_config.reconcile_db_models(
        prisma_client=prisma_client, proxy_logging_obj=MagicMock(), requested_at=time.monotonic()
    )

    assert outcome == ModelRoutesReload(routes_current=False)
    assert router.get_deployment(model_id="db-clip") is None


@pytest.mark.asyncio
async def test_quota_checks_queued_behind_a_model_reload_share_the_next_one(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy_config: Final = ProxyConfig()
    release_first_read: Final = asyncio.Event()
    read_models: Final = _count_model_reads(monkeypatch, proxy_config, release_first_read)

    in_flight: Final = asyncio.ensure_future(_reconcile(proxy_config))
    await asyncio.sleep(0)
    queued: Final = tuple(asyncio.ensure_future(_reconcile(proxy_config)) for _ in range(4))
    await asyncio.sleep(0)
    release_first_read.set()
    outcomes: Final = await asyncio.gather(in_flight, *queued)

    assert read_models.await_count == 2
    assert all(outcome.routes_current for outcome in outcomes)


@pytest.mark.asyncio
async def test_quota_check_after_a_finished_model_reload_reads_the_models_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_config: Final = ProxyConfig()
    read_models: Final = _count_model_reads(monkeypatch, proxy_config, first_read_held_until=None)

    _ = await _reconcile(proxy_config)
    _ = await _reconcile(proxy_config)

    assert read_models.await_count == 2
