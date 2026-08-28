from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.proxy_server import app
from litellm.types.router import ModelGroupInfo

client = TestClient(app)

MODEL_HUB_PATH = "/public/v1/model_hub"
LEGACY_MODEL_HUB_PATH = "/public/model_hub"


@dataclass(frozen=True, slots=True)
class _FakeRouter:
    """Stands in for the running Router: `_get_model_group_info` only ever asks it this."""

    infos: Mapping[str, ModelGroupInfo]

    def get_model_group_info(self, model_group: str) -> ModelGroupInfo | None:
        return self.infos.get(model_group)


def _info(
    name: str,
    *,
    mode: str = "chat",
    providers: Sequence[str] = ("openai",),
    **overrides: object,
) -> ModelGroupInfo:
    return ModelGroupInfo(model_group=name, mode=mode, providers=list(providers), **overrides)


def _publish(monkeypatch, infos: Sequence[ModelGroupInfo], prisma_client: object | None = None) -> None:
    monkeypatch.setattr(litellm, "public_model_groups", [info.model_group for info in infos])
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        _FakeRouter(infos=MappingProxyType({info.model_group: info for info in infos})),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)


def _named(count: int, **overrides: object) -> Sequence[ModelGroupInfo]:
    return tuple(_info(f"model-{index:03d}", **overrides) for index in range(count))


def _get(query: str = "", **kwargs):
    suffix = f"?{query}" if query else ""
    return client.get(f"{MODEL_HUB_PATH}{suffix}", **kwargs)


def _groups(response) -> list[str]:
    return [row["model_group"] for row in response.json()["data"]]


def _health_check(model_name: str, status: str = "healthy"):
    check = MagicMock()
    check.model_name = model_name
    check.model_id = None
    check.status = status
    check.response_time_ms = 12.5
    check.checked_at = datetime(2026, 8, 1, 9, 30, tzinfo=timezone.utc)
    return check


def _recording_prisma(checks: Sequence[object] = ()):
    """A prisma client whose only exercised call is the health-check read, recorded for assertions."""
    read = AsyncMock(return_value=list(checks))
    prisma_client = MagicMock()
    prisma_client.get_latest_health_checks_for_models = read
    return prisma_client, read


def _asked_about(read) -> list[str]:
    return list(read.call_args.args[0]) if read.call_args.args else list(read.call_args.kwargs["model_names"])


def test_the_route_is_registered_as_a_public_route():
    """`public_routes` membership is an exact-string check, so the path has to match literally."""
    assert MODEL_HUB_PATH in LiteLLMRoutes.public_routes.value


def test_a_page_slices_the_published_model_groups(monkeypatch):
    _publish(monkeypatch, _named(120))

    response = _get("page=2&page_size=25")

    assert response.status_code == 200, response.text
    assert _groups(response) == [f"model-{index:03d}" for index in range(25, 50)]
    assert response.json()["meta"] == {"total_count": 120, "page": 2, "page_size": 25, "total_pages": 5}


def test_every_page_link_resolves_to_the_page_it_names(monkeypatch):
    _publish(monkeypatch, _named(120))

    links = _get("page=2&page_size=25").json()["links"]

    assert client.get(links["first"]).json()["meta"]["page"] == 1
    assert client.get(links["prev"]).json()["meta"]["page"] == 1
    assert client.get(links["self"]).json()["meta"]["page"] == 2
    assert client.get(links["next"]).json()["meta"]["page"] == 3
    assert client.get(links["last"]).json()["meta"]["page"] == 5


def test_total_count_counts_the_whole_match_set_not_the_page(monkeypatch):
    _publish(monkeypatch, (*_named(30), _info("embedder-1", mode="embedding")))

    response = _get("filter[mode]=chat&page_size=5")

    assert len(response.json()["data"]) == 5
    assert response.json()["meta"]["total_count"] == 30


def test_health_is_resolved_only_for_the_rows_on_the_page(monkeypatch):
    """The bug this endpoint exists to fix: enriching before slicing costs the whole collection.

    An enrich-then-slice implementation asks about all 200 model groups here, not the 10 served.
    """
    prisma_client, read = _recording_prisma()
    _publish(monkeypatch, _named(200), prisma_client=prisma_client)

    response = _get("page=1&page_size=10")

    assert len(response.json()["data"]) == 10
    assert _asked_about(read) == [f"model-{index:03d}" for index in range(10)]


def test_health_is_asked_about_the_second_page_not_the_first(monkeypatch):
    prisma_client, read = _recording_prisma()
    _publish(monkeypatch, _named(200), prisma_client=prisma_client)

    _get("page=4&page_size=10")

    assert _asked_about(read) == [f"model-{index:03d}" for index in range(30, 40)]


def test_the_latest_health_check_lands_on_its_row(monkeypatch):
    prisma_client, _ = _recording_prisma([_health_check("model-001", status="unhealthy")])
    _publish(monkeypatch, _named(3), prisma_client=prisma_client)

    rows = {row["model_group"]: row for row in _get().json()["data"]}

    assert rows["model-001"]["health_status"] == "unhealthy"
    assert rows["model-001"]["health_response_time"] == 12.5
    assert rows["model-001"]["health_checked_at"] == "2026-08-01T09:30:00+00:00"
    assert rows["model-000"]["health_status"] is None


def test_a_health_read_that_returns_nothing_still_serves_the_page(monkeypatch):
    prisma_client, read = _recording_prisma()
    read.return_value = []
    _publish(monkeypatch, _named(3), prisma_client=prisma_client)

    response = _get()

    assert response.status_code == 200, response.text
    assert _groups(response) == ["model-000", "model-001", "model-002"]


def test_rows_are_alphabetical_by_default(monkeypatch):
    _publish(monkeypatch, (_info("zeta"), _info("alpha"), _info("mid")))

    assert _groups(_get()) == ["alpha", "mid", "zeta"]


def test_a_descending_sort_reverses_the_order(monkeypatch):
    _publish(monkeypatch, (_info("zeta"), _info("alpha"), _info("mid")))

    assert _groups(_get("sort=-model_group")) == ["zeta", "mid", "alpha"]


def test_sorting_by_a_numeric_field_puts_the_unset_ones_last_in_both_directions(monkeypatch):
    _publish(
        monkeypatch,
        (
            _info("cheap", input_cost_per_token=0.000001),
            _info("unpriced"),
            _info("dear", input_cost_per_token=0.00003),
        ),
    )

    assert _groups(_get("sort=input_cost_per_token")) == ["cheap", "dear", "unpriced"]
    assert _groups(_get("sort=-input_cost_per_token")) == ["dear", "cheap", "unpriced"]


def test_an_undeclared_sort_field_is_a_problem_naming_the_allowed_fields(monkeypatch):
    _publish(monkeypatch, _named(3))

    response = _get("sort=providers")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "providers" in body["detail"]
    assert body["allowed"] == [
        "input_cost_per_token",
        "max_input_tokens",
        "max_output_tokens",
        "mode",
        "model_group",
        "output_cost_per_token",
    ]


def test_an_unknown_query_parameter_is_a_problem_outside_management_v1(monkeypatch):
    """The `ManagementProblem` handler is registered on the app, not on the `/management/v1` prefix."""
    _publish(monkeypatch, _named(3))

    response = _get("limit=10")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "limit" in response.json()["detail"]


def test_a_repeated_query_parameter_is_rejected(monkeypatch):
    _publish(monkeypatch, _named(3))

    response = _get("page=1&page=99")

    assert response.status_code == 400
    assert "page" in response.json()["detail"]


def test_a_mode_filter_narrows_the_list(monkeypatch):
    _publish(monkeypatch, (_info("chatter"), _info("embedder", mode="embedding")))

    assert _groups(_get("filter[mode]=embedding")) == ["embedder"]
    assert _groups(_get("filter[mode][in]=chat,embedding")) == ["chatter", "embedder"]


def test_a_provider_filter_matches_a_model_group_serving_that_provider(monkeypatch):
    _publish(
        monkeypatch,
        (
            _info("openai-only"),
            _info("mixed", providers=["azure", "bedrock"]),
        ),
    )

    assert _groups(_get("filter[providers][contains]=bedrock")) == ["mixed"]
    assert _groups(_get("filter[providers][contains]=openai")) == ["openai-only"]
    assert _groups(_get("filter[providers][contains]=e, b")) == []


def test_the_search_matches_model_group_names_case_insensitively(monkeypatch):
    _publish(monkeypatch, (_info("gpt-4o"), _info("claude-opus"), _info("GPT-5")))

    assert _groups(_get("q=gpt")) == ["GPT-5", "gpt-4o"]


@pytest.fixture
def guarded(monkeypatch):
    """A proxy with a master key set, so anything but a public route would demand credentials."""
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-1234")
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})


def test_an_unauthenticated_caller_is_served(monkeypatch, guarded):
    _publish(monkeypatch, _named(2))

    response = _get()

    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2


def test_a_bad_api_key_does_not_turn_a_public_route_into_a_401(monkeypatch, guarded):
    _publish(monkeypatch, _named(2))

    response = _get(headers={"Authorization": "Bearer sk-definitely-not-a-real-key"})

    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2


def test_no_published_model_groups_yields_an_empty_but_coherent_envelope(monkeypatch):
    _publish(monkeypatch, ())
    monkeypatch.setattr(litellm, "public_model_groups", None)

    response = _get()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"] == []
    assert body["meta"] == {"total_count": 0, "page": 1, "page_size": 50, "total_pages": 0}
    assert body["links"]["first"].endswith("page=1")
    assert body["links"]["last"].endswith("page=1")
    assert body["links"]["next"] is None
    assert body["links"]["prev"] is None


def test_no_router_answers_with_a_problem_rather_than_the_openai_error_shape(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)

    response = _get()

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == "urn:litellm:error:no-llm-router"


def test_an_unexpected_router_failure_answers_as_a_problem_not_the_openai_error_shape(monkeypatch):
    class _Exploding:
        def get_model_group_info(self, model_group: str) -> ModelGroupInfo:
            raise RuntimeError("router blew up")

    monkeypatch.setattr(litellm, "public_model_groups", ["boom"])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", _Exploding())
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    response = _get()

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == "urn:litellm:error:internal-server-error"


@pytest.mark.parametrize("query", ["", "page=1&page_size=2"])
def test_the_endpoint_it_supersedes_still_answers_with_its_bare_array(monkeypatch, query: str):
    """`/public/model_hub` is what the shipped UI calls; this PR must not move it at all."""
    _publish(monkeypatch, _named(3))
    suffix = f"?{query}" if query else ""

    response = client.get(f"{LEGACY_MODEL_HUB_PATH}{suffix}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert [row["model_group"] for row in body] == ["model-000", "model-001", "model-002"]
