from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from openai import APIStatusError, AsyncOpenAI, OpenAI
from openai.types.eval_create_response import EvalCreateResponse as _Eval
from openai.types.eval_list_response import EvalListResponse as _EvalList
from openai.types.eval_update_response import EvalUpdateResponse as _EvalUpdate
from openai.types.evals.run_create_response import RunCreateResponse as _EvalRunCreate
from openai.types.evals.run_list_response import RunListResponse as _EvalRunList
from pydantic import JsonValue

from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model
from litellm.types.llms.openai_evals import CancelRunResponse as _CancelRunResponse
from litellm.types.videos.utils import encode_video_id_with_provider
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse

_EVAL_DATA_SOURCE_CONFIG: Final[dict[str, JsonValue]] = {
    "type": "custom",
    "item_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
}
_EVAL_DATA_SOURCE_CONFIG_RESPONSE: Final[dict[str, JsonValue]] = {
    "type": "custom",
    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
}
_EVAL_TESTING_CRITERIA: Final[list[JsonValue]] = [
    {"type": "string_check", "name": "exact", "input": "{{item.a}}", "reference": "{{item.a}}", "operation": "eq"}
]
_EVAL_CREATE_UPSTREAM_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "eval_abc",
    "object": "eval",
    "created_at": 1700000000,
    "name": "nightly",
    "data_source_config": _EVAL_DATA_SOURCE_CONFIG_RESPONSE,
    "testing_criteria": _EVAL_TESTING_CRITERIA,
    "metadata": {"suite": "nightly"},
}
_EVAL_CREATE_RESPONSE: Final[dict[str, JsonValue]] = {
    **_EVAL_CREATE_UPSTREAM_RESPONSE,
    "updated_at": None,
}
_EVAL_UPDATE_UPSTREAM_RESPONSE: Final[dict[str, JsonValue]] = {
    **_EVAL_CREATE_UPSTREAM_RESPONSE,
    "name": "renamed",
    "metadata": {"team": "search"},
}
_EVAL_UPDATE_RESPONSE: Final[dict[str, JsonValue]] = {
    **_EVAL_UPDATE_UPSTREAM_RESPONSE,
    "updated_at": None,
}
_EVAL_RUN_DATA_SOURCE: Final[dict[str, JsonValue]] = {
    "type": "jsonl",
    "source": {"type": "file_id", "id": "file-abc"},
}
_EVAL_RUN_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "run_abc",
    "object": "eval.run",
    "created_at": 1700000001,
    "eval_id": "eval_abc",
    "started_at": None,
    "completed_at": None,
    "data_source": _EVAL_RUN_DATA_SOURCE,
    "error": None,
    "per_model_usage": [],
    "per_testing_criteria_results": [],
    "report_url": "https://example.invalid/evals/run_abc",
    "result_counts": {"errored": 0, "failed": 0, "passed": 0, "total": 0},
    "shared_with_openai": None,
    "name": "run-1",
    "model": "gpt-4o-mini",
    "metadata": {"suite": "nightly"},
    "status": "queued",
}
_EVAL_RUN_CANCEL_RESPONSE: Final[dict[str, JsonValue]] = {
    **_EVAL_RUN_RESPONSE,
    "status": "cancelled",
}
_CANCEL_RUN_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "run_abc",
    "object": "eval.run",
    "status": "cancelled",
}


def _evals_respond(request: Request) -> Reply:
    path: Final = request.target.partition("?")[0]
    responses: Final = {
        ("POST", "/v1/evals"): _EVAL_CREATE_UPSTREAM_RESPONSE,
        ("GET", "/v1/evals"): {"object": "list", "data": [_EVAL_CREATE_UPSTREAM_RESPONSE], "has_more": False},
        ("POST", "/v1/evals/eval_abc"): _EVAL_UPDATE_UPSTREAM_RESPONSE,
        ("POST", "/v1/evals/eval_abc/runs"): _EVAL_RUN_RESPONSE,
        ("GET", "/v1/evals/eval_abc/runs"): {"object": "list", "data": [_EVAL_RUN_RESPONSE], "has_more": False},
        ("POST", "/v1/evals/eval_abc/runs/run_abc"): _EVAL_RUN_CANCEL_RESPONSE,
    }
    body: Final = responses.get((request.method, path))
    if body is None:
        return Reply(status=404, body=b'{"error":"unexpected upstream request"}')
    return Reply(body=json.dumps(body).encode())


def _provider_error(status: int) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"scripted provider status {status}",
            "type": "rate_limit_error"
            if status == 429
            else "server_error"
            if status >= 500
            else "invalid_request_error",
            "code": str(status),
        }
    }


class _ObservationBuffer:
    def __init__(self, upstream_url: str) -> None:
        self._url = upstream_url.rstrip("/")
        self._items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(client.get(f"{self._url}/__observations").json())
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self._items = (*self._items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self._items

    def route(self, scenario_id: str, suffix: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            item
            for item in self._items
            if f"/{scenario_id}/" in str(item.get("path")) and str(item.get("path")).endswith(suffix)
        )


def _ready(gateway: Gateway, model: str) -> None:
    eventually(
        lambda: gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "model readiness"}]},
        ),
        lambda response: not (response.status_code == 400 and "Invalid model name" in response.text),
        seconds=30,
    )


def _assert_observation(gateway: Gateway, scenario_id: str, suffix: str) -> tuple[dict[str, JsonValue], ...]:
    buffer: Final = _ObservationBuffer(gateway.upstream_url)
    result: Final = eventually(
        buffer.read,
        lambda _items: len(buffer.route(scenario_id, suffix)) >= 1,
        seconds=20,
    )
    observations: Final = buffer.route(scenario_id, suffix)
    assert observations, result
    return observations


def _add_vector_store(
    gateway: Gateway,
    scenario: Scenario,
    vector_store_id: str,
    alias: str,
    handle: ScenarioHandle,
) -> None:
    created: Final = gateway.request(
        "POST",
        "/vector_store/new",
        {
            "vector_store_id": vector_store_id,
            "custom_llm_provider": "openai",
            "litellm_params": {"model": alias, "api_base": handle.api_base(), "api_key": handle.scenario_id},
        },
    )
    assert created.status_code == 200, created.text
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": vector_store_id})


def _assert_provider_error(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text
    body: Final = JSON_OBJECT.validate_python(response.json())
    error: Final = object_value(body["error"])
    assert str(error.get("code")) == str(status), response.text
    assert "scripted provider status" in str(error.get("message")), response.text


def test_eval_update_forwards_only_client_fields(gateway: Gateway) -> None:
    with wire_server(_evals_respond) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        _ready(gateway, alias)
        wire.drain()
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            try:
                raw: Final = client.evals.with_raw_response.update(
                    "eval_abc", name="renamed", metadata={"team": "search"}, extra_body={"model": alias}
                )
            except APIStatusError as error:
                failure_text: Final = error.response.text
                failed_requests: Final = wire.drain()
                assert [(request.method, request.target) for request in failed_requests] == [
                    ("POST", "/v1/evals/eval_abc")
                ], failure_text
                assert json.loads(failed_requests[0].body) == {"name": "renamed", "metadata": {"team": "search"}}, (
                    failure_text
                )
                assert error.response.status_code == 200, failure_text
                raise
        assert raw.status_code == 200, raw.http_response.text
        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/evals/eval_abc")], (
            raw.http_response.text
        )
        assert json.loads(requests[0].body) == {"name": "renamed", "metadata": {"team": "search"}}, (
            raw.http_response.text
        )
        evaluation: Final[_EvalUpdate] = raw.parse()
        assert json.loads(raw.http_response.text) == _EVAL_UPDATE_RESPONSE, raw.http_response.text
        assert evaluation.name == "renamed", raw.http_response.text


def test_eval_run_cancel_uses_sdk_path(gateway: Gateway) -> None:
    with wire_server(_evals_respond) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        _ready(gateway, alias)
        wire.drain()
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            try:
                raw: Final = client.evals.runs.with_raw_response.cancel(
                    "run_abc", eval_id="eval_abc", extra_query={"model": alias}
                )
            except APIStatusError as error:
                failure_text: Final = error.response.text
                failed_requests: Final = wire.drain()
                assert [(request.method, request.target) for request in failed_requests] == [
                    ("POST", "/v1/evals/eval_abc/runs/run_abc")
                ], failure_text
                assert error.response.status_code == 200, failure_text
                raise
        assert raw.status_code == 200, raw.http_response.text
        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/evals/eval_abc/runs/run_abc")
        ], raw.http_response.text
        assert json.loads(requests[0].body) == {}, raw.http_response.text
        run: Final[_CancelRunResponse] = _CancelRunResponse.model_validate_json(raw.http_response.text)
        assert json.loads(raw.http_response.text) == _CANCEL_RUN_RESPONSE, raw.http_response.text
        assert run.status == "cancelled", raw.http_response.text


@pytest.mark.parametrize("suffix", ("", "/v1", "/v1/"), ids=("root", "v1", "v1-trailing-slash"))
def test_eval_routes_normalize_v1_api_base(gateway: Gateway, suffix: str) -> None:
    with wire_server(_evals_respond) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}{suffix}", api_key="synthetic-openai-key"
        )
        _ready(gateway, alias)
        wire.drain()
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            try:
                create_raw: Final = client.evals.with_raw_response.create(
                    name="nightly",
                    data_source_config=_EVAL_DATA_SOURCE_CONFIG,
                    testing_criteria=_EVAL_TESTING_CRITERIA,
                    metadata={"suite": "nightly"},
                    extra_body={"model": alias},
                )
            except APIStatusError as error:
                failure_text: Final = error.response.text
                failed_requests: Final = wire.drain()
                assert [(request.method, request.target) for request in failed_requests] == [("POST", "/v1/evals")], (
                    failure_text
                )
                assert error.response.status_code == 200, failure_text
                raise
            list_raw: Final = client.evals.with_raw_response.list(limit=2, extra_query={"model": alias})
            run_raw: Final = client.evals.runs.with_raw_response.create(
                "eval_abc",
                data_source=_EVAL_RUN_DATA_SOURCE,
                name="run-1",
                metadata={"suite": "nightly"},
                extra_body={"model": alias},
            )
            runs_raw: Final = client.evals.runs.with_raw_response.list(
                "eval_abc", limit=3, extra_query={"model": alias}
            )
            cancel_raw: Final = client.evals.runs.with_raw_response.cancel(
                "run_abc", eval_id="eval_abc", extra_query={"model": alias}
            )
        response_text: Final = "\n".join(
            response.http_response.text for response in (create_raw, list_raw, run_raw, runs_raw, cancel_raw)
        )
        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/evals"),
            ("GET", "/v1/evals?limit=2"),
            ("POST", "/v1/evals/eval_abc/runs"),
            ("GET", "/v1/evals/eval_abc/runs?limit=3"),
            ("POST", "/v1/evals/eval_abc/runs/run_abc"),
        ], response_text
        assert json.loads(requests[0].body) == {
            "data_source_config": _EVAL_DATA_SOURCE_CONFIG,
            "testing_criteria": _EVAL_TESTING_CRITERIA,
            "name": "nightly",
            "metadata": {"suite": "nightly"},
        }, response_text
        assert json.loads(requests[2].body) == {
            "data_source": _EVAL_RUN_DATA_SOURCE,
            "name": "run-1",
            "metadata": {"suite": "nightly"},
        }, response_text
        assert json.loads(requests[4].body) == {}, response_text
        assert tuple(response.status_code for response in (create_raw, list_raw, run_raw, runs_raw, cancel_raw)) == (
            200,
            200,
            200,
            200,
            200,
        ), response_text
        evaluation: Final[_Eval] = create_raw.parse()
        assert json.loads(create_raw.http_response.text) == _EVAL_CREATE_RESPONSE, response_text
        evaluations: Final[_EvalList] = list_raw.parse()
        assert [item.id for item in evaluations.data] == ["eval_abc"], response_text
        run: Final[_EvalRunCreate] = run_raw.parse()
        expected_run_response: Final = {**_EVAL_RUN_RESPONSE, "model": alias}
        assert json.loads(run_raw.http_response.text) == expected_run_response, response_text
        runs: Final[_EvalRunList] = runs_raw.parse()
        assert [item.id for item in runs.data] == ["run_abc"], response_text
        cancelled: Final[_CancelRunResponse] = _CancelRunResponse.model_validate_json(cancel_raw.http_response.text)
        assert json.loads(cancel_raw.http_response.text) == _CANCEL_RUN_RESPONSE, response_text
        assert (evaluation.id, run.id, cancelled.status) == ("eval_abc", "run_abc", "cancelled"), response_text


@pytest.mark.parametrize(
    "status", (400, 401, 404, 429, 500), ids=("bad-request", "unauthorized", "not-found", "rate-limit", "server-error")
)
def test_vector_store_lookup_preserves_provider_status(gateway: Gateway, status: int) -> None:
    with gateway.scenario() as scenario:
        vector_store_id: Final = f"vs-{uuid.uuid4().hex}"
        handle: Final = register_scenario(
            f"vector-{uuid.uuid4().hex}",
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    f"GET /vector_stores/{vector_store_id}": JsonResponse(
                        content_type="application/json",
                        status=status,
                        body=_provider_error(status),
                    )
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        alias: Final = scenario.model(api_base=handle.api_base(), api_key=handle.scenario_id)
        _add_vector_store(gateway, scenario, vector_store_id, alias, handle)
        _ready(gateway, alias)
        response: Final = gateway.request("GET", f"/v1/vector_stores/{vector_store_id}")
        _assert_provider_error(response, status)
        _assert_observation(gateway, handle.scenario_id, f"/vector_stores/{vector_store_id}")


@pytest.mark.parametrize(
    ("provider_route", "path", "model_name", "status"),
    (
        ("GET /videos/video-id", "/v1/videos/video-id", "openai/gpt-4o-mini", 404),
        ("GET /v1/evals/eval-id", "/v1/evals/eval-id", "openai/gpt-4o-mini", 404),
        ("GET /v1/skills/skill-id", "/v1/skills/skill-id?beta=true", "anthropic/claude-3-5-haiku-20241022", 404),
        ("GET /v1/batch/jobs/batch-id", "/v1/batches/batch-id", "mistral/mistral-large-latest", 404),
        ("GET /v1/messages/batches/batch-id", "/v1/batches/batch-id", "anthropic/claude-3-5-haiku-20241022", 500),
    ),
    ids=("video", "eval", "skill", "mistral-batch", "anthropic-batch-gap"),
)
def test_model_scoped_lookup_returns_scripted_provider_404(
    gateway: Gateway,
    provider_route: str,
    path: str,
    model_name: str,
    status: int,
) -> None:
    with gateway.scenario() as scenario:
        route_path: Final = provider_route.partition(" ")[2]
        handle: Final = register_scenario(
            f"scoped-{uuid.uuid4().hex}",
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    provider_route: JsonResponse(content_type="application/json", status=404, body=_provider_error(404))
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        alias: Final = scenario.model(model=model_name, api_base=handle.api_base(), api_key=handle.scenario_id)
        _ready(gateway, alias)
        request_path: Final = (
            f"/v1/videos/{encode_video_id_with_provider('video-id', 'openai', model_id=alias)}"
            if "/videos/" in route_path
            else f"/v1/batches/{encode_file_id_with_model('batch-id', alias, id_type='batch')}"
            if "/batches/" in route_path or "/batch/jobs/" in route_path
            else path
        )
        headers: Final = {"x-litellm-model": alias} if "/skills/" in request_path else {}
        params: Final = {"model": alias} if "/evals/" in request_path else None
        response: Final = gateway.request("GET", request_path, params=params, headers=headers)
        assert response.status_code == status, response.text
        body: Final = JSON_OBJECT.validate_python(response.json())
        message: Final = str(object_value(body["error"]).get("message"))
        assert (
            "Client error '404 Not Found'" in message and "/v1/messages/batches/batch-id" in message
            if status == 500
            else "scripted provider status 404" in message
        ), response.text
        suffix: Final = "/videos/video-id" if "/videos/" in route_path else route_path
        _assert_observation(gateway, handle.scenario_id, suffix)


def _sdk_file_lookup(gateway: Gateway, file_id: str) -> httpx.Response:
    with OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        try:
            client.files.retrieve(file_id)
        except APIStatusError as error:
            return error.response
    pytest.fail("OpenAI SDK file lookup unexpectedly succeeded")


async def _async_sdk_file_lookup(gateway: Gateway, file_id: str) -> httpx.Response:
    async with AsyncOpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        try:
            await client.files.retrieve(file_id)
        except APIStatusError as error:
            return error.response
    pytest.fail("OpenAI async SDK file lookup unexpectedly succeeded")


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_openai_sdk_file_lookup_returns_head_provider_error(
    gateway: Gateway,
    client_kind: Literal["sync", "async"],
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"d12-file-lookup-{client_kind}"
        handle: Final = register_scenario(
            scenario_id,
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    "GET /files/file-id": JsonResponse(
                        content_type="application/json",
                        status=404,
                        body=_provider_error(404),
                    )
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = f"audit-file-lookup-{uuid.uuid4().hex}"
        config: Final = tmp_path / f"d12-{client_kind}.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": model,
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_base": handle.api_base(),
                                "api_key": scenario_id,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {},
            config=config,
            workers=2,
        ) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            file_id: Final = encode_file_id_with_model("file-id", model)
            response: Final = (
                _sdk_file_lookup(candidate, file_id)
                if client_kind == "sync"
                else asyncio.run(_async_sdk_file_lookup(candidate, file_id))
            )
            expected: Final = {
                "error": {
                    "message": f"Error code: 404 - {_provider_error(404)}",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "404",
                }
            }
            assert response.status_code == 404, response.text
            assert response.json() == expected, response.text
            _assert_observation(candidate, scenario_id, "/files/file-id")
