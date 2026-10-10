import uuid
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse


_ASSEMBLYAI_PAYLOAD: Final = {"audio_url": "https://assembly.ai/wildfires.mp3", "speech_models": ["universal-2"]}


def _assemblyai_scenario(scenario: Scenario) -> ScenarioHandle:
    handle: Final = register_scenario(
        f"assemblyai-{uuid.uuid4().hex}",
        RoutedResponse(
            content_type="application/x-routed",
            routes={
                "POST /v2/transcript": JsonResponse(
                    content_type="application/json",
                    body={"id": "transcript-1", "status": "queued"},
                ),
                "GET /v2/transcript/transcript-1": JsonResponse(
                    content_type="application/json",
                    body={"id": "transcript-1", "status": "completed", "text": "wildfire report"},
                ),
                "DELETE /v2/transcript/transcript-1": JsonResponse(
                    content_type="application/json",
                    body={},
                ),
            },
        ),
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _observations(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        payload: Final = JSON_OBJECT.validate_python(upstream.get("/__observations").json())
    requests: Final = payload.get("requests")
    assert isinstance(requests, list)
    return tuple(object_value(request) for request in requests if isinstance(request, dict))


def _assemblyai_model(scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model="assemblyai/universal-2",
        custom_llm_provider="assemblyai",
        api_key="assemblyai-integration-key",
        api_base=api_base,
        use_in_pass_through=True,
    )


def test_assemblyai_passthrough_forwards_transcript_requests(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        upstream: Final = _assemblyai_scenario(scenario)
        model: Final = _assemblyai_model(scenario, upstream.api_base())
        key: Final = scenario.key()

        create: Final = gateway.request("POST", "/assemblyai/v2/transcript", _ASSEMBLYAI_PAYLOAD, key=key)
        assert create.status_code == 200, create.text
        assert create.json() == {"id": "transcript-1", "status": "queued"}

        poll: Final = gateway.request("GET", "/assemblyai/v2/transcript/transcript-1", key=key)
        assert poll.status_code == 200, poll.text
        assert poll.json() == {"id": "transcript-1", "status": "completed", "text": "wildfire report"}

        delete: Final = gateway.request("DELETE", "/assemblyai/v2/transcript/transcript-1", key=key)
        assert delete.status_code == 200, delete.text

        scenario_prefix: Final = f"/{upstream.scenario_id}/v2/transcript"
        requests: Final = tuple(
            request for request in _observations(gateway) if str(request["path"]).startswith(scenario_prefix)
        )
        create_requests: Final = tuple(request for request in requests if request["method"] == "POST")
        assert len(create_requests) == 1
        assert object_value(create_requests[0]["body"]) == _ASSEMBLYAI_PAYLOAD
        assert create_requests[0]["authorization"] == "assemblyai-integration-key"
        assert any(
            request["method"] == "GET" and request["path"] == f"{scenario_prefix}/transcript-1"
            for request in requests
        )
        assert any(
            request["method"] == "DELETE" and request["path"] == f"{scenario_prefix}/transcript-1"
            for request in requests
        )


def test_assemblyai_routes_reject_invalid_proxy_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        upstream: Final = _assemblyai_scenario(scenario)
        _assemblyai_model(scenario, upstream.api_base())
        responses: Final = tuple(
            gateway.request(
                "POST",
                path,
                _ASSEMBLYAI_PAYLOAD,
                key="sk-12222",
            )
            for path in ("/assemblyai/v2/transcript", "/eu.assemblyai/v2/transcript")
        )
        assert tuple(response.status_code for response in responses) == (401, 401)
        assert not tuple(
            request
            for request in _observations(gateway)
            if str(request["path"]).startswith(f"/{upstream.scenario_id}/v2/transcript")
        )
