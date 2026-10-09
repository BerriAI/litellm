import json
import os
import uuid
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import scratch_database
from integration._support.process import owned_proxy
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue

_DECISIONS_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "pplx-decider-v1-27b",
        "answers": {"reachable": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_CONFIGURED_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "jev-custom",
        "answers": {"alive": {"type": "choice", "choice": "yes", "confidence": 0.9, "probabilities": {"yes": 0.9}}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_STRANDS_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "strands-decider-2B-hobson-v19",
        "answers": {"reachable": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_CONFIGURED_STATE: Final[dict[str, JsonValue]] = {"ticket": "health probe"}
_CONFIGURED_QUESTIONS: Final[dict[str, JsonValue]] = {
    "alive": {"type": "choice", "criteria": {"yes": "the service answers", "no": "the service is down"}}
}
_HEALTH_CHAT_MESSAGES: Final = (
    [{"role": "user", "content": "Hey how's it going?"}],
    [{"role": "user", "content": "What's 1 + 1?"}],
)


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _multipart_text_fields(parts: tuple[Message, ...]) -> dict[str, str]:
    return {
        string_value(part.get_param("name", header="content-disposition")): part.get_payload(decode=True).decode()
        for part in parts
        if part.get_filename() is None
    }


def _multipart_file_fields(parts: tuple[Message, ...]) -> dict[str, tuple[str | None, str | None, bytes]]:
    return {
        string_value(part.get_param("name", header="content-disposition")): (
            part.get_filename(),
            part.get_content_type(),
            part.get_payload(decode=True),
        )
        for part in parts
        if part.get_filename() is not None
    }


def test_health_check_of_a_model_added_through_the_api_calls_its_upstream_and_reports_it_healthy(
    gateway: Gateway,
) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        provider_model: Final = f"health-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        key: Final = scenario.key(models=[model])
        listed: Final = gateway.request("GET", "/v2/model/info", key=key, params={"model": model})
        assert listed.status_code == 200, listed.text
        assert [entry["model_name"] for entry in listed.json()["data"]] == [model]
        assert (
            object_value(gateway.chat(model, key=key, text=f"health {uuid.uuid4().hex}")["usage"])["total_tokens"] == 40
        )
        upstream.get("/__observations").raise_for_status()
        health: Final = gateway.request("GET", "/health", params={"model": model})
        assert health.status_code == 200, health.text
        report: Final = health.json()
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert [(endpoint["model"], endpoint["api_base"]) for endpoint in report["healthy_endpoints"]] == [
            (f"openai/{provider_model}", f"{gateway.upstream_url}/v1")
        ]
        assert [request["body"]["model"] for request in upstream.get("/__observations").json()["requests"]] == [
            provider_model
        ]


def _health_report(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    health: Final = gateway.request("GET", "/health", params={"model": model})
    assert health.status_code == 200, health.text
    return health.json()


def _background_health_reply(request: Request, *, model: str, api_key: str) -> Reply:
    if request.method == "GET":
        assert request.target == "/v1/models", request.target
        assert request.headers["authorization"] == f"Bearer {api_key}", request.headers
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"id": model, "object": "model", "created": 1700000000, "owned_by": "integration"}],
                }
            ).encode()
        )
    return _health_chat_reply(request, api_key=api_key, model=model)


def _assert_background_health_requests(wire: Wire, *, model: str, api_key: str) -> None:
    requests: Final = wire.drain()
    assert len(requests) == 2, requests
    assert (
        requests[0].method,
        requests[0].target,
        requests[0].headers["authorization"],
        requests[0].body,
    ) == ("GET", "/v1/models", f"Bearer {api_key}", b""), requests
    assert (
        requests[1].method,
        requests[1].target,
        requests[1].headers["authorization"],
        JSON_OBJECT.validate_json(requests[1].body),
    ) in (
        (
            "POST",
            "/v1/chat/completions",
            f"Bearer {api_key}",
            {"model": model, "messages": _HEALTH_CHAT_MESSAGES[0], "max_tokens": 16},
        ),
        (
            "POST",
            "/v1/chat/completions",
            f"Bearer {api_key}",
            {"model": model, "messages": _HEALTH_CHAT_MESSAGES[1], "max_tokens": 16},
        ),
    ), requests[1].body.decode()


def test_background_health_results_are_scoped_to_the_key_models(gateway: Gateway, tmp_path: Path) -> None:
    public_a: Final = f"background-health-a-{uuid.uuid4().hex}"
    public_b: Final = f"background-health-b-{uuid.uuid4().hex}"
    provider_model_a: Final = f"openai/background-health-a-{uuid.uuid4().hex}"
    provider_model_b: Final = f"openai/background-health-b-{uuid.uuid4().hex}"
    wire_model_a: Final = provider_model_a.removeprefix("openai/")
    wire_model_b: Final = provider_model_b.removeprefix("openai/")
    model_id_a: Final = f"background-health-a-{uuid.uuid4().hex}"
    model_id_b: Final = f"background-health-b-{uuid.uuid4().hex}"
    api_key_a: Final = "synthetic-background-health-a"
    api_key_b: Final = "synthetic-background-health-b"

    with (
        scratch_database() as database_url,
        wire_server(lambda request: _background_health_reply(request, model=wire_model_a, api_key=api_key_a)) as wire_a,
        wire_server(lambda request: _background_health_reply(request, model=wire_model_b, api_key=api_key_b)) as wire_b,
    ):
        base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config: Final = {
            **base,
            "general_settings": {
                **base["general_settings"],
                "background_health_checks": True,
                "health_check_interval": 3600,
            },
            "model_list": [
                {
                    "model_name": public_a,
                    "litellm_params": {
                        "model": provider_model_a,
                        "api_base": f"{wire_a.url}/v1",
                        "api_key": api_key_a,
                    },
                    "model_info": {"id": model_id_a},
                },
                {
                    "model_name": public_b,
                    "litellm_params": {
                        "model": provider_model_b,
                        "api_base": f"{wire_b.url}/v1",
                        "api_key": api_key_b,
                    },
                    "model_info": {"id": model_id_b},
                },
            ],
        }
        config_path: Final = tmp_path / "background_health.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with owned_proxy(gateway, tmp_path, {"DATABASE_URL": database_url}, config=config_path) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=[public_a])
                scoped_health: Final = eventually(
                    lambda: candidate.request("GET", "/health", key=key),
                    lambda response: (
                        response.status_code == 200
                        and any(
                            endpoint.get("model_id") == model_id_a
                            for endpoint in response.json().get("healthy_endpoints", [])
                        )
                    ),
                    seconds=70,
                    return_last_on_timeout=True,
                )
                assert scoped_health.status_code == 200, scoped_health.text
                assert JSON_OBJECT.validate_json(scoped_health.content) == {
                    "healthy_endpoints": [{"model": provider_model_a, "model_id": model_id_a}],
                    "unhealthy_endpoints": [],
                    "healthy_count": 1,
                    "unhealthy_count": 0,
                }, scoped_health.text

                admin_health: Final = candidate.request("GET", "/health")
                assert admin_health.status_code == 200, admin_health.text
                assert JSON_OBJECT.validate_json(admin_health.content) == {
                    "healthy_endpoints": [
                        {"model": provider_model_a, "api_base": f"{wire_a.url}/v1", "model_id": model_id_a},
                        {"model": provider_model_b, "api_base": f"{wire_b.url}/v1", "model_id": model_id_b},
                    ],
                    "unhealthy_endpoints": [],
                    "healthy_count": 2,
                    "unhealthy_count": 0,
                }, admin_health.text

        _assert_background_health_requests(wire_a, model=wire_model_a, api_key=api_key_a)
        _assert_background_health_requests(wire_b, model=wire_model_b, api_key=api_key_b)


def _probes_sent_to(gateway: Gateway, handle: ScenarioHandle) -> list[tuple[str, JsonValue]]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        requests: Final = upstream.get("/__observations").json()["requests"]
    return [
        (string_value(request["path"]), request["body"])
        for request in map(object_value, requests)
        if string_value(request["path"]).startswith(f"/{handle.scenario_id}/")
    ]


def test_evaluation_mode_health_check_resolves_the_mode_from_the_cost_map_and_sends_the_default_probe(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _DECISIONS_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(model="perplexity/pplx-decider-v1-27b", api_base=handle.api_base())
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/decisions",
                {
                    "model": "pplx-decider-v1-27b",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]


def test_evaluation_mode_health_check_sends_the_configured_state_and_questions(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _CONFIGURED_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="typesafe/jev-custom",
            api_base=handle.api_base(),
            model_info={
                "mode": "evaluation",
                "health_check_params": {"state": _CONFIGURED_STATE, "questions": _CONFIGURED_QUESTIONS},
            },
        )
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/systemone",
                {"model": "jev-custom", "state": _CONFIGURED_STATE, "questions": _CONFIGURED_QUESTIONS},
            )
        ]


def test_evaluation_mode_health_check_of_the_self_hosted_strands_model_resolves_the_mode_from_the_cost_map(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _STRANDS_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="strands_decider/strands-decider-2B-hobson-v19", api_base=handle.api_base(), api_key=None
        )
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/systemone",
                {
                    "model": "strands-decider-2B-hobson-v19",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]


def _health_chat_reply(request: Request, *, api_key: str, model: str, status: int = 200) -> Reply:
    assert request.method == "POST"
    assert request.target == "/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {api_key}"
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body in (
        {"model": model, "messages": _HEALTH_CHAT_MESSAGES[0], "max_tokens": 16},
        {"model": model, "messages": _HEALTH_CHAT_MESSAGES[1], "max_tokens": 16},
    ), request.body.decode()
    if status != 200:
        return Reply(status=status, body=b'{"error":{"message":"synthetic health failure"}}')
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-health",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "healthy"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode(),
    )


def _create_team_health_deployment(
    gateway: Gateway,
    scenario: Scenario,
    *,
    team_id: str,
    public_model_name: str,
    upstream_url: str,
    api_key: str,
    access_group: str | None = None,
    api_version: str | None = None,
    aws_bedrock_runtime_endpoint: str | None = None,
) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": public_model_name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": api_key,
                "api_base": f"{upstream_url}/v1",
                **({"api_version": api_version} if api_version is not None else {}),
                **(
                    {"aws_bedrock_runtime_endpoint": aws_bedrock_runtime_endpoint}
                    if aws_bedrock_runtime_endpoint is not None
                    else {}
                ),
            },
            "model_info": {
                "team_id": team_id,
                **({"access_groups": [access_group]} if access_group is not None else {}),
            },
        },
    )
    model_info: Final = object_value(created["model_info"])
    identity: Final = string_value(model_info["id"])
    assert model_info["team_id"] == team_id, created
    assert model_info["team_public_model_name"] == public_model_name, created
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _health_failure_raw_request(endpoint: dict[str, JsonValue], upstream_url: str) -> dict[str, JsonValue]:
    raw_request: Final = object_value(endpoint["raw_request_typed_dict"])
    raw_request_body: Final = object_value(raw_request["raw_request_body"])
    expected_request_bodies: Final = tuple(
        {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 16,
            "extra_body": {},
        }
        for messages in _HEALTH_CHAT_MESSAGES
    )
    assert raw_request == {
        "raw_request_api_base": f"{upstream_url}/v1/",
        "raw_request_body": raw_request_body,
        "raw_request_headers": {"Authorization": "Be****ey"},
        "error": None,
    }, raw_request
    assert raw_request_body in expected_request_bodies, raw_request
    return raw_request


def test_team_scoped_health_probes_access_group_targets_and_hides_routing_fields(gateway: Gateway) -> None:
    with (
        wire_server(lambda request: _health_chat_reply(request, api_key="team-a-key", model="gpt-4o-mini")) as healthy,
        wire_server(
            lambda request: _health_chat_reply(request, api_key="team-a-key", model="gpt-4o-mini", status=500)
        ) as unhealthy,
        wire_server(
            lambda request: _health_chat_reply(request, api_key="team-b-key", model="gpt-4o-mini")
        ) as team_b_wire,
        gateway.scenario() as scenario,
    ):
        public_a: Final = f"team-a-{uuid.uuid4().hex}"
        public_b: Final = f"team-b-{uuid.uuid4().hex}"
        access_group: Final = f"health-group-{uuid.uuid4().hex}"
        team_a: Final = scenario.team(models=[public_a, public_b, access_group])
        team_b_id: Final = scenario.team(models=[public_b])
        user_id: Final = scenario.member(team_a)
        healthy_id: Final = _create_team_health_deployment(
            gateway,
            scenario,
            team_id=team_a,
            public_model_name=public_a,
            upstream_url=healthy.url,
            api_key="team-a-key",
            access_group=access_group,
            api_version="2024-10-21",
            aws_bedrock_runtime_endpoint=f"{healthy.url}/runtime",
        )
        unhealthy_id: Final = _create_team_health_deployment(
            gateway,
            scenario,
            team_id=team_a,
            public_model_name=public_a,
            upstream_url=unhealthy.url,
            api_key="team-a-key",
            access_group=access_group,
            api_version="2024-10-21",
            aws_bedrock_runtime_endpoint=f"{unhealthy.url}/runtime",
        )
        team_b_model_id: Final = _create_team_health_deployment(
            gateway,
            scenario,
            team_id=team_b_id,
            public_model_name=public_b,
            upstream_url=team_b_wire.url,
            api_key="team-b-key",
        )
        team_key: Final = scenario.key(
            team_id=team_a,
            user_id=user_id,
            models=[public_a, public_b],
        )
        access_group_key: Final = scenario.key(
            team_id=team_a,
            user_id=user_id,
            models=[access_group],
        )

        by_other_team_name: Final = gateway.request("GET", "/health", key=team_key, params={"model": public_b})
        assert by_other_team_name.status_code == 403, by_other_team_name.text
        assert by_other_team_name.json() == {
            "detail": {"error": f"key not allowed to health-check model {public_b}"}
        }, by_other_team_name.text
        by_other_team_id: Final = gateway.request("GET", "/health", key=team_key, params={"model_id": team_b_model_id})
        assert by_other_team_id.status_code == 403, by_other_team_id.text
        assert by_other_team_id.json() == {
            "detail": {"error": f"key not allowed to health-check model_id {team_b_model_id}"}
        }, by_other_team_id.text
        assert team_b_wire.drain() == ()

        public_model_report: Final = gateway.request("GET", "/health", key=team_key, params={"model": public_a})
        assert public_model_report.status_code == 200, public_model_report.text
        public_model_body: Final = public_model_report.json()
        healthy_endpoint: Final = {
            "model": "openai/gpt-4o-mini",
            "model_id": healthy_id,
        }
        public_model_unhealthy_endpoint: Final = object_value(public_model_body["unhealthy_endpoints"][0])
        public_model_error: Final = string_value(public_model_unhealthy_endpoint["error"])
        public_model_raw_request: Final = _health_failure_raw_request(public_model_unhealthy_endpoint, unhealthy.url)
        assert "synthetic health failure" in public_model_error, public_model_report.text
        for endpoint in (*public_model_body["healthy_endpoints"], *public_model_body["unhealthy_endpoints"]):
            assert {
                "api_base",
                "api_version",
                "aws_bedrock_runtime_endpoint",
            }.isdisjoint(endpoint), public_model_report.text
        assert public_model_unhealthy_endpoint == {
            "model": "openai/gpt-4o-mini",
            "model_id": unhealthy_id,
            "error": public_model_error,
            "raw_request_typed_dict": public_model_raw_request,
            "exception_status": 500,
        }, public_model_report.text
        assert public_model_body == {
            "healthy_endpoints": [healthy_endpoint],
            "unhealthy_endpoints": [public_model_unhealthy_endpoint],
            "healthy_count": 1,
            "unhealthy_count": 1,
        }, public_model_report.text
        assert len(healthy.drain()) == 1
        assert len(unhealthy.drain()) == 3

        expanded_group_report: Final = gateway.request("GET", "/health", key=access_group_key)
        assert expanded_group_report.status_code == 200, expanded_group_report.text
        expanded_group_body: Final = expanded_group_report.json()
        expanded_group_unhealthy_endpoint: Final = object_value(expanded_group_body["unhealthy_endpoints"][0])
        expanded_group_error: Final = string_value(expanded_group_unhealthy_endpoint["error"])
        expanded_group_raw_request: Final = _health_failure_raw_request(
            expanded_group_unhealthy_endpoint, unhealthy.url
        )
        assert "synthetic health failure" in expanded_group_error, expanded_group_report.text
        assert expanded_group_body == {
            "healthy_endpoints": [healthy_endpoint],
            "unhealthy_endpoints": [
                {
                    "model": "openai/gpt-4o-mini",
                    "model_id": unhealthy_id,
                    "error": expanded_group_error,
                    "raw_request_typed_dict": expanded_group_raw_request,
                    "exception_status": 500,
                }
            ],
            "healthy_count": 1,
            "unhealthy_count": 1,
        }, expanded_group_report.text
        for endpoint in (*expanded_group_body["healthy_endpoints"], *expanded_group_body["unhealthy_endpoints"]):
            assert {
                "api_base",
                "api_version",
                "aws_bedrock_runtime_endpoint",
            }.isdisjoint(endpoint), expanded_group_report.text
        assert len(healthy.drain()) == 1
        assert len(unhealthy.drain()) == 3
        assert team_b_wire.drain() == ()

        targeted: Final = gateway.request(
            "GET",
            "/health",
            key=team_key,
            params={"model": public_b, "model_id": unhealthy_id},
        )
        assert targeted.status_code == 503, targeted.text
        targeted_body: Final = targeted.json()
        targeted_endpoint: Final = object_value(targeted_body["unhealthy_endpoints"][0])
        targeted_error: Final = string_value(targeted_endpoint["error"])
        targeted_raw_request: Final = _health_failure_raw_request(targeted_endpoint, unhealthy.url)
        assert "synthetic health failure" in targeted_error, targeted.text
        assert targeted_body == {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [
                {
                    "model": "openai/gpt-4o-mini",
                    "model_id": unhealthy_id,
                    "error": targeted_error,
                    "raw_request_typed_dict": targeted_raw_request,
                    "exception_status": 500,
                }
            ],
            "healthy_count": 0,
            "unhealthy_count": 1,
        }, targeted.text
        assert healthy.drain() == ()
        assert len(unhealthy.drain()) == 3
        assert team_b_wire.drain() == ()

        admin: Final = gateway.request(
            "GET",
            "/health",
            params={"model": public_b, "model_id": unhealthy_id},
        )
        assert admin.status_code == 503, admin.text
        admin_body: Final = admin.json()
        admin_endpoint: Final = object_value(admin_body["unhealthy_endpoints"][0])
        admin_error: Final = string_value(admin_endpoint["error"])
        admin_raw_request: Final = _health_failure_raw_request(admin_endpoint, unhealthy.url)
        assert "synthetic health failure" in admin_error, admin.text
        assert admin_body == {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [
                {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{unhealthy.url}/v1",
                    "api_version": "2024-10-21",
                    "aws_bedrock_runtime_endpoint": f"{unhealthy.url}/runtime",
                    "model_id": unhealthy_id,
                    "error": admin_error,
                    "raw_request_typed_dict": admin_raw_request,
                    "exception_status": 500,
                }
            ],
            "healthy_count": 0,
            "unhealthy_count": 1,
        }, admin.text
        assert len(unhealthy.drain()) == 3

        unrestricted_team: Final = scenario.team(models=[])
        unrestricted_user: Final = scenario.member(unrestricted_team)
        unrestricted_key: Final = scenario.key(
            team_id=unrestricted_team,
            user_id=unrestricted_user,
            models=[],
        )
        shared_public_name: Final = f"team-shared-{uuid.uuid4().hex}"
        unrestricted_deployment_id: Final = _create_team_health_deployment(
            gateway,
            scenario,
            team_id=unrestricted_team,
            public_model_name=shared_public_name,
            upstream_url=healthy.url,
            api_key="team-a-key",
        )
        _create_team_health_deployment(
            gateway,
            scenario,
            team_id=team_b_id,
            public_model_name=shared_public_name,
            upstream_url=team_b_wire.url,
            api_key="team-b-key",
        )
        refused_unrestricted: Final = gateway.request(
            "GET",
            "/health",
            key=unrestricted_key,
            params={"model_id": team_b_model_id},
        )
        assert refused_unrestricted.status_code == 403, refused_unrestricted.text
        assert refused_unrestricted.json() == {
            "detail": {"error": f"key not allowed to health-check model_id {team_b_model_id}"}
        }, refused_unrestricted.text
        assert team_b_wire.drain() == ()

        unrestricted_report: Final = gateway.request(
            "GET",
            "/health",
            key=unrestricted_key,
            params={"model": shared_public_name},
        )
        assert unrestricted_report.status_code == 200, unrestricted_report.text
        unrestricted_body: Final = unrestricted_report.json()
        assert unrestricted_body == {
            "healthy_endpoints": [
                {
                    "model": "openai/gpt-4o-mini",
                    "model_id": unrestricted_deployment_id,
                }
            ],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
        }, unrestricted_report.text
        endpoint_ids: Final = tuple(
            endpoint["model_id"]
            for endpoint in (*unrestricted_body["healthy_endpoints"], *unrestricted_body["unhealthy_endpoints"])
        )
        assert team_b_model_id not in endpoint_ids, unrestricted_report.text
        for endpoint in (*unrestricted_body["healthy_endpoints"], *unrestricted_body["unhealthy_endpoints"]):
            assert {
                "api_base",
                "api_version",
                "aws_bedrock_runtime_endpoint",
            }.isdisjoint(endpoint), unrestricted_report.text
        assert len(healthy.drain()) == 1
        assert team_b_wire.drain() == ()


def _delete_health_credential_if_present(gateway: Gateway, credential_name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{credential_name}")
    assert response.status_code in (200, 404), response.text


def _health_connection_credential(gateway: Gateway, credential_name: str, api_key: str) -> None:
    created: Final = gateway.request(
        "POST",
        "/credentials",
        {
            "credential_name": credential_name,
            "credential_values": {"api_key": api_key},
            "credential_info": {},
        },
    )
    assert created.status_code == 200, created.text


def _create_health_connection_deployment(
    gateway: Gateway,
    scenario: Scenario,
    *,
    public_model_name: str,
    provider_model: str,
    api_base: str,
    api_key: str,
) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": public_model_name,
            "litellm_params": {
                "model": f"openai/{provider_model}",
                "api_base": api_base,
                "api_key": api_key,
            },
        },
    )
    model_info: Final = object_value(created["model_info"])
    model_id: Final = string_value(model_info["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_id


def _health_connection_response(
    request: Request,
    model: str,
    *,
    validate_request_body: bool = True,
    expected_temperature: float | None = None,
    expected_api_key: str = "synthetic-health-credential",
) -> Reply:
    response_body: Final = {
        "/v1/chat/completions": {
            "id": "chatcmpl-connection",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "healthy"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        "/v1/embeddings": {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
        "/v1/images/generations": {"created": 1700000000, "data": [{"b64_json": "c3ludGhldGljLWltYWdl"}]},
        "/v1/audio/transcriptions": {"text": "healthy"},
        "/v1/responses": {
            "id": "resp-health",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "id": "msg-health",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "healthy", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
        "/v1/messages": {
            "id": "msg-health",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "healthy"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    assert request.method == "POST"
    assert request.target in response_body, request.target
    if validate_request_body:
        if request.target == "/v1/audio/transcriptions":
            parts: Final = _multipart_parts(request)
            assert _multipart_text_fields(parts) == {
                "model": model,
                "response_format": "verbose_json",
            }, request.target
            files: Final = _multipart_file_fields(parts)
            assert tuple(files) == ("file",), files
            filename, content_type, audio_bytes = files["file"]
            assert filename == "audio_health_check.wav", files
            assert content_type == "audio/x-wav", files
            expected_audio: Final = (
                Path(__file__).resolve().parents[3]
                / "litellm"
                / "litellm_core_utils"
                / "audio_utils"
                / "audio_health_check.wav"
            ).read_bytes()
            assert audio_bytes == expected_audio, files
        else:
            body: Final = JSON_OBJECT.validate_json(request.body)
            if request.target == "/v1/chat/completions":
                assert body in (
                    {
                        "model": model,
                        "messages": _HEALTH_CHAT_MESSAGES[0],
                        "max_tokens": 16,
                        **({"temperature": expected_temperature} if expected_temperature is not None else {}),
                    },
                    {
                        "model": model,
                        "messages": _HEALTH_CHAT_MESSAGES[1],
                        "max_tokens": 16,
                        **({"temperature": expected_temperature} if expected_temperature is not None else {}),
                    },
                ), request.body.decode()
            elif request.target == "/v1/embeddings":
                assert body == {
                    "model": model,
                    "input": ["test from litellm"],
                }, request.body.decode()
            elif request.target == "/v1/images/generations":
                assert body == {"model": model, "prompt": "test from litellm"}, request.body.decode()
            elif request.target == "/v1/responses":
                assert body == {"model": model, "input": "test from litellm"}, request.body.decode()
            else:
                assert body in (
                    {
                        "model": model,
                        "messages": _HEALTH_CHAT_MESSAGES[0],
                        "max_tokens": 16,
                        "stream": False,
                    },
                    {
                        "model": model,
                        "messages": _HEALTH_CHAT_MESSAGES[1],
                        "max_tokens": 16,
                        "stream": False,
                    },
                ), request.body.decode()
    if request.target == "/v1/messages":
        assert request.headers["x-api-key"] == "synthetic-health-credential", request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert "authorization" not in request.headers, request.headers
    else:
        assert request.headers["authorization"] == f"Bearer {expected_api_key}", request.headers
    return Reply(body=json.dumps(response_body[request.target]).encode())


def test_health_test_connection_modes_use_stored_credentials_and_reject_environment_references(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        for mode, provider in (
            ("chat", "openai"),
            ("responses", "openai"),
            ("anthropic_messages", "anthropic"),
        ):
            model = f"{provider}/health-{mode}-{uuid.uuid4().hex}"
            provider_model = model.split("/", maxsplit=1)[1]
            with wire_server(
                lambda request, expected_model=provider_model: _health_connection_response(request, expected_model)
            ) as wire:
                mode_credential_name = f"health-{uuid.uuid4().hex}"
                _health_connection_credential(gateway, mode_credential_name, "synthetic-health-credential")
                scenario.cleanups.callback(_delete_health_credential_if_present, gateway, mode_credential_name)
                base = wire.url if mode == "anthropic_messages" else f"{wire.url}/v1"
                response = gateway.request(
                    "POST",
                    "/health/test_connection",
                    {
                        "litellm_params": {
                            "model": model,
                            "api_base": base,
                            "litellm_credential_name": mode_credential_name,
                        },
                        "mode": mode,
                    },
                )
                assert response.status_code == 200, response.text
                assert JSON_OBJECT.validate_json(response.content) == {
                    "status": "success",
                    "result": {"model": model, "api_base": base},
                }, response.text
                requests = wire.drain()
                assert len(requests) == 1, requests
                assert (
                    requests[0].target
                    == {
                        "chat": "/v1/chat/completions",
                        "embedding": "/v1/embeddings",
                        "image_generation": "/v1/images/generations",
                        "audio_transcription": "/v1/audio/transcriptions",
                        "responses": "/v1/responses",
                        "anthropic_messages": "/v1/messages",
                    }[mode]
                ), requests

        configured_provider_model: Final = f"health-credential-{uuid.uuid4().hex}"
        configured_model: Final = f"openai/{configured_provider_model}"
        with (
            wire_server(
                lambda request: _health_connection_response(
                    request, configured_provider_model, expected_temperature=0.2
                )
            ) as submitted_wire,
            wire_server(lambda request: Reply(status=500)) as configured_wire,
        ):
            configured_credential_name: Final = f"health-submitted-{uuid.uuid4().hex}"
            _health_connection_credential(gateway, configured_credential_name, "synthetic-health-credential")
            scenario.cleanups.callback(_delete_health_credential_if_present, gateway, configured_credential_name)
            scenario.model(
                model=configured_model,
                api_base=f"{configured_wire.url}/v1",
                api_key="synthetic-config-credential",
                temperature=0.9,
            )
            submitted_base: Final = f"{submitted_wire.url}/v1"
            submitted: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {
                    "litellm_params": {
                        "model": configured_model,
                        "api_base": submitted_base,
                        "litellm_credential_name": configured_credential_name,
                        "temperature": 0.2,
                    },
                    "model_info": {"mode": "chat"},
                    "mode": "chat",
                },
            )
            assert submitted.status_code == 200, submitted.text
            assert JSON_OBJECT.validate_json(submitted.content) == {
                "status": "success",
                "result": {"model": configured_model, "api_base": submitted_base},
            }, submitted.text
            submitted_requests: Final = submitted_wire.drain()
            assert len(submitted_requests) == 1, submitted.text
            assert submitted_requests[0].headers["authorization"] == "Bearer synthetic-health-credential", (
                submitted.text
            )
            assert JSON_OBJECT.validate_json(submitted_requests[0].body) in (
                {
                    "model": configured_provider_model,
                    "messages": _HEALTH_CHAT_MESSAGES[0],
                    "max_tokens": 16,
                    "temperature": 0.2,
                },
                {
                    "model": configured_provider_model,
                    "messages": _HEALTH_CHAT_MESSAGES[1],
                    "max_tokens": 16,
                    "temperature": 0.2,
                },
            ), submitted.text
            assert configured_wire.drain() == ()

            rejected: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {
                    "litellm_params": {
                        "model": configured_model,
                        "api_base": submitted_base,
                        "api_key": "os.environ/X",
                    },
                    "model_info": {"mode": "chat"},
                    "mode": "chat",
                },
            )
            assert rejected.status_code == 400, rejected.text
            assert JSON_OBJECT.validate_json(rejected.content) == {
                "detail": {"error": "Environment variable references are not permitted in request parameters."}
            }, rejected.text
            assert submitted_wire.drain() == ()
            assert configured_wire.drain() == ()

            rejected_model_info: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {
                    "litellm_params": {
                        "model": configured_model,
                        "api_base": submitted_base,
                    },
                    "model_info": {"mode": "chat", "base_model": "os.environ/X"},
                    "mode": "chat",
                },
            )
            assert rejected_model_info.status_code == 400, rejected_model_info.text
            assert JSON_OBJECT.validate_json(rejected_model_info.content) == {
                "detail": {"error": "Environment variable references are not permitted in request parameters."}
            }, rejected_model_info.text
            assert submitted_wire.drain() == ()
            assert configured_wire.drain() == ()


def test_health_test_connection_uses_model_info_id_for_duplicate_provider_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        provider_model: Final = f"health-duplicate-{uuid.uuid4().hex}"
        public_model_name: Final = f"health-duplicate-public-{uuid.uuid4().hex}"
        api_key_a: Final = "synthetic-health-credential-a"
        api_key_b: Final = "synthetic-health-credential-b"
        with (
            wire_server(
                lambda request: _health_connection_response(request, provider_model, expected_api_key=api_key_a)
            ) as wire_a,
            wire_server(
                lambda request: _health_connection_response(request, provider_model, expected_api_key=api_key_b)
            ) as wire_b,
        ):
            api_base_a: Final = f"{wire_a.url}/v1"
            api_base_b: Final = f"{wire_b.url}/v1"
            model_id_a: Final = _create_health_connection_deployment(
                gateway,
                scenario,
                public_model_name=public_model_name,
                provider_model=provider_model,
                api_base=api_base_a,
                api_key=api_key_a,
            )
            model_id_b: Final = _create_health_connection_deployment(
                gateway,
                scenario,
                public_model_name=public_model_name,
                provider_model=provider_model,
                api_base=api_base_b,
                api_key=api_key_b,
            )
            model: Final = f"openai/{provider_model}"
            request_body: Final = {"litellm_params": {"model": model}}

            response_a: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {**request_body, "model_info": {"id": model_id_a}, "mode": "chat"},
            )
            assert response_a.status_code == 200, response_a.text
            assert JSON_OBJECT.validate_json(response_a.content) == {
                "status": "success",
                "result": {"model": model, "api_base": api_base_a},
            }, response_a.text
            requests_a: Final = wire_a.drain()
            assert len(requests_a) == 1, response_a.text
            assert (
                requests_a[0].method,
                requests_a[0].target,
                requests_a[0].headers["authorization"],
                JSON_OBJECT.validate_json(requests_a[0].body),
            ) in (
                (
                    "POST",
                    "/v1/chat/completions",
                    f"Bearer {api_key_a}",
                    {"model": provider_model, "messages": _HEALTH_CHAT_MESSAGES[0], "max_tokens": 16},
                ),
                (
                    "POST",
                    "/v1/chat/completions",
                    f"Bearer {api_key_a}",
                    {"model": provider_model, "messages": _HEALTH_CHAT_MESSAGES[1], "max_tokens": 16},
                ),
            ), requests_a[0].body.decode()
            assert wire_b.drain() == ()

            response_b: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {**request_body, "model_info": {"id": model_id_b}, "mode": "chat"},
            )
            assert response_b.status_code == 200, response_b.text
            assert JSON_OBJECT.validate_json(response_b.content) == {
                "status": "success",
                "result": {"model": model, "api_base": api_base_b},
            }, response_b.text
            requests_b: Final = wire_b.drain()
            assert len(requests_b) == 1, response_b.text
            assert (
                requests_b[0].method,
                requests_b[0].target,
                requests_b[0].headers["authorization"],
                JSON_OBJECT.validate_json(requests_b[0].body),
            ) in (
                (
                    "POST",
                    "/v1/chat/completions",
                    f"Bearer {api_key_b}",
                    {"model": provider_model, "messages": _HEALTH_CHAT_MESSAGES[0], "max_tokens": 16},
                ),
                (
                    "POST",
                    "/v1/chat/completions",
                    f"Bearer {api_key_b}",
                    {"model": provider_model, "messages": _HEALTH_CHAT_MESSAGES[1], "max_tokens": 16},
                ),
            ), requests_b[0].body.decode()
            assert wire_a.drain() == ()


@pytest.mark.parametrize(
    ("mode", "expected_path"),
    (
        ("embedding", "/v1/embeddings"),
        ("image_generation", "/v1/images/generations"),
        ("audio_transcription", "/v1/audio/transcriptions"),
    ),
)
def test_health_test_connection_non_chat_modes_do_not_receive_chat_max_tokens(
    gateway: Gateway, mode: str, expected_path: str
) -> None:
    pytest.skip(
        "BUG: /health/test_connection sends chat max_tokens 16 to embedding, "
        "image_generation and audio_transcription probes"
    )
    with gateway.scenario() as scenario:
        provider_model: Final = f"health-{mode}-{uuid.uuid4().hex[:8]}"
        model: Final = f"openai/{provider_model}"
        with wire_server(
            lambda request, expected_model=provider_model: _health_connection_response(
                request, expected_model, validate_request_body=False
            )
        ) as wire:
            credential_name: Final = f"health-{uuid.uuid4().hex}"
            _health_connection_credential(gateway, credential_name, "synthetic-health-credential")
            scenario.cleanups.callback(_delete_health_credential_if_present, gateway, credential_name)
            api_base: Final = f"{wire.url}/v1"
            response: Final = gateway.request(
                "POST",
                "/health/test_connection",
                {
                    "litellm_params": {
                        "model": model,
                        "api_base": api_base,
                        "litellm_credential_name": credential_name,
                    },
                    "mode": mode,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json() == {
                "status": "success",
                "result": {"model": model, "api_base": api_base},
            }, response.text

            requests: Final = wire.drain()
            assert len(requests) == 1, response.text
            request: Final = requests[0]
            assert request.target == expected_path, response.text
            if mode == "embedding":
                assert JSON_OBJECT.validate_json(request.body) == {
                    "model": provider_model,
                    "input": ["test from litellm"],
                }, response.text
            elif mode == "image_generation":
                assert JSON_OBJECT.validate_json(request.body) == {
                    "model": provider_model,
                    "prompt": "test from litellm",
                }, response.text
            else:
                parts: Final = _multipart_parts(request)
                assert _multipart_text_fields(parts) == {
                    "model": provider_model,
                    "response_format": "verbose_json",
                }, response.text
                files: Final = _multipart_file_fields(parts)
                assert tuple(files) == ("file",), response.text
                assert files["file"] == (
                    "audio_health_check.wav",
                    "audio/x-wav",
                    (
                        Path(__file__).resolve().parents[3]
                        / "litellm"
                        / "litellm_core_utils"
                        / "audio_utils"
                        / "audio_health_check.wav"
                    ).read_bytes(),
                ), response.text
