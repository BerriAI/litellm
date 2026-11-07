"""Two Vertex deployments share a base model but live in different projects with different
buckets. An unmanaged gs:// batch created on the second one must be polled through that
deployment (LIT-10326). Before the fix the cost poller matched the base model name in the
gs:// path to the first deployment in the list and asked the wrong project for the job."""

import json
import re
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.process import owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

LOCATION: Final = "us-central1"
BASE_MODEL: Final = "gemini-2.5-flash"
TEAMS: Final = ("a", "b")
_JOB_PATH: Final = re.compile(
    r"^/v1/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/batchPredictionJobs(?:/(?P<job>\d+))?$"
)


def _project(team: str) -> str:
    return f"scripted-project-{team}"


def _deployment(team: str) -> str:
    return f"{BASE_MODEL}-team-{team}"


def _input_file(team: str) -> str:
    return f"gs://scripted-bucket-{team}/litellm-vertex-files/publishers/google/models/{BASE_MODEL}/{uuid4().hex}/input.jsonl"


def _job_number(team: str) -> str:
    return f"{TEAMS.index(team) + 1}{uuid4().int % 10**17:017d}"


def _vertex_peer(jobs: dict[str, tuple[str, str]]) -> Callable[[Request], Reply]:
    """A Vertex double that remembers which project each job was created in. A retrieve
    from any other project answers the 404 real Vertex gives (the job does not exist
    there), a retrieve from the right one reports the job still running."""

    def respond(request: Request) -> Reply:
        path: Final = _JOB_PATH.match(request.target.split("?", 1)[0])
        assert path is not None, request.target
        assert request.headers.get("authorization") == "Bearer scripted-token", request.headers
        project, location, job = path.group("project"), path.group("location"), path.group("job")
        if request.method == "POST":
            team: Final = project.rsplit("-", 1)[-1]
            uri: Final = re.search(r'"(gs://[^"]+)"', request.body.decode())
            assert uri is not None and uri.group(1).startswith(f"gs://scripted-bucket-{team}/"), (project, request.body)
            created, created_uri = jobs.setdefault(project, (_job_number(team), uri.group(1)))
            return Reply(body=json.dumps(_job(project, location, created, created_uri, "JOB_STATE_PENDING")).encode())
        assert request.method == "GET" and job is not None, (request.method, request.target)
        known: Final = jobs.get(project)
        if known is None or known[0] != job:
            return Reply(
                status=404,
                body=json.dumps(
                    {
                        "error": {
                            "code": 404,
                            "message": f"The BatchPredictionJob does not exist: {job}",
                            "status": "NOT_FOUND",
                        }
                    }
                ).encode(),
            )
        return Reply(body=json.dumps(_job(project, location, job, known[1], "JOB_STATE_RUNNING")).encode())

    return respond


def _job(project: str, location: str, job: str, input_uri: str, state: str) -> dict[str, JsonValue]:
    return {
        "name": f"projects/{project}/locations/{location}/batchPredictionJobs/{job}",
        "displayName": "litellm-vertex-batch-scripted",
        "model": f"publishers/google/models/{BASE_MODEL}",
        "inputConfig": {"gcsSource": {"uris": [input_uri]}, "instancesFormat": "jsonl"},
        "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": input_uri.rsplit("/", 1)[0]}},
        "state": state,
        "createTime": "2026-10-30T20:00:00.000000Z",
        "updateTime": "2026-10-30T20:00:00.000000Z",
    }


def _owned_config(path: Path, gateway: Gateway, wire_url: str) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _deployment(team),
                        "litellm_params": {
                            "model": f"vertex_ai/{BASE_MODEL}",
                            "api_base": wire_url,
                            "vertex_project": _project(team),
                            "vertex_location": LOCATION,
                            "vertex_credentials": service_account_json(_project(team), gateway.upstream_url.rstrip("/")),
                            "gcs_bucket_name": f"scripted-bucket-{team}",
                        },
                    }
                    for team in TEAMS
                ],
                "general_settings": {
                    **object_value(config.get("general_settings") or {}),
                    "track_unmanaged_batch_cost": True,
                },
                "litellm_settings": {
                    **object_value(config.get("litellm_settings") or {}),
                    "enable_loadbalancing_on_batch_endpoints": True,
                    "user_url_allowed_hosts": ["127.0.0.1"],
                },
            }
        )
    )
    return path


def _create_batch(gateway: Gateway, key: str, team: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": _input_file(team),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": _deployment(team),
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _retrieve_targets(requests: tuple[Request, ...], job: str) -> tuple[str, ...]:
    return tuple(request.target for request in requests if request.method == "GET" and request.target.endswith(f"/{job}"))


@pytest.mark.timeout(240)
def test_unmanaged_batch_on_the_second_deployment_sharing_a_base_model_is_polled_through_its_own_project(
    gateway: Gateway, tmp_path: Path
) -> None:
    jobs: Final[dict[str, tuple[str, str]]] = {}  # mutable-ok: the wire double records which project created each job
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_vertex_peer(jobs)))
        config: Final = _owned_config(tmp_path / "two-vertex-teams.yaml", gateway, wire.url)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        key: Final = scenario.key(models=[_deployment(team) for team in TEAMS])

        batch_a: Final = _create_batch(owned.gateway, key, "a")
        batch_b: Final = _create_batch(owned.gateway, key, "b")
        assert (batch_a, batch_b) == (jobs[_project("a")][0], jobs[_project("b")][0]), (batch_a, batch_b, jobs)

        seen: Final[list[Request]] = []  # mutable-ok: drain empties the wire queue, so samples accumulate across retries

        def polled_so_far() -> tuple[Request, ...]:
            seen.extend(wire.drain())
            return tuple(seen)

        polled: Final = eventually(
            polled_so_far,
            lambda requests: bool(_retrieve_targets(requests, batch_b)) and bool(_retrieve_targets(requests, batch_a)),
            seconds=90,
        )
        for team, batch in (("a", batch_a), ("b", batch_b)):
            assert all(
                target.startswith(f"/v1/projects/{_project(team)}/") for target in _retrieve_targets(polled, batch)
            ), (team, _retrieve_targets(polled, batch))

