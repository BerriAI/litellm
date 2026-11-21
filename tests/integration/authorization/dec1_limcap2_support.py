"""Shared pieces for the ``test_dec1_limcap2_*`` contracts (rate limits and model capacity, round 2).

These files fill the "Not covered" lists of the ``test_dec1_ratelim_*``, ``test_dec1_capacity_*``,
``test_dec1_budlim2_*`` and ``test_dec1_passthru_*`` families. The provider, probes, refusal assertions and Redis
readers come from ``dec1_limits_support`` and ``dec1_ratelim_support``; the capacity proxy and its texts come from
``tests/integration/routing/dec1_capacity_support.py``. None of those modules is changed.

What is added here: a provider that also answers ``/audio/transcriptions``, a streaming Anthropic provider, a small
stateful batch provider (files, batches, cancel), customer organization / team / member creation through
``/management/v1``, a blocking content filter guardrail, a proxy on a Redis this test owns, and Prometheus helpers.

Every literal in these tests is read from the Python product on main (49b2058c). The Rust ports in flight
(#51537 for ``parallel_request_limiter_v3``, #51540 for ``dynamic_rate_limiter_v3``) rebuild these paths, which is
why the texts are pinned in full wherever the code makes them deterministic.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.anthropic_sse import message_json, message_stream, stream_reply
from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration._support.wire import Reply, Request
from tests.integration.authorization.dec1_limits_support import (
    ANTHROPIC_MODEL,
    DEFAULT_USAGE,
    Usage,
    limits_config,
    probe,
    provider,
)

CUSTOMERS: Final = "/management/v1"
MEMBER_HEADER: Final = "x-litellm-customer-member-id"
ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
TEAM_HEADER: Final = "x-litellm-customer-team-id"
TRANSCRIPTION_TEXT: Final = "limcap2 transcript"
STREAM_INPUT_TOKENS: Final = 5
STREAM_OUTPUT_TOKENS: Final = 3
UNVERIFIABLE: Final = (
    "Rate limit enforcement unavailable: request counters could not be verified against Redis, and "
    "fail_closed_rate_limit_enforcement is enabled, so the request was rejected to avoid exceeding the "
    "configured rate limit. Retry shortly."
)
"""``RATE_LIMIT_UNVERIFIABLE_MESSAGE`` (parallel_request_limiter_v3.py:410)."""


def unique(label: str) -> str:
    return f"limcap2-{label}-{uuid.uuid4().hex[:10]}"


def media_provider(usage: Usage = DEFAULT_USAGE) -> Callable[[Request], Reply]:
    """The limits provider, plus ``POST /audio/transcriptions`` answered with a fixed transcript."""
    base: Final = provider(usage)

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.target.split("?", 1)[0].endswith("/audio/transcriptions"):
            return Reply(body=json.dumps({"text": TRANSCRIPTION_TEXT}).encode())
        return base(request)

    return respond


def anthropic_stream_provider() -> Callable[[Request], Reply]:
    """An Anthropic provider. A streamed ``/v1/messages`` gets ``message_stream`` (5 input tokens in
    ``message_start``, 3 output tokens in ``message_delta``); a plain one gets ``message_json`` with the same usage."""

    def respond(request: Request) -> Reply:
        identity: Final = f"msg_{uuid.uuid4().hex[:12]}"
        if request.method == "POST" and JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return stream_reply(message_stream(identity, ANTHROPIC_MODEL, "limcap2"))
        return Reply(body=message_json(identity, ANTHROPIC_MODEL, "limcap2"))

    return respond


def messages_stream(gateway: Gateway, model: str, key: str, *, max_tokens: int = 16) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": probe()}],
        },
        key=key,
        headers={"anthropic-version": "2023-06-01"},
    )


class BatchProvider:
    """A stateful OpenAI files and batches provider.

    ``POST /v1/files`` stores the upload, ``GET /v1/files/<id>/content`` returns it, ``POST /v1/batches`` answers
    ``validating`` and ``POST /v1/batches/<id>/cancel`` or a later ``GET /v1/batches/<id>`` answers ``cancelled``
    once ``finish(batch_id)`` was called. Chat posts are answered by the limits provider.
    """

    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._files: Final[dict[str, bytes]] = {}
        self._finished: Final[set[str]] = set()
        self._chat: Final = provider()

    def finish(self, batch_id: str) -> None:
        with self._lock:
            self._finished.add(batch_id)

    def _batch(self, batch_id: str, status: str) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": batch_id,
                    "object": "batch",
                    "endpoint": "/v1/chat/completions",
                    "errors": None,
                    "input_file_id": "file-limcap2",
                    "completion_window": "24h",
                    "status": status,
                    "output_file_id": None,
                    "error_file_id": None,
                    "created_at": 1,
                    "expires_at": 1,
                    "request_counts": {"total": 0, "completed": 0, "failed": 0},
                    "metadata": None,
                }
            ).encode()
        )

    def __call__(self, request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "POST" and path.endswith("/files"):
            identity = f"file-{uuid.uuid4().hex[:12]}"
            with self._lock:
                self._files[identity] = request.body
            return Reply(
                body=json.dumps(
                    {
                        "id": identity,
                        "object": "file",
                        "purpose": "batch",
                        "bytes": len(request.body),
                        "created_at": 1,
                        "filename": "in.jsonl",
                        "status": "processed",
                    }
                ).encode()
            )
        if request.method == "GET" and "/files/" in path and path.endswith("/content"):
            identity = path.rsplit("/", 2)[-2]
            with self._lock:
                stored = self._files.get(identity, b"")
            start = stored.find(b'{"custom_id"')
            end = stored.rfind(b"}\n")
            return Reply(content_type="application/jsonl", body=stored[start : end + 2] if start >= 0 else stored)
        if request.method == "POST" and path.endswith("/batches"):
            return self._batch(f"batch_{uuid.uuid4().hex[:12]}", "validating")
        if "/batches/" in path:
            identity = path.split("/batches/", 1)[1].split("/", 1)[0]
            with self._lock:
                finished = identity in self._finished or path.endswith("/cancel")
            return self._batch(identity, "cancelled" if finished else "in_progress")
        return self._chat(request)


def batch_lines(model: str, rows: int, *, words: int = 20) -> bytes:
    """``rows`` chat rows naming ``model``, each with ``words`` repetitions of ``limcap2`` as the prompt."""
    return "".join(
        json.dumps(
            {
                "custom_id": f"r{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": [{"role": "user", "content": "limcap2 " * words}]},
            },
            separators=(",", ":"),
        )
        + "\n"
        for index in range(rows)
    ).encode()


def submit_batch(gateway: Gateway, model: str, key: str, rows: int, *, words: int = 20) -> httpx.Response:
    """Upload ``rows`` rows for ``model`` through ``/v1/files`` and create a batch from the file."""
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": model},
        {"file": ("in.jsonl", batch_lines(model, rows, words=words), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )


def batch_id(response: httpx.Response) -> str:
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def transcribe(gateway: Gateway, model: str, key: str, *, metadata: str | None = None) -> httpx.Response:
    """A multipart ``/v1/audio/transcriptions`` call whose ``metadata`` form field is sent as given (a raw string)."""
    fields: Final[dict[str, str]] = {"model": model, **({"metadata": metadata} if metadata is not None else {})}
    return gateway.request_multipart(
        "/v1/audio/transcriptions", fields, {"file": ("probe.wav", b"RIFF0000WAVEfmt limcap2", "audio/wav")}, key=key
    )


def create_customer(
    gateway: Gateway,
    scenario: Scenario,
    *,
    org_budget: Mapping[str, JsonValue] | None = None,
    team_budget: Mapping[str, JsonValue] | None = None,
    member_budget: Mapping[str, JsonValue] | None = None,
) -> tuple[str, str, str, str]:
    """One customer organization (with an alias), one customer team under it and one member of that team,
    created through ``/management/v1``. Returns ``(org_id, org_alias, team_id, member_id)``; the organization is
    deleted with ``cascade=true`` when the scenario ends."""
    suffix: Final = uuid.uuid4().hex[:10]
    org_id, alias, team_id, member_id = (
        f"limcap2-org-{suffix}",
        f"limcap2-alias-{suffix}",
        f"limcap2-cteam-{suffix}",
        f"limcap2-member-{suffix}",
    )
    member_template: Final[dict[str, JsonValue]] = {"name": "member", **dict(member_budget or {})}
    bodies: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        (
            "customer_organizations",
            {
                "customer_org_id": org_id,
                "alias": alias,
                "default_budgets": [member_template],
                "default_budget_template": "member",
                **({"budget": dict(org_budget)} if org_budget is not None else {}),
            },
        ),
        (
            "customer_teams",
            {
                "customer_team_id": team_id,
                "parent_org_id": org_id,
                **({"budget": dict(team_budget)} if team_budget is not None else {}),
            },
        ),
        ("customer_members", {"customer_member_id": member_id, "parent_team_ids": [team_id]}),
    )
    for collection, body in bodies:
        created = gateway.request("POST", f"{CUSTOMERS}/{collection}", body)
        assert created.status_code == 201, f"{collection}: {created.status_code} {created.text}"
        if collection == "customer_organizations":
            scenario.cleanups.callback(
                gateway.request, "DELETE", f"{CUSTOMERS}/customer_organizations/{org_id}?cascade=true"
            )
    return org_id, alias, team_id, member_id


def blocking_guardrail(gateway: Gateway, scenario: Scenario, word: str) -> str:
    """A ``litellm_content_filter`` guardrail in ``pre_call`` mode that blocks ``word``, created through
    ``POST /guardrails`` (guardrail names given to keys must exist first) and deleted when the scenario ends."""
    name: Final = unique("guard")
    created: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "pre_call",
                    "default_on": False,
                    "blocked_words": [{"keyword": word, "action": "BLOCK"}],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    identity: Final = string_value(JSON_OBJECT.validate_json(created.content)["guardrail_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/guardrails/{identity}")
    return name


@contextmanager
def proxy_on_own_redis(
    gateway: Gateway, directory: Path, general_settings: Mapping[str, JsonValue] | None = None
) -> Iterator[tuple[Gateway, OwnedRedis]]:
    """A proxy whose only Redis is one this test owns and can stop (same rig as
    ``test_dec1_limits_multi_instance.py``)."""
    with owned_redis(directory / "redis") as redis:
        config: Final = limits_config(directory, general_settings=dict(general_settings or {}))
        with owned_proxy(
            gateway, directory, {"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)}, config=config
        ) as candidate:
            yield candidate, redis


def series(samples: Sequence[Sample], name: str, **labels: str) -> tuple[Sample, ...]:
    """Every sample of ``name`` whose labels include ``labels``."""
    return tuple(
        sample
        for sample in samples
        if sample.name == name and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def metric_total(gateway: Gateway, name: str, expected: float, seconds: float = 30, **labels: str) -> float:
    """Poll ``/metrics`` until the summed value of ``name`` for ``labels`` equals ``expected``; return the last sum."""
    return eventually(
        lambda: sum(sample.value for sample in series(scrape(gateway), name, **labels)),
        lambda total: total == expected,
        seconds=seconds,
        return_last_on_timeout=True,
    )
