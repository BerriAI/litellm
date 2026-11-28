import asyncio
import copy
import itertools
import json
import os
import re
from collections.abc import AsyncIterable, Awaitable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.forward_proxy import ForwardProxy, refusing_forward_proxy
from integration._support.parameter_matrix import (
    AMBIENT,
    BRIDGE_SESSION_STATE,
    SAMPLES,
    Cell,
    Endpoint,
    Group,
    Model,
    Sample,
    chat_name,
)
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import PROVIDER_UNVALIDATED_PARAMS

_PROMPT: Final = "Reply with the single word ok"
_PIXEL: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_DEADLINE_SECONDS: Final = 120
_CANDIDATES: Final = 3
_REASON_LENGTH: Final = 400
SILENT: Final = "LiteLLM answered without sending the provider a request"
_ELSEWHERE: Final = "LiteLLM sent the request to {hosts} instead of the api_base it was given"
_HANDLED_BY_LITELLM: Final = PROVIDER_UNVALIDATED_PARAMS | {"metadata"}
_PROMPT_CONTENT: Final = frozenset({"instructions", "system"})
_SENT_ONLY_TO_THE_PROVIDERS_OWN_HOST: Final = frozenset(
    {("mistral", "prompt_cache_key"), ("codestral", "prompt_cache_key")}
)
_FIXED_HOST_PROVIDERS: Final = frozenset({"replicate"})
_FIXED_HOST_MODELS: Final = frozenset({"vertex_ai/gemma-4-26b-a4b-it", "vertex_ai/deep-research-preview-04-2026"})
_VOLATILE: Final = re.compile(
    r"authorization|key|token|date|amz|signature|content-length|stainless|idempot|request-id|cookie|user-agent"
    r"|trace|host|accept|connection|x-goog-user|content-type",
    re.IGNORECASE,
)
_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EXTRACT: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"word": {"type": "string", "description": "the single word the user asks for"}},
    "required": ["word"],
    "additionalProperties": False,
}
_SCALEDOWN_INPUTS: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "classify": {"labels": [{"name": "ok", "rubric": "The text is the single word ok"}]},
        "decisions": {"questions": {"word": {"type": "choice", "criteria": {"ok": "The text is the single word ok"}}}},
    }
)

_TOKEN: Final = Reply(
    status=200,
    body=json.dumps({"access_token": "matrix-token", "expires_in": 3600, "token_type": "Bearer"}).encode(),
)
_CONVERSATION: Final = Reply(status=201, body=json.dumps({"id": "matrix-conversation"}).encode())
_REFUSAL: Final = Reply(
    status=400,
    body=json.dumps(
        {
            "error": {
                "message": "matrix peer refuses every request",
                "type": "invalid_request_error",
                "code": "matrix_peer",
            }
        }
    ).encode(),
)


def _peer(request: Request) -> Reply:
    if request.target.startswith("/_oauth/"):
        return _TOKEN
    if request.target.endswith("/copilot/conversations"):
        return _CONVERSATION
    return _REFUSAL


class _Rerouted(httpx.AsyncBaseTransport):
    def __init__(self, peer: str) -> None:
        self._peer: Final = httpx.URL(peer)
        self._inner: Final = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(
            httpx.Request(
                request.method,
                self._peer.copy_with(raw_path=request.url.raw_path),
                headers=[(name, value) for name, value in request.headers.raw if name.lower() != b"host"],
                content=await request.aread(),
            )
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass(frozen=True, slots=True, eq=False)
class Rig:
    runner: asyncio.Runner
    provider: Wire
    elsewhere: ForwardProxy
    rerouted: AsyncHTTPHandler
    credentials: Mapping[str, Mapping[str, str]]
    environment: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Failure:
    reason: str
    refusal: bool


@dataclass(frozen=True, slots=True)
class Exchange:
    requests: tuple[Request, ...]
    strayed: frozenset[str]
    failure: Failure | None


@dataclass(frozen=True, slots=True)
class Baseline:
    model: Model
    view: Mapping[str, str]
    unstable: frozenset[str]
    beside: frozenset[str]


@dataclass(frozen=True, slots=True)
class Forwarded:
    pass


@dataclass(frozen=True, slots=True)
class Dropped:
    pass


@dataclass(frozen=True, slots=True)
class Refused:
    reason: str


@dataclass(frozen=True, slots=True)
class Broken:
    reason: str


@dataclass(frozen=True, slots=True)
class Unbuildable:
    model: Model
    reason: str


@dataclass(frozen=True, slots=True)
class Unobservable:
    model: Model


@dataclass(frozen=True, slots=True)
class Unsampled:
    pass


Outcome = Forwarded | Dropped | Refused | Broken | Unbuildable | Unobservable | Unsampled


class GapKind(Enum):
    LISTED_NOT_SENT = "listed_not_sent"
    LISTED_REFUSED = "listed_refused"
    UNLISTED_VANISHES = "unlisted_vanishes"
    REFUSED_WITH_DROP = "refused_with_drop"
    BROKEN = "broken"
    UNSAMPLED = "unsampled"


@dataclass(frozen=True, slots=True)
class Gap:
    kind: GapKind
    problem: str


def _credentials(peer: str) -> Mapping[str, Mapping[str, str]]:
    signing_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return MappingProxyType(
        {
            "vertex_ai": {
                "vertex_credentials": service_account_json("matrix-project", peer),
                "vertex_project": "matrix-project",
                "vertex_location": "global",
            },
            "oci": {
                "oci_user": "ocid1.user.oc1..matrix",
                "oci_fingerprint": "00:00",
                "oci_tenancy": "ocid1.tenancy.oc1..matrix",
                "oci_compartment_id": "ocid1.compartment.oc1..matrix",
                "oci_key": signing_key,
            },
            "sagemaker": {"sagemaker_base_url": f"{peer}/sagemaker"},
            "nlp_cloud": {"api_base": f"{peer}/"},
        }
    )


def _environment(scratch: Path, elsewhere: str) -> Mapping[str, str]:
    tokens: Final = scratch / "github-copilot"
    tokens.mkdir()
    (tokens / "access-token").write_text("matrix-token")
    return MappingProxyType(
        {
            "GIGACHAT_ACCESS_TOKEN": "matrix-token",
            "WATSONX_TOKEN": "matrix-token",
            "WATSONX_PROJECT_ID": "matrix-project",
            "AWS_ACCESS_KEY_ID": "matrix-access-key",
            "AWS_SECRET_ACCESS_KEY": "matrix",
            "AWS_REGION_NAME": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "GITHUB_COPILOT_TOKEN_DIR": str(tokens),
            "ANTHROPIC_AWS_WORKSPACE_ID": "wrkspc_matrix",
            "HTTP_PROXY": elsewhere,
            "HTTPS_PROXY": elsewhere,
            "ALL_PROXY": elsewhere,
            "grpc_proxy": elsewhere,
            "NO_PROXY": "127.0.0.1,localhost",
            "AIOHTTP_TRUST_ENV": "True",
        }
    )


def isolate(patch: pytest.MonkeyPatch, rig: Rig) -> None:
    for name in tuple(os.environ):
        if AMBIENT.search(name):
            patch.delenv(name)
    for name, value in rig.environment.items():
        patch.setenv(name, value)
    patch.setattr(litellm, "suppress_debug_info", True)


async def _rerouting_client(peer: str) -> AsyncHTTPHandler:
    return AsyncHTTPHandler(transport=_Rerouted(peer))


@contextmanager
def matrix_rig(scratch: Path) -> Iterator[Rig]:
    with (
        wire_server(_peer) as provider,
        refusing_forward_proxy() as elsewhere,
        asyncio.Runner() as runner,
    ):
        rerouted: Final = runner.run(_rerouting_client(provider.url))
        try:
            yield Rig(
                runner=runner,
                provider=provider,
                elsewhere=elsewhere,
                rerouted=rerouted,
                credentials=_credentials(provider.url),
                environment=_environment(scratch, elsewhere.url),
            )
        finally:
            runner.run(rerouted.close())
            plain.cache_clear()
            baseline.cache_clear()
            _reference.cache_clear()


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"[:_REASON_LENGTH]


def _failure(error: Exception) -> Failure:
    return Failure(
        reason=_reason(error),
        refusal=isinstance(error, litellm.BadRequestError) or "drop_params" in str(error),
    )


async def _settle(call: Awaitable[object]) -> Failure | None:
    try:
        result: Final = await call
        if isinstance(result, AsyncIterable):
            async for _ in result:
                pass
    except Exception as error:
        return _failure(error)
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await GLOBAL_LOGGING_WORKER.stop()
    return None


async def _within_deadline(call: Awaitable[object]) -> Failure | None:
    async with asyncio.timeout(_DEADLINE_SECONDS):
        return await _settle(call)


def _picture(group: Group) -> bool:
    return group.provider == "fal_ai"


def _chat_content(group: Group) -> JsonValue:
    if not _picture(group):
        return _PROMPT
    return [
        {"type": "text", "text": _PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PIXEL}"}},
    ]


def _responses_input(group: Group) -> JsonValue:
    if not _picture(group):
        return _PROMPT
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": _PROMPT},
                {"type": "input_image", "image_url": f"data:image/png;base64,{_PIXEL}"},
            ],
        }
    ]


def _messages_content(group: Group) -> JsonValue:
    if not _picture(group):
        return _PROMPT
    return [
        {"type": "text", "text": _PROMPT},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PIXEL}},
    ]


def _required(group: Group, model: Model) -> Mapping[str, JsonValue]:
    if group.provider != "scaledown":
        return {}
    if model.name.endswith("extract"):
        return _extraction(group)
    return _SCALEDOWN_INPUTS.get(model.name, {})


def _extraction(group: Group) -> Mapping[str, JsonValue]:
    match group.endpoint:
        case Endpoint.CHAT:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "matrix", "schema": _EXTRACT, "strict": False},
                }
            }
        case Endpoint.RESPONSES:
            return {"text": {"format": {"type": "json_schema", "name": "matrix", "schema": _EXTRACT, "strict": False}}}
        case Endpoint.MESSAGES:
            return {"output_format": {"type": "json_schema", "schema": _EXTRACT}}
        case _:
            assert_never(group.endpoint)


def _reroutes(group: Group) -> bool:
    if group.provider == "microsoft_365_copilot":
        return True
    return group.provider == "github_copilot" and group.endpoint is Endpoint.MESSAGES and group.native


def _settings(rig: Rig, group: Group) -> Mapping[str, object]:
    return {
        "api_base": rig.provider.url,
        "api_key": "sk-matrix",
        **rig.credentials.get(group.provider, {}),
        **({"client": rig.rerouted} if _reroutes(group) else {}),
    }


def _call(rig: Rig, group: Group, model: Model, parameters: Mapping[str, JsonValue]) -> Awaitable[object]:
    settings: Final = _settings(rig, group)
    sent: Final = copy.deepcopy({**_required(group, model), **parameters})
    match group.endpoint:
        case Endpoint.CHAT:
            return litellm.acompletion(
                model=model.name,
                custom_llm_provider=model.provider,
                messages=[{"role": "user", "content": _chat_content(group)}],
                **settings,
                **sent,
            )
        case Endpoint.RESPONSES:
            return litellm.aresponses(
                model=model.name,
                custom_llm_provider=model.provider,
                input=_responses_input(group),
                **settings,
                **sent,
            )
        case Endpoint.MESSAGES:
            return litellm.anthropic_messages(
                model=model.name,
                custom_llm_provider=model.provider,
                messages=[{"role": "user", "content": _messages_content(group)}],
                **settings,
                **{"max_tokens": 64, **sent},
            )
        case _:
            assert_never(group.endpoint)


def exchange(rig: Rig, group: Group, model: Model, drop_params: bool, parameters: Mapping[str, JsonValue]) -> Exchange:
    rig.provider.drain()
    rig.elsewhere.drain()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(litellm, "drop_params", drop_params)
        failure: Final = rig.runner.run(_within_deadline(_call(rig, group, model, parameters)))
    return Exchange(
        requests=tuple(request for request in rig.provider.drain() if not request.target.startswith("/_oauth/")),
        strayed=frozenset(rig.elsewhere.targets()),
        failure=failure,
    )


def _body(index: int, body: bytes) -> Iterator[tuple[str, str]]:
    try:
        fields: Final = _BODY.validate_json(body)
    except ValidationError:
        yield (f"{index} body", body.decode("utf-8", "replace"))
        return
    for name, value in fields.items():
        yield (f"{index} body {name}", json.dumps(value, sort_keys=True))


def _fields(index: int, request: Request) -> Iterator[tuple[str, str]]:
    yield (f"{index} target", f"{request.method} {request.target}")
    for name, value in request.headers.items():
        if not _VOLATILE.search(name):
            yield (f"{index} header {name.lower()}", value)
    yield from _body(index, request.body)


def view(requests: tuple[Request, ...]) -> Mapping[str, str]:
    return MappingProxyType(
        dict(itertools.chain.from_iterable(_fields(index, request) for index, request in enumerate(requests)))
    )


def fixed_host(group: Group, model: Model) -> bool:
    return group.provider in _FIXED_HOST_PROVIDERS or model.key in _FIXED_HOST_MODELS


def _differing(first: Mapping[str, str], second: Mapping[str, str]) -> frozenset[str]:
    return frozenset(name for name in first.keys() | second.keys() if first.get(name) != second.get(name))


@lru_cache(maxsize=None)
def plain(rig: Rig, group: Group, model: Model, drop_params: bool) -> Exchange:
    return exchange(rig, group, model, drop_params, {})


def _attempts(rig: Rig, group: Group, drop_params: bool) -> Iterator[tuple[Model, Exchange]]:
    for model in group.models[:_CANDIDATES]:
        attempt: Final = plain(rig, group, model, drop_params)
        yield model, attempt
        if attempt.requests:
            return


@lru_cache(maxsize=None)
def baseline(rig: Rig, group: Group, drop_params: bool) -> Baseline | Unobservable | Unbuildable:
    attempts: Final = tuple(_attempts(rig, group, drop_params))
    model, last = attempts[-1]
    if last.requests:
        repeat: Final = exchange(rig, group, model, drop_params, {})
        seen: Final = view(last.requests)
        return Baseline(
            model=model,
            view=seen,
            unstable=_differing(seen, view(repeat.requests)),
            beside=last.strayed | repeat.strayed,
        )
    first_model, first = attempts[0]
    if any(attempt.strayed and fixed_host(group, tried) for tried, attempt in attempts):
        return Unobservable(model=first_model)
    return Unbuildable(model=first_model, reason=first.failure.reason if first.failure is not None else SILENT)


def _ways(sample: Sample) -> tuple[Sample, ...]:
    return (sample, *sample.otherwise)


@lru_cache(maxsize=None)
def _reference(rig: Rig, group: Group, drop_params: bool, attribute: str, way: int) -> Mapping[str, str] | None:
    found: Final = baseline(rig, group, drop_params)
    if not isinstance(found, Baseline):
        return None
    companions: Final = _ways(SAMPLES[group.endpoint][attribute])[way].companions
    if not companions:
        return found.view
    accompanied: Final = exchange(rig, group, found.model, drop_params, companions)
    return view(accompanied.requests) if accompanied.requests else None


def _echoed(
    rig: Rig, cell: Cell, found: Baseline, parameters: Mapping[str, JsonValue], seen: Mapping[str, str]
) -> bool:
    repeat: Final = view(exchange(rig, cell.group, found.model, cell.drop_params, parameters).requests)
    return any(name in seen and seen[name] == repeat.get(name) for name in found.unstable)


def _observe(rig: Rig, cell: Cell, found: Baseline, way: int, value: JsonValue) -> Outcome:
    sample: Final = _ways(SAMPLES[cell.group.endpoint][cell.attribute])[way]
    parameters: Final = {**sample.companions, cell.attribute: value}
    sent: Final = exchange(rig, cell.group, found.model, cell.drop_params, parameters)
    if not sent.requests:
        elsewhere: Final = sent.strayed - found.beside
        if elsewhere:
            return Broken(reason=_ELSEWHERE.format(hosts=", ".join(sorted(elsewhere))))
        if sent.failure is None:
            return Broken(reason=SILENT)
        return Refused(reason=sent.failure.reason) if sent.failure.refusal else Broken(reason=sent.failure.reason)
    reference: Final = _reference(rig, cell.group, cell.drop_params, cell.attribute, way)
    if reference is None:
        return Forwarded()
    seen: Final = view(sent.requests)
    if _differing(seen, reference) - found.unstable:
        return Forwarded()
    if found.unstable and _echoed(rig, cell, found, parameters, seen):
        return Forwarded()
    return Dropped()


def _alternatives(rig: Rig, cell: Cell, found: Baseline, sample: Sample) -> Iterator[Outcome]:
    for value in sample.alternatives:
        yield _observe(rig, cell, found, 0, value)
    for way, other in enumerate(sample.otherwise, start=1):
        yield _observe(rig, cell, found, way, other.value)


def verdict(rig: Rig, cell: Cell) -> Outcome:
    sample: Final = SAMPLES[cell.group.endpoint].get(cell.attribute)
    if sample is None:
        return Unsampled()
    found: Final = baseline(rig, cell.group, cell.drop_params)
    if not isinstance(found, Baseline):
        return found
    if (
        cell.group.endpoint is Endpoint.CHAT
        and (cell.group.provider, cell.attribute) in _SENT_ONLY_TO_THE_PROVIDERS_OWN_HOST
    ):
        return Unobservable(model=found.model)
    first: Final = _observe(rig, cell, found, 0, sample.value)
    expected_drop: Final = cell.drop_params and cell.attribute not in cell.group.supported
    if isinstance(first, Forwarded) or (expected_drop and isinstance(first, Dropped)):
        return first
    if any(isinstance(outcome, Forwarded) for outcome in _alternatives(rig, cell, found, sample)):
        return Forwarded()
    return first


def _claim(cell: Cell) -> str:
    where: Final = f"{cell.group.provider} on {cell.group.endpoint.value}"
    if cell.group.native:
        return f"LiteLLM lists {cell.attribute} as supported for {where}"
    return (
        f"LiteLLM bridges {where} to chat completions, where it lists the parameter {cell.attribute} becomes as "
        f"supported for {cell.group.provider}"
    )


def _consumed_by_litellm(cell: Cell) -> bool:
    group: Final = cell.group
    if cell.attribute in PROVIDER_UNVALIDATED_PARAMS or chat_name(group, cell.attribute) in _HANDLED_BY_LITELLM:
        return True
    return not group.native and group.endpoint is Endpoint.RESPONSES and cell.attribute in BRIDGE_SESSION_STATE


def _dropped(cell: Cell) -> Gap | None:
    attribute: Final = cell.attribute
    if _consumed_by_litellm(cell):
        return None
    if attribute in cell.group.supported:
        setting: Final = "on" if cell.drop_params else "off"
        return Gap(
            GapKind.LISTED_NOT_SENT,
            f"{_claim(cell)}, but with drop_params {setting} the request it sends the provider is the same with and "
            f"without {attribute}",
        )
    if cell.drop_params:
        return None
    return Gap(
        GapKind.UNLISTED_VANISHES,
        f"LiteLLM does not list {attribute} as supported for {cell.group.provider} on {cell.group.endpoint.value}, "
        f"and with drop_params off it neither raised its unsupported-parameter error nor sent {attribute} to the "
        "provider",
    )


def _refused(cell: Cell, reason: str) -> Gap | None:
    attribute: Final = cell.attribute
    if attribute in _PROMPT_CONTENT:
        return None
    where: Final = f"{cell.group.provider} on {cell.group.endpoint.value}"
    if cell.drop_params:
        return Gap(
            GapKind.REFUSED_WITH_DROP,
            f"With drop_params on, LiteLLM refused {attribute} for {where} instead of dropping it: {reason}",
        )
    if attribute in cell.group.supported:
        return Gap(
            GapKind.LISTED_REFUSED,
            f"{_claim(cell)}, but with drop_params off no sample value of {attribute} reaches the provider: {reason}",
        )
    return None


def gap(cell: Cell, outcome: Outcome) -> Gap | None:
    setting: Final = "on" if cell.drop_params else "off"
    match outcome:
        case Forwarded() | Unobservable() | Unbuildable():
            return None
        case Dropped():
            return _dropped(cell)
        case Refused(reason=reason):
            return _refused(cell, reason)
        case Broken(reason=reason):
            return Gap(
                GapKind.BROKEN,
                f"With drop_params {setting}, LiteLLM failed before sending {cell.group.provider} on "
                f"{cell.group.endpoint.value} a request carrying {cell.attribute}: {reason}",
            )
        case Unsampled():
            return Gap(
                GapKind.UNSAMPLED,
                f"No sample value for {cell.attribute} on {cell.group.endpoint.value}, add one to SAMPLES",
            )
        case _:
            assert_never(outcome)
