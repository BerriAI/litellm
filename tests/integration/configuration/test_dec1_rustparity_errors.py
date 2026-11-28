"""Error parity: what a caller sees when something fails, Python only versus Rust first.

Contract: every refusal is decided by Python in both modes (gateway-forward/AGENTS.md: "Python still owns every
route and policy"), so for the same config and the same scripted provider the caller must get the same status, the
same error envelope (OpenAI shape on OpenAI routes, Anthropic shape on `/v1/messages`), the same `retry-after` and
`x-litellm-*` headers, and the same provider traffic (none, for refusals) in both modes. Covered: provider 4xx and
5xx on every endpoint and provider in the matrix, provider 429 with `retry-after`, missing, wrong and blocked keys,
a key without access to the model, unknown model, malformed and non-object JSON, an empty body, bodies over
`max_request_size_mb` (with and without `Content-Length`), gzip request bodies (decoded, over the decoded limit, the
`x-gzip` alias and a mislabelled plain body), budget refusal and rate limit refusal.

The rig sets `general_settings.max_request_size_mb: 1`, which the proxy enforces only with an enterprise license
(`premium_user`, proxy_server.py:23201); CI exports `LITELLM_LICENSE` to the integration jobs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration.configuration.dec1_rustparity_support import (
    ANTHROPIC,
    AZURE,
    CHAT,
    EMBEDDINGS,
    GENERATION_CELLS,
    MASTER_KEY,
    MESSAGES,
    OPENAI,
    OPENAI_EMBED,
    PROVIDER_RETRY_AFTER,
    RESPONSES,
    UNARY_CELLS,
    Booted,
    Cell,
    Pair,
    Rig,
    Seen,
    chat_bytes,
    comparable_error,
    directive,
    gzipped,
    litellm_header_names,
    marker,
    parity_rig,
    seen_of,
)

pytestmark: Final = pytest.mark.timeout(900)

ROUTES: Final = pytest.mark.parametrize(
    "cell",
    [Cell(CHAT, OPENAI), Cell(MESSAGES, ANTHROPIC), Cell(RESPONSES, OPENAI), Cell(EMBEDDINGS, OPENAI_EMBED)],
    ids=["chat", "messages", "responses", "embeddings"],
)
TWO_MIB: Final = 2 * 1024 * 1024
OPENAI_TOO_LARGE: Final = {"error": "Request size is too large. Max size is 1 MB"}
ANTHROPIC_TOO_LARGE: Final = {
    "type": "error",
    "error": {"type": "request_too_large", "message": "Request size is too large. Max size is 1 MB"},
}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with parity_rig(
        tmp_path_factory.mktemp("dec1-rustparity-errors"), general_settings={"max_request_size_mb": 1}
    ) as booted:
        yield booted


def _same_refusal(seen: Pair[Seen]) -> None:
    assert seen.rust.status == seen.python.status, (seen.python.body[:400], seen.rust.body[:400])
    assert seen.rust.headers.get("content-type") == seen.python.headers.get("content-type")
    assert comparable_error(seen.rust) == comparable_error(seen.python)


# ------------------------------------------------------------------------------------------- provider errors


@pytest.mark.parametrize("cell", list(GENERATION_CELLS.values()), ids=list(GENERATION_CELLS))
def test_a_provider_400_is_relayed_with_the_same_status_and_envelope_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The provider answers 400 in its own error shape. Both proxies must answer with the same status and the same
    JSON error body (OpenAI envelope on chat and responses, Anthropic envelope on messages), after the marker and
    call id are normalised. Provider exception mapping is Python code in both modes."""
    _same_refusal(rig.send(cell.path, directive(cell.body(), "fail=400")))


@pytest.mark.parametrize("cell", list(UNARY_CELLS.values()), ids=list(UNARY_CELLS))
def test_a_provider_500_is_relayed_with_the_same_status_and_envelope_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The provider answers 500 on every endpoint and provider of the matrix, embeddings included. Both proxies
    must give the same status and the same error body; `router_settings.num_retries: 0` makes this one provider
    call per proxy."""
    _same_refusal(rig.send(cell.path, directive(cell.body(), "fail=500")))


@pytest.mark.parametrize(
    ("cell", "status"),
    [
        (Cell(CHAT, OPENAI), 401),
        (Cell(CHAT, OPENAI), 404),
        (Cell(CHAT, OPENAI), 503),
        (Cell(MESSAGES, ANTHROPIC), 401),
        (Cell(MESSAGES, ANTHROPIC), 529),
        (Cell(CHAT, AZURE), 503),
        (Cell(EMBEDDINGS, OPENAI_EMBED), 401),
    ],
    ids=["chat-401", "chat-404", "chat-503", "messages-401", "messages-529", "azure-503", "embeddings-401"],
)
def test_other_provider_statuses_are_relayed_the_same_in_both_modes(rig: Rig, cell: Cell, status: int) -> None:
    """The provider answers 401, 404, 503 or Anthropic's 529. Both proxies must give the same status and body: a
    provider credential failure in particular must stay distinguishable from the caller's own 401 in both modes."""
    _same_refusal(rig.send(cell.path, directive(cell.body(), f"fail={status}")))


@pytest.mark.parametrize(
    "cell",
    [Cell(CHAT, OPENAI), Cell(MESSAGES, ANTHROPIC), Cell(CHAT, AZURE), Cell(RESPONSES, OPENAI)],
    ids=["chat-openai", "messages-anthropic", "chat-azure", "responses-openai"],
)
def test_a_provider_429_reaches_the_caller_with_the_same_retry_after_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The provider answers 429 with `retry-after: 7`. Both proxies must answer the same status, and the
    `retry-after` header the caller sees must be identical in both modes (present with the same value, or absent in
    both). SDK backoff reads this header, and `retry-after` is end-to-end, so the hop must not drop it."""
    seen: Final = rig.send(cell.path, directive(cell.body(), "fail=429"))
    assert seen.rust.status == seen.python.status == 429, (seen.python.body[:300], seen.rust.body[:300])
    assert seen.rust.headers.get("retry-after") == seen.python.headers.get("retry-after")
    assert seen.python.headers.get("retry-after") in (PROVIDER_RETRY_AFTER, None)


def test_a_provider_error_sets_the_same_x_litellm_header_names_in_both_modes(rig: Rig) -> None:
    """The provider answers 500 on a chat call. The set of `x-litellm-*` header names on the error answer must be
    equal in both modes (the error path adds `x-litellm-call-id` among others)."""
    seen: Final = rig.send(CHAT, directive(Cell(CHAT, OPENAI).body(), "fail=500"))
    assert litellm_header_names(seen.rust) == litellm_header_names(seen.python)


# ------------------------------------------------------------------------------------------- auth


@ROUTES
def test_a_call_without_a_key_is_refused_the_same_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The cell's request with no credentials at all. Both proxies must answer 401 with the same body (OpenAI
    `auth_error` envelope, Anthropic `authentication_error` envelope on `/v1/messages`) and the provider must not be
    called by either."""
    seen: Final = rig.send(cell.path, cell.body(), key=None)
    assert (seen.python.status, seen.rust.status) == (401, 401)
    _same_refusal(seen)
    assert rig.provider_requests(seen) == Pair((), ())


@ROUTES
def test_a_wrong_key_is_refused_the_same_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The cell's request with `Authorization: Bearer sk-dec1-rustparity-wrong`. Both proxies must answer 401 with
    the same body, and neither body may contain the presented key."""
    seen: Final = rig.send(cell.path, cell.body(), key="sk-dec1-rustparity-wrong")
    assert (seen.python.status, seen.rust.status) == (401, 401)
    _same_refusal(seen)
    assert b"sk-dec1-rustparity-wrong" not in seen.python.body + seen.rust.body


def test_an_auth_failure_on_messages_uses_the_anthropic_envelope_in_both_modes(rig: Rig) -> None:
    """`/v1/messages` with no key. Both bodies must have `type: "error"` and `error.type: "authentication_error"`
    (ANTHROPIC_ERROR_TYPE_MAP 401, anthropic_error_payload.py `route_error_content`): the envelope is chosen from the
    request path, which the forwarder passes unchanged."""
    seen: Final = rig.send(MESSAGES, Cell(MESSAGES, ANTHROPIC).body(), key=None)
    shapes: Final = tuple(
        (observed.json().get("type"), dict(observed.json().get("error") or {}).get("type"))  # type: ignore[arg-type]
        for observed in (seen.python, seen.rust)
    )
    assert shapes == (("error", "authentication_error"), ("error", "authentication_error"))


def test_a_key_without_access_to_the_model_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A key created on both proxies with `models: ["rp-openai"]` calls `rp-anthropic`. Both must refuse with the
    same status and body, error type `key_model_access_denied`, and neither may call the provider."""
    key: Final = rig.same_key(models=[OPENAI])
    seen: Final = rig.send(CHAT, Cell(CHAT, ANTHROPIC).body(), key=key)
    _same_refusal(seen)
    assert dict(seen.python.json()["error"]).get("type") == "key_model_access_denied"  # type: ignore[arg-type]
    assert rig.provider_requests(seen) == Pair((), ())


def test_a_blocked_key_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A key is created on both proxies and then blocked with `POST /key/block`. A chat call with it must be refused
    by both with the same status and body, and the provider must not be called."""
    key: Final = rig.same_key()
    for proxy in (rig.python, rig.rust):
        proxy.gateway.post("/key/block", {"key": key})
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), key=key)
    assert seen.python.status in (401, 403), seen.python.body[:300]
    _same_refusal(seen)


# ------------------------------------------------------------------------------------------- request problems


@ROUTES
def test_an_unknown_model_is_refused_the_same_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The cell's request names `rp-no-such-model`. Both proxies must answer 400 with the same body, which names
    the route and says `Invalid model name passed in model=rp-no-such-model` (route_llm_request.py:239)."""
    seen: Final = rig.send(cell.path, lambda case: {**cell.body()(case), "model": "rp-no-such-model"})
    assert (seen.python.status, seen.rust.status) == (400, 400), (seen.python.body[:300], seen.rust.body[:300])
    assert b"Invalid model name passed in model=rp-no-such-model" in seen.python.body
    _same_refusal(seen)


@pytest.mark.parametrize("path", [CHAT, MESSAGES, RESPONSES, EMBEDDINGS])
def test_malformed_json_is_refused_the_same_in_both_modes(rig: Rig, path: str) -> None:
    """The master key posts `{"model": "rp-openai", "messages": [` (cut off) to the route. Both proxies must answer
    400 with the same body; on chat it is the OpenAI envelope with `param: "request_body"` and a message starting
    `Invalid JSON payload: ` (http_parsing_utils.py)."""
    seen: Final = rig.send_raw(
        path,
        b'{"model": "rp-openai", "messages": [',
        {"Authorization": f"Bearer {MASTER_KEY}", "content-type": "application/json"},
    )
    assert (seen.python.status, seen.rust.status) == (400, 400)
    _same_refusal(seen)


@pytest.mark.parametrize("data", [b"[]", b'"text"', b"42", b"null"], ids=["array", "string", "number", "null"])
def test_json_that_is_not_an_object_is_refused_the_same_in_both_modes(rig: Rig, data: bytes) -> None:
    """Valid JSON that is not an object is posted to chat. Both proxies must answer 400 with the same body,
    `Invalid JSON payload: expected an object` in the OpenAI envelope."""
    seen: Final = rig.send_raw(
        CHAT, data, {"Authorization": f"Bearer {MASTER_KEY}", "content-type": "application/json"}
    )
    assert (seen.python.status, seen.rust.status) == (400, 400)
    _same_refusal(seen)


def test_an_empty_body_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A chat POST with `Content-Length: 0`. Both proxies must give the same status and body and call no provider:
    an empty body has a known length of zero, so the forwarder sends no `transfer-encoding: chunked` for it."""
    _same_refusal(
        rig.send_raw(CHAT, b"", {"Authorization": f"Bearer {MASTER_KEY}", "content-type": "application/json"})
    )


@pytest.mark.parametrize(
    ("path", "expected"), [(CHAT, OPENAI_TOO_LARGE), (MESSAGES, ANTHROPIC_TOO_LARGE)], ids=["chat", "messages"]
)
def test_a_body_over_max_request_size_mb_is_refused_with_413_in_both_modes(
    rig: Rig, path: str, expected: Mapping[str, JsonValue]
) -> None:
    """`max_request_size_mb: 1`, and the caller posts a 2 MiB JSON body with its `Content-Length`. Both proxies must
    answer 413 with exactly `expected` (request_size_limit_middleware.py `send_request_too_large`, the Anthropic
    envelope on `/v1/messages`) and call no provider. Python rejects from the header alone, before reading the body,
    while the forwarder is still streaming it (connector.rs keeps reading Python's answer)."""
    body: Final = chat_bytes(OPENAI, "x" * TWO_MIB)
    seen: Final = rig.send_raw(
        path, body, {"Authorization": f"Bearer {MASTER_KEY}", "content-type": "application/json"}
    )
    assert (seen.python.status, seen.rust.status) == (413, 413)
    assert (seen.python.json(), seen.rust.json()) == (dict(expected), dict(expected))


def _chunked_upload(proxy: Booted) -> tuple[int, bytes]:
    body: Final = chat_bytes(OPENAI, "x" * TWO_MIB)
    pieces: Final = (body[start : start + 65536] for start in range(0, len(body), 65536))
    response: Final = proxy.gateway.client.post(
        CHAT, content=pieces, headers={"Authorization": f"Bearer {MASTER_KEY}", "content-type": "application/json"}
    )
    return response.status_code, response.content


def test_a_chunked_body_over_max_request_size_mb_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """The same 2 MiB chat body sent with `Transfer-Encoding: chunked` and no length. Both proxies must answer 413
    with `{"error": "Request size is too large. Max size is 1 MB"}`: Python counts bytes as they arrive
    (`RequestSizeLimitMiddleware.limited_receive`), and the forwarder must stream them, never cap or buffer them
    itself."""
    answers: Final = rig.both(_chunked_upload)
    assert answers.python[0] == answers.rust[0] == 413, answers
    assert answers.rust[1] == answers.python[1]


# ------------------------------------------------------------------------------------------- compressed bodies


def _compressed(rig: Rig, coding: str, data: Callable[[str], bytes]) -> Pair[Seen]:
    def one(proxy: Booted) -> Seen:
        case: Final = marker()
        response: Final = proxy.gateway.client.post(
            CHAT,
            content=data(case),
            headers={
                "Authorization": f"Bearer {MASTER_KEY}",
                "content-type": "application/json",
                "content-encoding": coding,
            },
        )
        return seen_of(response, case=case)

    return rig.both(one)


def test_a_gzip_request_body_is_decoded_the_same_in_both_modes(rig: Rig) -> None:
    """A chat body compressed with gzip and sent with `Content-Encoding: gzip`. Both proxies must answer 200, and
    the provider must receive plain JSON with the caller's prompt and no `content-encoding` in both modes:
    `content-encoding` is a `content-*` header the forwarder always passes (rewrite.rs `must_forward`), and
    decoding is Python's `RequestBodyDecompressionMiddleware`."""
    seen: Final = _compressed(rig, "gzip", lambda case: gzipped(chat_bytes(OPENAI, case)))
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:300], seen.rust.body[:300])
    sent: Final = rig.provider_requests(seen)
    assert (len(sent.python), len(sent.rust)) == (1, 1)
    assert ("content-encoding" in sent.python[0].headers, "content-encoding" in sent.rust[0].headers) == (False, False)


def test_the_x_gzip_alias_is_decoded_the_same_in_both_modes(rig: Rig) -> None:
    """The same gzip body labelled `Content-Encoding: x-gzip`, the legacy alias (`CODING_ALIASES`). Both proxies
    must answer 200 and each provider must receive exactly one call."""
    seen: Final = _compressed(rig, "x-gzip", lambda case: gzipped(chat_bytes(OPENAI, case)))
    assert (seen.python.status, seen.rust.status) == (200, 200)
    assert tuple(len(calls) for calls in (rig.provider_requests(seen).python, rig.provider_requests(seen).rust)) == (
        1,
        1,
    )


def test_a_plain_body_labelled_gzip_is_served_the_same_in_both_modes(rig: Rig) -> None:
    """A plain JSON chat body sent with `Content-Encoding: gzip`. It does not start with the gzip magic, so Python
    passes it on as-is without the header (`_decode_if_encoded`), and both proxies must answer 200."""
    seen: Final = _compressed(rig, "gzip", lambda case: chat_bytes(OPENAI, case))
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:300], seen.rust.body[:300])


def test_a_gzip_body_that_decodes_over_the_limit_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A small gzip body that decodes to more than 2 MiB, under `max_request_size_mb: 1`. Both proxies must answer
    413 with the same body and call no provider: the decoded size limit is the operator's limit
    (`_effective_max_mb`), enforced in Python whatever is in front of it."""
    seen: Final = _compressed(rig, "gzip", lambda case: gzipped(chat_bytes(OPENAI, case + " " + "x" * TWO_MIB)))
    assert (seen.python.status, seen.rust.status) == (413, 413), (seen.python.body[:300], seen.rust.body[:300])
    _same_refusal(seen)
    assert rig.provider_requests(seen) == Pair((), ())


def test_a_truncated_gzip_body_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A gzip body cut in half (valid magic, broken stream). Both proxies must give the same 4xx status and body.
    UNVERIFIED: the exact status, 400 is expected from `_decode` failing (request_body_decompression_middleware.py)."""
    seen: Final = _compressed(rig, "gzip", lambda case: gzipped(chat_bytes(OPENAI, case))[:40])
    assert 400 <= seen.python.status < 500, seen.python.body[:300]
    _same_refusal(seen)


# ------------------------------------------------------------------------------------------- budgets and limits


def _refused_after_spend(proxy: Booted, key: str) -> httpx.Response:
    def call() -> httpx.Response:
        return proxy.gateway.client.post(
            CHAT,
            json={"model": OPENAI, "messages": [{"role": "user", "content": marker()}]},
            headers={"Authorization": f"Bearer {key}"},
        )

    return eventually(call, lambda response: response.status_code != 200, seconds=90)


def test_a_key_over_its_budget_is_refused_the_same_in_both_modes(rig: Rig) -> None:
    """A key with `max_budget: 0.000001` is created on both proxies with the same raw value; its first chat call
    costs 0.011. Repeating the call must, once spend lands, be refused by both proxies with status 422
    (`litellm.budget_exceeded_status_code`, litellm/__init__.py:393) and error type `budget_exceeded`."""
    key: Final = rig.same_key(max_budget=0.000001)
    refused: Final = rig.both(lambda proxy: _refused_after_spend(proxy, key))
    assert (refused.python.status_code, refused.rust.status_code) == (422, 422), (
        refused.python.text,
        refused.rust.text,
    )
    types: Final = tuple(response.json()["error"]["type"] for response in (refused.python, refused.rust))
    assert types == ("budget_exceeded", "budget_exceeded")


def _second_call(proxy: Booted, key: str) -> httpx.Response:
    for _ in range(2):
        response = proxy.gateway.client.post(  # rebind-ok: the second answer is the one kept
            CHAT,
            json={"model": OPENAI, "messages": [{"role": "user", "content": marker()}]},
            headers={"Authorization": f"Bearer {key}"},
        )
    return response


def test_a_key_over_its_rpm_limit_is_refused_with_the_same_headers_in_both_modes(rig: Rig) -> None:
    """A key with `rpm_limit: 1` is created on both proxies and makes two chat calls. The second must be a 429 in
    both modes carrying `retry-after: 60`, `rate_limited_by: api_key` and `rate_limit_type: requests`, with equal
    error bodies (the message names the hashed key, which is equal because the raw key is)."""
    key: Final = rig.same_key(rpm_limit=1)
    refused: Final = rig.both(lambda proxy: _second_call(proxy, key))
    picked: Final = tuple(
        (
            response.status_code,
            response.headers.get("retry-after"),
            response.headers.get("rate_limited_by"),
            response.headers.get("rate_limit_type"),
        )
        for response in (refused.python, refused.rust)
    )
    assert picked == ((429, "60", "api_key", "requests"),) * 2
    assert comparable_error(seen_of(refused.rust)) == comparable_error(seen_of(refused.python))
