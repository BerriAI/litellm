"""Which fallback list a big provider's error takes: `fallbacks`, `context_window_fallbacks` or `content_policy_fallbacks`.

The router core chooses the list by the exception class litellm's provider mapping raised
(`fallback_targets` in `litellm/rust/crates/router-core/src/call.rs:747`). A `ContextWindowExceededError`
takes the group's `context_window_fallbacks` entry and is not retried first (`retry.rs:77`). A
`ContentPolicyViolationError` takes the `content_policy_fallbacks` entry. Every other failure takes the
regular `fallbacks`. With `fallback_on_unsupported_params: false` a provider's plain 400 (a
`BadRequestError` that names its provider) is returned without any fallback (`call.rs:253`), while context
window and content policy errors still fall back.

Provider mapping decides the class: Bedrock `Input is too long`, Anthropic `prompt is too long`, Google
`exceeds the maximum number of tokens allowed` and OpenAI `context_length_exceeded` are context window
errors. Azure's `content_filter`, Anthropic's `content filtering policy` and OpenAI's safety system text are
content policy errors, but Bedrock has no content policy mapping, so a Bedrock refusal is a plain 400. A
Vertex 403 is answered as a provider 400 and so is not fallen back with the flag off, while a Bedrock 403 is
a `PermissionDeniedError` and is.

One owned proxy runs with `num_retries: 0`, `disable_cooldowns: true`, `fallback_on_unsupported_params:
false` and, for every primary group, all three lists pointing at three different healthy OpenAI groups, so
the group that answered shows which list was used.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_reliab_support import (
    ANTHROPIC_FILTER_TEXT,
    ANTHROPIC_RATE_TEXT,
    ANTHROPIC_TOO_LONG_TEXT,
    BEDROCK_ACCESS_TEXT,
    BEDROCK_THROTTLE_TEXT,
    BEDROCK_TOO_LONG_TEXT,
    BEDROCK_VALIDATION_TEXT,
    BUILDERS,
    GOOGLE_EXHAUSTED_TEXT,
    GOOGLE_INVALID_TEXT,
    GOOGLE_OVERLOADED_TEXT,
    GOOGLE_PERMISSION_TEXT,
    GOOGLE_TOO_LONG_TEXT,
    OPENAI_POLICY_TEXT,
    OPENAI_TOO_LONG_TEXT,
    Rig,
    anthropic_error,
    anthropic_overloaded,
    azure_content_filter,
    azure_rate_limit,
    bedrock_error,
    chat,
    content_of,
    error_message,
    google_error,
    openai_deployment,
    openai_error,
    reliab_rig,
)
from pydantic import JsonValue

from tests.integration._support.wire import Reply

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

REGULAR: Final = "rescue-regular"
CONTEXT: Final = "rescue-context"
POLICY: Final = "rescue-policy"

# primary group -> (provider, its fixed reply)
PRIMARY: Final[dict[str, tuple[str, Reply]]] = {
    "fb-cw-bedrock": ("bedrock", bedrock_error(400, "ValidationException", BEDROCK_TOO_LONG_TEXT)),
    "fb-cw-anthropic": ("anthropic", anthropic_error(400, "invalid_request_error", ANTHROPIC_TOO_LONG_TEXT)),
    "fb-cw-gemini": ("gemini", google_error(400, "INVALID_ARGUMENT", GOOGLE_TOO_LONG_TEXT)),
    "fb-cw-openai": ("openai", openai_error(400, OPENAI_TOO_LONG_TEXT, code="context_length_exceeded")),
    "fb-cp-azure": ("azure", azure_content_filter()),
    "fb-cp-anthropic": ("anthropic", anthropic_error(400, "invalid_request_error", ANTHROPIC_FILTER_TEXT)),
    "fb-cp-openai": ("openai", openai_error(400, OPENAI_POLICY_TEXT, code="content_policy_violation")),
    "fb-rl-bedrock": ("bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT)),
    "fb-rl-vertex": ("vertex", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT)),
    "fb-rl-azure": ("azure", azure_rate_limit()),
    "fb-rl-anthropic": ("anthropic", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT)),
    "fb-5xx-anthropic": ("anthropic", anthropic_overloaded()),
    "fb-5xx-gemini": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT)),
    "fb-400-bedrock-refusal": (
        "bedrock",
        bedrock_error(400, "ValidationException", "The request was blocked: content policy refusal"),
    ),
    "fb-400-bedrock-validation": ("bedrock", bedrock_error(400, "ValidationException", BEDROCK_VALIDATION_TEXT)),
    "fb-400-vertex-invalid": ("vertex", google_error(400, "INVALID_ARGUMENT", GOOGLE_INVALID_TEXT)),
    "fb-403-vertex": ("vertex", google_error(403, "PERMISSION_DENIED", GOOGLE_PERMISSION_TEXT)),
    "fb-403-bedrock": ("bedrock", bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT)),
}
# A Bedrock group whose only fallback is an Anthropic group that is overloaded too.
CHAIN_PRIMARY: Final = "fb-chain-bedrock"
CHAIN_TARGET: Final = "fb-chain-anthropic"


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    entries: Final[list[dict[str, JsonValue]]] = [
        openai_deployment(REGULAR, REGULAR, base),
        openai_deployment(CONTEXT, CONTEXT, base),
        openai_deployment(POLICY, POLICY, base),
        BUILDERS["bedrock"](CHAIN_PRIMARY, CHAIN_PRIMARY, base),
        BUILDERS["anthropic"](CHAIN_TARGET, CHAIN_TARGET, base),
    ]  # mutable-ok: built once for the config
    entries.extend(BUILDERS[provider](group, group, base) for group, (provider, _) in PRIMARY.items())
    return entries


ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "fallback_on_unsupported_params": False,
    "fallbacks": [*({group: [REGULAR]} for group in PRIMARY), {CHAIN_PRIMARY: [CHAIN_TARGET]}],
    "context_window_fallbacks": [{group: [CONTEXT]} for group in PRIMARY],
    "content_policy_fallbacks": [{group: [POLICY]} for group in PRIMARY],
}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-fallbacks"), _models, ROUTER_SETTINGS) as value:
        for group, (_, reply) in PRIMARY.items():
            value.peer.always(group, reply)
        value.peer.always(CHAIN_PRIMARY, bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT))
        value.peer.always(CHAIN_TARGET, anthropic_overloaded())
        yield value


@pytest.mark.parametrize("group", ["fb-cw-bedrock", "fb-cw-anthropic", "fb-cw-gemini", "fb-cw-openai"])
def test_a_providers_context_overflow_takes_the_context_window_fallback(rig: Rig, group: str) -> None:
    """The primary group answers its provider's context overflow and has all three lists configured.

    The caller must get HTTP 200 with `served rescue-context`, `x-litellm-attempted-fallbacks: 1` and
    `x-litellm-model-id: dec1-reliab-rescue-context`. The primary was called once (a context window error is
    not retried when a context window list exists) and the regular rescue group was never called for it.
    """
    before_regular: Final = rig.peer.count(REGULAR)
    response: Final = chat(rig.gateway, group, num_retries=2)
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {CONTEXT}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    assert response.headers["x-litellm-model-id"] == f"dec1-reliab-{CONTEXT}"
    assert rig.peer.count(group) == 1
    assert rig.peer.count(REGULAR) == before_regular


@pytest.mark.parametrize("group", ["fb-cp-azure", "fb-cp-anthropic", "fb-cp-openai"])
def test_a_providers_content_refusal_takes_the_content_policy_fallback(rig: Rig, group: str) -> None:
    """The primary group refuses with its provider's content policy error (Azure `content_filter`, Anthropic
    `content filtering policy`, OpenAI safety system text).

    The caller must get HTTP 200 with `served rescue-policy` and `x-litellm-attempted-fallbacks: 1`, and the
    primary was called exactly once even with `num_retries: 2` on the request.
    """
    response: Final = chat(rig.gateway, group, num_retries=2)
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {POLICY}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    assert rig.peer.count(group) == 1


def test_a_bedrock_refusal_is_not_a_content_policy_error_and_is_returned_as_a_400(rig: Rig) -> None:
    """The Bedrock primary answers 400 `ValidationException` whose text speaks of a content policy refusal.

    `_map_bedrock_exception` has no content policy branch, so it is a provider 400 `BadRequestError`. With
    `fallback_on_unsupported_params: false` it is returned as HTTP 400 carrying the AWS text: the content
    policy rescue group is not called, and neither is the regular one.
    """
    policy_before: Final = rig.peer.count(POLICY)
    regular_before: Final = rig.peer.count(REGULAR)
    response: Final = chat(rig.gateway, "fb-400-bedrock-refusal")
    assert response.status_code == 400, response.text
    assert "content policy refusal" in error_message(response)
    assert rig.peer.count(POLICY) == policy_before
    assert rig.peer.count(REGULAR) == regular_before


@pytest.mark.parametrize("group", ["fb-rl-bedrock", "fb-rl-vertex", "fb-rl-azure", "fb-rl-anthropic"])
def test_a_providers_throttle_takes_the_regular_fallback(rig: Rig, group: str) -> None:
    """The primary group answers its provider's throttle (Bedrock `ThrottlingException`, Vertex
    `RESOURCE_EXHAUSTED`, Azure 429, Anthropic `rate_limit_error`).

    The caller must get HTTP 200 with `served rescue-regular`, `x-litellm-attempted-fallbacks: 1` and the
    rescue deployment's id in `x-litellm-model-id`.
    """
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {REGULAR}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    assert response.headers["x-litellm-model-id"] == f"dec1-reliab-{REGULAR}"


@pytest.mark.parametrize("group", ["fb-5xx-anthropic", "fb-5xx-gemini"])
def test_a_providers_overload_takes_the_regular_fallback(rig: Rig, group: str) -> None:
    """The primary answers Anthropic 529 `overloaded_error` or Google's 503 overload text.

    Both are `InternalServerError`s and take the regular list: HTTP 200 with `served rescue-regular` and
    one attempted fallback.
    """
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {REGULAR}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


@pytest.mark.parametrize("group", ["fb-400-bedrock-validation", "fb-400-vertex-invalid"])
def test_a_provider_400_is_returned_without_a_fallback_when_unsupported_params_fallback_is_off(
    rig: Rig, group: str
) -> None:
    """The primary answers Bedrock `ValidationException` or Google `INVALID_ARGUMENT`.

    With `fallback_on_unsupported_params: false` a provider's plain 400 has no fallback: the caller gets
    HTTP 400 with the provider's text and no `x-litellm-attempted-fallbacks` header.
    """
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 400, response.text
    assert "x-litellm-attempted-fallbacks" not in response.headers


def test_a_vertex_403_is_returned_without_a_fallback_because_it_is_answered_as_a_400(rig: Rig) -> None:
    """The Vertex primary answers 403 `PERMISSION_DENIED`.

    Vertex's 403 is mapped to a provider `BadRequestError`, so with `fallback_on_unsupported_params: false`
    the caller gets HTTP 400 and the regular rescue group is not called. A credentials outage on Vertex
    therefore does not fail over, which a Bedrock 403 does.
    """
    regular_before: Final = rig.peer.count(REGULAR)
    response: Final = chat(rig.gateway, "fb-403-vertex")
    assert response.status_code == 400, response.text
    assert rig.peer.count(REGULAR) == regular_before


def test_a_bedrock_403_access_denied_takes_the_regular_fallback(rig: Rig) -> None:
    """The Bedrock primary answers 403 `AccessDeniedException`.

    A `PermissionDeniedError` is not a bad request, so it falls back even with the flag off: HTTP 200 with
    `served rescue-regular` and one attempted fallback.
    """
    response: Final = chat(rig.gateway, "fb-403-bedrock")
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {REGULAR}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_when_the_fallback_provider_also_fails_the_caller_gets_the_primary_providers_error(rig: Rig) -> None:
    """The Bedrock primary throttles and its only fallback, an Anthropic group, answers 529 overloaded.

    The caller must get the primary's error, HTTP 429 carrying Bedrock's throttle text, not the fallback's 503
    overload (`call.rs:446`). Bedrock and Anthropic were each called once.
    """
    response: Final = chat(rig.gateway, CHAIN_PRIMARY)
    assert response.status_code == 429, response.text
    assert BEDROCK_THROTTLE_TEXT in error_message(response)
    assert rig.peer.count(CHAIN_PRIMARY) == 1
    assert rig.peer.count(CHAIN_TARGET) == 1


def test_disable_fallbacks_returns_the_providers_own_error_on_a_throttle(rig: Rig) -> None:
    """The Bedrock throttle primary is called with `disable_fallbacks: true` in the body.

    The caller gets HTTP 429 with Bedrock's text and the regular rescue group is not called.
    """
    regular_before: Final = rig.peer.count(REGULAR)
    response: Final = chat(rig.gateway, "fb-rl-bedrock", disable_fallbacks=True)
    assert response.status_code == 429, response.text
    assert BEDROCK_THROTTLE_TEXT in error_message(response)
    assert rig.peer.count(REGULAR) == regular_before
