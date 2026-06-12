"""Differential parity for compat_sdk (wave-1a) streams, pinned per provider.

The wave's stream claim is that every surviving provider rides
``CustomStreamWrapper``'s DEFAULT openai arm (no dedicated branch, not in
``litellm._custom_providers``) and therefore decodes identically to v2's
``openai`` chunk dialect. That claim is pinned by replaying the openai SDK
chunk corpus through the wrapper with each provider's
``custom_llm_provider`` and requiring byte-identical output to the v2 fold —
a provider growing a dedicated wrapper branch (the baseten failure mode)
breaks its row here.

baseten is the dropped 14th provider for exactly that reason; the canary at
the bottom pins the evidence so the drop reason stays true at HEAD.
"""

import copy
import json
import time
from typing import get_args

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from litellm.translation.dispatch import Provider
from litellm.translation.engine import pipeline

from ._compat_sdk_corpus import SDK_PROVIDERS
from .test_differential_openai_stream import MODEL, STREAMS, USAGE_STREAM, _v2_chunks

_STREAM_ROWS = ("text", "tools", "empty_keepalive_swallowed")


def _v1_chunks(provider: str, events: list, stream_options=None) -> list:
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id=f"diff-{provider}-stream",
        function_id=f"diff-{provider}-stream",
    )
    sdk_chunks = (
        ChatCompletionChunk.model_validate(event) for event in copy.deepcopy(events)
    )
    wrapper = CustomStreamWrapper(
        completion_stream=sdk_chunks,
        model=MODEL,
        custom_llm_provider=provider,
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


def _rows():
    # SDK-path members only: cometapi's v1 decode is the httpx line seam
    # (its own chunk handler), pinned in test_differential_cometapi_stream.py.
    return sorted(
        (provider, name) for provider in SDK_PROVIDERS for name in _STREAM_ROWS
    )


@pytest.mark.parametrize("provider,name", _rows())
def test_v2_stream_matches_v1(provider: str, name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(provider, events))


_CITATIONS_STREAM = [
    {
        "id": "chunk-cite",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hi"},
                "finish_reason": None,
            }
        ],
        "citations": ["https://a.example"],
    },
    {
        "id": "chunk-cite",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": None}],
    },
    {
        "id": "chunk-cite",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "citations": ["https://a.example"],
    },
]


def test_perplexity_wire_citations_survive_like_v1(frozen_ambient) -> None:
    """Real perplexity streams carry top-level ``citations``: v1 copies the
    wire value onto content chunks (and the citations-bearing finish chunk
    via preserve_upstream), presetting None only where the wire carried
    none. The seam's citations preset must NOT clobber a body-carried value
    (the wave-2a to_model_response_stream fix; the de-biased middle chunk
    pins the None preset direction too)."""
    v1 = _v1_chunks("perplexity", _CITATIONS_STREAM)
    v2 = _v2_chunks(_CITATIONS_STREAM)
    assert _norm(v2) == _norm(v1)
    assert v2[0]["citations"] == ["https://a.example"]
    assert v2[1]["citations"] is None
    assert v2[2]["citations"] == ["https://a.example"]


@pytest.mark.parametrize("provider", SDK_PROVIDERS)
def test_usage_tail_seam_contract(provider: str, frozen_ambient) -> None:
    """Same contract the openai stream differential pins: byte-identical
    prefix; v1's tail is the wrapper-synthesized usage chunk, v2's tail is
    the wire ``choices: []`` usage chunk verbatim with equal numbers."""
    v1 = _v1_chunks(provider, USAGE_STREAM, stream_options={"include_usage": True})
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2)
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    assert v2[-1]["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1[-1]["usage"][key] == v2[-1]["usage"][key]


def test_baseten_drop_canary(frozen_ambient) -> None:
    """baseten is DROPPED from wave 1a: the wrapper routes its chunks into a
    dedicated legacy decoder (``handle_baseten_chunk``,
    streaming_handler.py:1246-1248) instead of the default openai arm, so
    its v1 stream behavior is not the openai dialect and cannot be honestly
    registered. If the DRIFT half fails, the legacy branch is gone at HEAD —
    re-evaluate porting baseten (request/response sides are trivial:
    own list, mct rename, user supported). The REGISTRATION half makes the
    steady state loud (critic-wave1a M2 / verifier F3): registering baseten
    anywhere requires deliberately deleting these negative asserts, which
    forces the registrar past the evidence above."""
    # drift half: the legacy branch still exists and still diverges
    assert hasattr(CustomStreamWrapper, "handle_baseten_chunk")
    v1_baseten = _v1_chunks("baseten", STREAMS["text"])
    v1_openai = _v1_chunks("custom_openai", STREAMS["text"])
    assert _norm(v1_baseten) != _norm(v1_openai)
    # registration half: naive registration fails HERE, corpus row or not
    assert "baseten" not in get_args(Provider)
    assert "baseten" not in pipeline._SERIALIZERS
    assert "baseten" not in pipeline._RESPONSE_PARSERS
    assert "baseten" not in pipeline._RESPONSE_DIALECTS
    assert "baseten" not in pipeline._RAW_GUARDS
