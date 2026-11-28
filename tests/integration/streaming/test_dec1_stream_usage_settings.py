"""`general_settings.always_include_stream_usage` and what it does to a chat stream.

The setting has three states and each changes both the request the provider receives and the stream the
caller sees (`stream_usage_tracking_updates` in `litellm/proxy/common_request_processing.py`):

- unset (the shared proxy's state, covered in `test_dec1_stream_usage_and_billing.py`): the provider is
  asked for usage for billing, and the usage chunk is hidden from a caller who did not ask for it.
- `true`: a caller who sent no `stream_options`, or `stream_options` without `include_usage`, gets the
  usage chunk anyway. A caller who explicitly sent `include_usage: false` is left alone.
- `false`: the proxy adds nothing. The provider only reports usage when the caller asked.

Each test boots its own proxy from the shared config plus one static deployment, because the setting is
read from the config file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Wire, wire_server
from integration.streaming.dec1_stream_support import (
    chunks_of,
    content_of,
    marker,
    request_json,
    serving,
    text_frames,
    usage_chunks,
)

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(2 * graceful_stop_seconds() + 120)]

_MODEL: Final = "dec1-stream-usage-setting"


@contextmanager
def _proxy(gateway: Gateway, tmp_path: Path, always_include: bool) -> Iterator[tuple[Gateway, Wire]]:
    """An owned proxy whose only extra deployment streams `settled text` with usage 11, 4 and 15."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("settled ", "text"))
    with wire_server(serving(frames)) as wire:
        shared: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config: Final = {
            **shared,
            "model_list": [
                {
                    "model_name": _MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "synthetic-provider-key",
                        "api_base": wire.url + "/v1",
                    },
                }
            ],
            "general_settings": {**shared["general_settings"], "always_include_stream_usage": always_include},
        }
        path: Final = tmp_path / "dec1-stream-usage-setting.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            yield candidate, wire


def _stream(candidate: Gateway, **extra: object) -> list[dict]:
    response: Final = candidate.client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": f"setting probe {marker()}"}], "stream": True, **extra},
        headers={"Authorization": f"Bearer {candidate.key}"},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return chunks_of(response.text)


def _provider_stream_options(wire: Wire) -> list[object]:
    return [request_json(request).get("stream_options") for request in wire.drain() if request.method == "POST"]


def test_setting_true_gives_a_usage_chunk_to_a_caller_who_sent_no_stream_options(gateway: Gateway, tmp_path: Path) -> None:
    """With `always_include_stream_usage: true`, a caller streams without `stream_options`. The caller's
    stream contains exactly one chunk with `usage` (11, 4, 15) after the text, and the provider was
    asked with `stream_options: {"include_usage": true}`. The exact `choices` of that chunk (empty, or
    one empty choice as for deployment level stream options) is not pinned here."""
    with _proxy(gateway, tmp_path, always_include=True) as (candidate, wire):
        chunks: Final = _stream(candidate)
        assert content_of(chunks) == "settled text"
        usages: Final = usage_chunks(chunks)
        assert len(usages) == 1 and usages[0] is chunks[-1]
        assert object_value(usages[0]["usage"])["total_tokens"] == 15
        assert _provider_stream_options(wire) == [{"include_usage": True}]


def test_setting_true_adds_include_usage_to_stream_options_that_lack_it(gateway: Gateway, tmp_path: Path) -> None:
    """With the setting `true`, a caller sends `stream_options: {}`. The provider receives
    `stream_options` with `include_usage: true` merged in, and the caller gets a usage chunk. Other keys
    the caller put in `stream_options` must survive the merge."""
    with _proxy(gateway, tmp_path, always_include=True) as (candidate, wire):
        chunks: Final = _stream(candidate, stream_options={})
        assert len(usage_chunks(chunks)) == 1
        assert _provider_stream_options(wire) == [{"include_usage": True}]


def test_setting_true_respects_an_explicit_include_usage_false(gateway: Gateway, tmp_path: Path) -> None:
    """With the setting `true`, a caller explicitly sends `stream_options: {"include_usage": false}`. The
    proxy does not override it: the provider receives `include_usage: false` and the caller's stream has
    no usage chunk. The setting fills in a missing choice; it does not overrule a stated one."""
    with _proxy(gateway, tmp_path, always_include=True) as (candidate, wire):
        chunks: Final = _stream(candidate, stream_options={"include_usage": False})
        assert usage_chunks(chunks) == []
        assert _provider_stream_options(wire) == [{"include_usage": False}]


def test_setting_false_sends_no_stream_options_to_the_provider(gateway: Gateway, tmp_path: Path) -> None:
    """With `always_include_stream_usage: false`, a caller streams without `stream_options`. The provider
    request has no `stream_options` key at all and the caller's stream has no usage chunk. Operators set
    this for providers that reject the field."""
    with _proxy(gateway, tmp_path, always_include=False) as (candidate, wire):
        chunks: Final = _stream(candidate)
        assert content_of(chunks) == "settled text"
        assert usage_chunks(chunks) == []
        assert _provider_stream_options(wire) == [None]


def test_setting_false_still_forwards_a_callers_own_include_usage(gateway: Gateway, tmp_path: Path) -> None:
    """With the setting `false`, a caller who asks for usage still gets it: the provider receives
    `stream_options: {"include_usage": true}` and the caller's last chunk has `choices: []` and usage
    11, 4 and 15."""
    with _proxy(gateway, tmp_path, always_include=False) as (candidate, wire):
        chunks: Final = _stream(candidate, stream_options={"include_usage": True})
        assert chunks[-1]["choices"] == []
        usage: Final = object_value(chunks[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15)
        assert json.dumps(_provider_stream_options(wire)) == json.dumps([{"include_usage": True}])
