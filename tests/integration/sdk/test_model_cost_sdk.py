import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_MODEL: Final = "lit-sdk-chat"
_ALIAS: Final = "lit-sdk-chat-alias"
_PROMPT: Final = 1000
_CACHED: Final = 400
_COMPLETION: Final = 500
_CUSTOM_INPUT: Final = 9e-06
_CUSTOM_OUTPUT: Final = 7e-06
_CHARGES: Final = TypeAdapter(list[float | bool | None])
_BUNDLED: Final = TypeAdapter(dict[str, JsonValue]).validate_json(
    files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()
)


class _Requested(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str


_REQUEST: Final = TypeAdapter(_Requested)


@dataclass(frozen=True, slots=True)
class _Rates:
    input: float
    output: float
    cache_read: float

    def cost(self) -> float:
        return (_PROMPT - _CACHED) * self.input + _CACHED * self.cache_read + _COMPLETION * self.output


def _catalog(scale: float) -> _Rates:
    return _Rates(input=1e-06 * scale, output=2e-06 * scale, cache_read=5e-07 * scale)


def _document(scale: float) -> bytes:
    rates: Final = _catalog(scale)
    return json.dumps(
        {
            **_BUNDLED,
            _MODEL: {
                "aliases": [_ALIAS],
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": rates.input,
                "output_cost_per_token": rates.output,
                "cache_read_input_token_cost": rates.cache_read,
            },
        }
    ).encode()


def _respond(request: Request) -> Reply:
    match request.target:
        case "/p1.json":
            return Reply(body=_document(1.0))
        case "/p2.json":
            return Reply(body=_document(2.0))
        case "/missing.json":
            return Reply(status=404, body=b"{}")
        case _:
            return Reply(
                body=json.dumps(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": _REQUEST.validate_json(request.body).model,
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "billed"}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": _PROMPT,
                            "completion_tokens": _COMPLETION,
                            "total_tokens": _PROMPT + _COMPLETION,
                            "prompt_tokens_details": {"cached_tokens": _CACHED},
                        },
                    }
                ).encode()
            )


@pytest.fixture(scope="module")
def cost_map_peer() -> Iterator[Wire]:
    with wire_server(_respond) as peer:
        yield peer


def _run(case: str, peer: Wire, directory: Path) -> list[float | bool | None]:
    document: Final = directory / "p2.json"
    document.write_bytes(_document(2.0))
    environment: Final = {
        **{name: value for name, value in os.environ.items() if name != "LITELLM_LOCAL_MODEL_COST_MAP"},
        "LITELLM_MODEL_COST_MAP_URL": f"{peer.url}/p1.json",
        "MODEL_COST_SDK_BASE_URL": peer.url,
        "MODEL_COST_SDK_DOCUMENT": str(document),
    }
    result: Final = subprocess.run(  # test-quality-ok: each case is one SDK session in its own interpreter
        (sys.executable, "-m", "integration.sdk._model_cost_sdk_child", case),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return _CHARGES.validate_json(result.stdout.strip().splitlines()[-1])


def _custom_input(rates: _Rates) -> _Rates:
    return _Rates(input=_CUSTOM_INPUT, output=rates.output, cache_read=rates.cache_read)


def _custom_input_and_output(rates: _Rates) -> _Rates:
    return _Rates(input=_CUSTOM_INPUT, output=_CUSTOM_OUTPUT, cache_read=rates.cache_read)


def _charged(rates: _Rates) -> object:
    return pytest.approx(rates.cost(), rel=1e-9)


def _billed_and_listed(rates: _Rates) -> list[object]:
    return [_charged(rates), pytest.approx(rates.input, rel=1e-9), pytest.approx(rates.output, rel=1e-9)] * 3


_P1: Final = _catalog(1.0)
_P2: Final = _catalog(2.0)
_P3: Final = _catalog(3.0)
_FINETUNE_BUG: Final = pytest.mark.xfail(
    strict=True,
    reason="BUG: a fine-tune registered with only an input price bills output at 0 and cache reads at input",
)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("reload_reprices", [_charged(_P1), True, _charged(_P2)]),
        ("failed_reload_keeps_prices", [_charged(_P1), False, _charged(_P1)]),
        (
            "every_name_bills_and_lists_the_row_across_a_reload",
            [*_billed_and_listed(_P1), True, *_billed_and_listed(_P2)],
        ),
        ("registration_survives_reload", [_charged(_P3), _charged(_P3), True, _charged(_P3), None]),
        ("partial_registration_follows_reload", [_charged(_custom_input(_P1)), True, _charged(_custom_input(_P2))]),
        (
            "partial_registrations_merge_and_follow_reload",
            [_charged(_custom_input_and_output(_P1)), True, _charged(_custom_input_and_output(_P2))],
        ),
        ("finetune_registration_bills_the_same_after_a_reload", [True, True, True]),
        pytest.param(
            "finetune_registration_inherits_the_base_prices", [_charged(_custom_input(_P1))], marks=_FINETUNE_BUG
        ),
        ("new_row_after_a_miss_without_invalidating", [None, _charged(_P3)]),
        ("value_edited_in_place_then_invalidate", [_charged(_P1), _charged(_custom_input(_P1))]),
        ("table_swapped_then_invalidate", [_charged(_P1), _charged(_P2)]),
        ("table_swapped_then_helper_cache_cleared", [_charged(_P1), _charged(_P2)]),
        ("contents_replaced_then_invalidate", [_charged(_P1), _charged(_P2)]),
        ("finalized_snapshot_replaces_contents", [_charged(_P2)]),
        (
            "alias_registration_after_an_adopted_reload",
            [True, _charged(_custom_input(_P2)), _charged(_custom_input(_P2))],
        ),
        ("small_map_finalizes_and_leaves_the_live_map", [True, _charged(_P1)]),
        ("popped_row_then_registered", [_charged(_P1), _charged(_P3)]),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_sdk_code_that_reads_or_writes_the_cost_map_gets_what_it_set_up(
    case: str, expected: list[object], cost_map_peer: Wire, tmp_path: Path
) -> None:
    assert _run(case, cost_map_peer, tmp_path) == expected
