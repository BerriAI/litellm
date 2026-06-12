"""compat_httpx response parsing: two v1 construction styles, no seam preset.

On the httpx path the response starts from a FRESH ``ModelResponse``
(model=None) — the xai R4 rule; the seam must NOT preset
``{provider}/{model}`` for this family. The nine configs then split by HOW
v1 materializes the response (pinned in-process at HEAD; the family's
``RESPONSE_STYLES`` table is the truth the seam fork must read; the
construction-arm gate in the request differential makes the
``response_dialect()`` shortcut a test failure — verifier-wave1b F3):

- ``"openai"`` (cdr style): heroku / minimax / ovhcloud / cometapi inherit
  the base GPT ``transform_response`` -> ``convert_to_model_response_object``
  — the same live normalizer the openai parser mirrors (stop->tool_calls
  rewrite, cdr-built choices). Parser: ``openai_compat.parse_response``
  verbatim. cometapi's no-prefix pins live in
  test_differential_cometapi_response.py (researcher-4 §5: cdr via the base
  transform_response).
- ``"openai_like"`` (direct style): bedrock_mantle / datarobot /
  gradient_ai (OpenAILikeChatConfig._transform_response), compactifai (its
  own override, same construction), amazon_nova / lemonade (OpenAILike via
  super + a model overwrite) all build ``ModelResponse(**response_json)``
  DIRECTLY: NO finish rewrite, and the pydantic dump differs from
  cdr-built objects. Parser: the openai parser for fail-closed shape
  validation, then the VERBATIM raw body rides ``ChatResponse.wire`` (with
  the model overwritten to ``f"{prefix}/{REQUEST model}"`` for
  compactifai/amazon-nova/lemonade — the literal "amazon-nova" hyphen, wire
  model ignored) so the seam's ``openai_like`` arm reproduces v1's
  construction byte-for-byte.

The OpenAILike usage-null sanitize is observationally dead (``Usage``
coerces None -> 0 in the constructor on every path — differential-pinned);
the OpenAILike json_mode tool->content conversion is dormant here (no
config in this family ever sets json_mode).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

from expression import Ok, Result, Some

from ...errors import TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response
from .params import CompatHttpxProvider
from .serialize import PROFILES

_ParseResult = Result[ChatResponse, TranslationError]
ResponseParser = Callable[[PlainJson, ChatRequest], _ParseResult]

ResponseStyle = Literal["openai", "openai_like"]

_CDR_STYLE: frozenset[CompatHttpxProvider] = frozenset(
    {"heroku", "minimax", "ovhcloud", "cometapi"}
)

RESPONSE_STYLES: Mapping[CompatHttpxProvider, ResponseStyle] = MappingProxyType(
    {
        provider: ("openai" if provider in _CDR_STYLE else "openai_like")
        for provider in PROFILES
    }
)


def _direct_parser(prefix: str | None) -> ResponseParser:
    def parse(raw: PlainJson, request: ChatRequest) -> _ParseResult:
        return openai_parse_response(raw, request).bind(
            lambda response: _verbatim_wire(response, raw, request, prefix)
        )

    return parse


def _verbatim_wire(
    response: ChatResponse, raw: PlainJson, request: ChatRequest, prefix: str | None
) -> _ParseResult:
    body: dict[str, PlainJson] = dict(raw) if isinstance(raw, dict) else {}
    if prefix is not None:
        model = f"{prefix}/{request.model}"
        body = {**body, "model": model}
        return Ok(replace(response, model=model, wire=Some(JsonBlob(value=body))))
    return Ok(replace(response, wire=Some(JsonBlob(value=body))))


PARSERS: Mapping[CompatHttpxProvider, ResponseParser] = MappingProxyType(
    {
        provider: (
            openai_parse_response
            if RESPONSE_STYLES[provider] == "openai"
            else _direct_parser(profile.response_model_prefix)
        )
        for provider, profile in PROFILES.items()
    }
)
