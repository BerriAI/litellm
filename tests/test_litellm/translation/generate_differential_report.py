"""Regenerate DIFFERENTIAL_REPORT.md: the v1-vs-v2 parity merge artifact.

Run:  python -m tests.test_litellm.translation.generate_differential_report
"""

import json
import pathlib
import subprocess
import sys
import time
import uuid

_HERE = pathlib.Path(__file__).parent


def _freeze_ambient() -> None:
    import fastuuid

    import litellm._uuid

    counter = iter(range(1, 1_000_000))
    fake = lambda: uuid.UUID(int=next(counter))  # noqa: E731
    uuid.uuid4 = fake  # type: ignore[assignment]
    fastuuid.uuid4 = fake  # type: ignore[assignment]
    litellm._uuid.uuid4 = fake  # type: ignore[assignment]
    time.time = lambda: 1718064000.0  # type: ignore[assignment]


def _anthropic_rows(lines: list) -> int:
    from . import test_differential_anthropic_request as req
    from . import test_differential_anthropic_response as resp
    from . import test_differential_anthropic_stream as stream

    failures = 0
    lines += [
        "## anthropic: request bodies (v1 map_openai_params + transform_request vs v2)",
        "",
    ]
    for name in sorted(req.CORPUS):
        same = req._norm(req._v2_body(req.CORPUS[name])) == req._norm(
            req._v1_body(req.CORPUS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += ["", "## anthropic: responses (v1 transform_response vs v2)", ""]
    for name in sorted(resp._REQUESTS):
        same = resp._norm(resp._v2_model_response(name)) == resp._norm(
            resp._v1_model_response(name)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## anthropic: streams (v1 CustomStreamWrapper replay vs v2 engine/stream)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        same = stream._norm(stream._v2_chunks(stream.STREAMS[name])) == stream._norm(
            stream._v1_chunks(stream.STREAMS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    return failures


def _openai_rows(lines: list) -> int:
    from litellm.translation import translate_chat_request

    from . import test_differential_openai_request as req
    from . import test_differential_openai_response as resp
    from . import test_differential_openai_stream as stream
    from .conftest import build_real_deps

    failures = 0
    lines += [
        "",
        "## openai_compat: request bodies (v1 map_openai_params + transform_request vs v2)",
        "",
    ]
    for name in sorted(req.CORPUS):
        result = req._v2_body(req.CORPUS[name])
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req._v1_body(req.CORPUS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.EXPECTED_FALLBACKS):
        case, reason = req.EXPECTED_FALLBACKS[name]
        result = translate_chat_request(dict(case), "openai_compat", build_real_deps())
        ok = result.is_error() and reason in result.error.summary
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## openai_compat: responses (v1 convert_to_model_response_object vs v2)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        same = resp._norm(resp._v2_model_response(resp._RESPONSES[name])) == resp._norm(
            resp._v1_model_response(resp._RESPONSES[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for preset in (resp.PRESET_PREFIXED_MODEL, "no-slash-model"):
        same = resp._norm(
            resp._v2_model_response(resp._RESPONSES["text"], preset_model=preset)
        ) == resp._norm(
            resp._v1_model_response(resp._RESPONSES["text"], preset_model=preset)
        )
        failures += 0 if same else 1
        lines.append(
            f"- {'IDENTICAL' if same else 'DIVERGENT'}: "
            f"text (pre-set model_response.model={preset!r}; the compat"
            " provider/wire-model re-prefix arm)"
        )
    lines += [
        "",
        "## openai_compat: streams (v1 CustomStreamWrapper over SDK chunks vs v2 fold)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        same = stream._norm(stream._v2_chunks(stream.STREAMS[name])) == stream._norm(
            stream._v1_chunks(stream.STREAMS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through; v1's"
        " wrapper synthesizes its final usage chunk from it, which is the"
        " streaming seam's envelope to reproduce)"
    )
    return failures


def _xai_rows(lines: list) -> int:
    from litellm.exceptions import UnsupportedParamsError

    from . import _xai_corpus as corpus
    from . import test_differential_xai_request as req
    from . import test_differential_xai_response as resp
    from . import test_differential_xai_stream as stream

    failures = 0
    lines += [
        "",
        "## xai: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON; v1 = get_optional_params('xai') + transform_request)",
        "",
    ]
    for name in sorted(req.CASES):
        case = req.CASES[name]
        snapshot = (corpus.SNAPSHOTS_DIR / "requests" / f"{name}.json").read_text()
        v1_same = corpus.canonical_json(corpus.run_v1_request_transform(case)) == (
            snapshot
        )
        result = req._v2(case)
        v2_same = result.is_ok() and req._norm(result.ok) == req._norm(
            corpus.load_json(corpus.SNAPSHOTS_DIR / "requests" / f"{name}.json")
        )
        same = v1_same and v2_same
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.V1_RAISES):
        case, reason = req.V1_RAISES[name]
        result = req._v2(case)
        try:
            corpus.run_v1_request_transform(case)
            raised = False
        except UnsupportedParamsError:
            raised = True
        ok = result.is_error() and reason in result.error.summary and raised
        failures += 0 if ok else 1
        label = "FALLBACK (v1 raises UnsupportedParamsError)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    for name in sorted(req.EXPECTED_FALLBACKS):
        case, reason = req.EXPECTED_FALLBACKS[name]
        result = req._v2(case)
        ok = result.is_error() and reason in result.error.summary
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## xai: responses (snapshot == v1 XAIChatConfig.transform_response == v2; the LIVE httpx-path normalizer incl. the usage post-steps)",
        "",
    ]
    responses = corpus.corpus("responses")
    for name in sorted(responses):
        row = responses[name]
        snapshot = (corpus.SNAPSHOTS_DIR / "responses" / f"{name}.json").read_text()
        same = (
            corpus.canonical_json(
                corpus.run_v1_response_transform(row["body"], row["model"])
            )
            == snapshot
            and corpus.canonical_json(
                resp._v2_model_response(row["body"], row["model"])
            )
            == snapshot
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## xai: streams (snapshot == v1 line-seam replay through XAIChatCompletionStreamingHandler + CustomStreamWrapper('xai') == v2 xai dialect)",
        "",
    ]
    streams = corpus.corpus("streams")
    for name in sorted(streams):
        row = streams[name]
        snapshot_text = (corpus.SNAPSHOTS_DIR / "streams" / f"{name}.json").read_text()
        v1_same = (
            corpus.canonical_json(
                corpus.replay_xai_sse_lines(row["events"], row["stream_options"])
            )
            == snapshot_text
        )
        if name == stream._TAIL_ROW:
            snapshot = corpus.load_json(
                corpus.SNAPSHOTS_DIR / "streams" / f"{name}.json"
            )
            v2 = stream._v2_chunks(row["events"])
            tail_ok = (
                v1_same
                and len(v2) == len(snapshot)
                and stream._norm(v2[:-1]) == stream._norm(snapshot[: len(v2) - 1])
                and v2[-1]["choices"] == []
                and all(
                    snapshot[-1]["usage"][k] == v2[-1]["usage"][k]
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            )
            failures += 0 if tail_ok else 1
            lines.append(
                ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
                + f"{name} (v1's chunk_parser injects a dummy choice so the"
                " wrapper swallows the tail and synthesizes the final usage"
                " chunk; v2 passes the wire choices=[] chunk through with the"
                " FOLDED usage for the streaming seam to synthesize from)"
            )
            continue
        v2_same = stream._norm(stream._v2_chunks(row["events"])) == stream._norm(
            corpus.load_json(corpus.SNAPSHOTS_DIR / "streams" / f"{name}.json")
        )
        same = v1_same and v2_same
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    return failures


def _compat_sdk_rows(lines: list) -> int:
    from litellm.exceptions import UnsupportedParamsError

    from . import _compat_sdk_corpus as corpus
    from . import test_differential_compat_sdk_request as req
    from . import test_differential_compat_sdk_response as resp
    from . import test_differential_compat_sdk_stream as stream

    failures = 0
    for provider in corpus.PROVIDERS:
        lines += [
            "",
            f"## {provider}: request bodies (v1 get_optional_params('{provider}')"
            " + transform_request vs v2 compat_sdk)",
            "",
        ]
        for name in sorted(corpus.corpus_for(provider)):
            case = corpus.corpus_for(provider)[name]
            result = req._v2(provider, case)
            same = result.is_ok() and req._norm(result.ok) == req._norm(
                corpus.run_v1_request_transform(provider, case)
            )
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
        for name in sorted(k for k in req.V1_RAISES if k.startswith(f"{provider}:")):
            p, case, reason = req.V1_RAISES[name]
            result = req._v2(p, case)
            try:
                corpus.run_v1_request_transform(p, case)
                raised = False
            except UnsupportedParamsError:
                raised = True
            ok = result.is_error() and reason in result.error.summary and raised
            failures += 0 if ok else 1
            label = "FALLBACK (v1 raises UnsupportedParamsError)" if ok else "DIVERGENT"
            lines.append(f"- {label}: {name} ({reason})")
        for name in sorted(
            k for k in req.V1_RAISES_VALUE_ERROR if k.startswith(f"{provider}:")
        ):
            p, case, reason = req.V1_RAISES_VALUE_ERROR[name]
            result = req._v2(p, case)
            try:
                corpus.run_v1_request_transform(p, case)
                raised = False
            except ValueError as err:
                raised = type(err) is ValueError
            ok = result.is_error() and reason in result.error.summary and raised
            failures += 0 if ok else 1
            label = "FALLBACK (v1 raises ValueError)" if ok else "DIVERGENT"
            lines.append(f"- {label}: {name} ({reason})")
        for name in sorted(
            k for k in req.EXPECTED_FALLBACKS if k.startswith(f"{provider}:")
        ):
            p, case, reason = req.EXPECTED_FALLBACKS[name]
            result = req._v2(p, case)
            ok = result.is_error() and reason in result.error.summary
            failures += 0 if ok else 1
            label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
            lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## compat_sdk family: responses (v1 convert_to_model_response_object"
        " with the SDK-path {provider}/{model} preset vs v2 + seam re-prefix"
        " arm; cometapi is a compat_httpx row — its no-prefix rows are below)",
        "",
    ]
    for provider, name in resp._rows():
        raw = resp._RESPONSES[name]
        preset = f"{provider}/{corpus.SPECS[provider].model}"
        same = resp._norm(resp._v2_model_response(provider, raw, preset)) == resp._norm(
            resp._v1_model_response(raw, preset)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider} {name}")
    citations = resp._PERPLEXITY_CITATIONS_RESPONSE
    preset = "perplexity/sonar"
    v1_cite = resp._v1_model_response(citations, preset)
    v2_cite = resp._v2_model_response("perplexity", citations, preset)
    cite_ok = (
        resp._norm(v2_cite) == resp._norm(v1_cite)
        and v2_cite["citations"] == ["https://a.example"]
        and v2_cite["choices"][0]["message"].get("annotations") is None
        and "citation_tokens" not in v2_cite["usage"]
    )
    failures += 0 if cite_ok else 1
    lines.append(
        ("- IDENTICAL: " if cite_ok else "- DIVERGENT: ")
        + "perplexity citations dormancy (transform_response's annotation/"
        "citation-token enrichment is DEAD on the SDK path; citations/"
        "search_results survive via cdr's unknown-key mirror only)"
    )
    lines += [
        "",
        "## compat_sdk family: streams (v1 CustomStreamWrapper(provider) over"
        " SDK chunks vs v2 openai dialect; SDK-path members only)",
        "",
    ]
    for provider, name in stream._rows():
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(provider, events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider} {name}")
    cite_stream_ok = stream._norm(
        stream._v2_chunks(stream._CITATIONS_STREAM)
    ) == stream._norm(stream._v1_chunks("perplexity", stream._CITATIONS_STREAM))
    failures += 0 if cite_stream_ok else 1
    lines.append(
        ("- IDENTICAL: " if cite_stream_ok else "- DIVERGENT: ")
        + "perplexity wire-carried citations (body value survives the seam's"
        " citations preset; None preset where the wire carried none)"
    )
    for provider in corpus.PROVIDERS:
        v1 = stream._v1_chunks(
            provider, stream.USAGE_STREAM, stream_options={"include_usage": True}
        )
        v2 = stream._v2_chunks(stream.USAGE_STREAM)
        tail_ok = (
            len(v1) == len(v2)
            and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
            and v2[-1]["choices"] == []
            and all(
                v1[-1]["usage"][k] == v2[-1]["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        )
        failures += 0 if tail_ok else 1
        lines.append(
            ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
            + f"{provider} usage tail (v2 passes the wire choices=[] usage"
            " chunk through; the streaming seam owns v1's synthesized final"
            " chunk)"
        )
    lines += [
        "",
        "- DROPPED FROM WAVE 1A: baseten (streams ride the dedicated legacy"
        " handle_baseten_chunk wrapper branch, not the openai dialect;"
        " unregistered, typed v1 fallback; canary"
        " test_baseten_drop_canary pins the evidence)",
        "- DROPPED FROM WAVE 1B: aiml (AIMLChatConfig unregistered at HEAD;"
        " v1 serves it through the generic openai fallback stack whose mct"
        " rename flips the day the config registers; canary"
        " test_aiml_drop_canary)",
        "- DROPPED FROM WAVE 1B: veniceai, abliteration, llamagate, gmi,"
        " sarvam, aihubmix, crusoe (JSON-registry rows WITHOUT LlmProviders"
        " enum membership: no provider config at param/transform time, the"
        " JSON gates are dead in v1; canary"
        " test_json_non_enum_providers_stay_dropped)",
    ]
    return failures


def _compat_httpx_rows(lines: list) -> int:
    from litellm.exceptions import UnsupportedParamsError

    from . import _compat_httpx_corpus as corpus
    from . import test_differential_compat_httpx_request as req
    from . import test_differential_compat_httpx_response as resp
    from . import test_differential_compat_httpx_stream as stream

    failures = 0
    for provider in corpus.PROVIDERS:
        lines += [
            "",
            f"## {provider}: request bodies (v1 get_optional_params('{provider}')"
            " + the LIVE httpx transform_request vs v2 compat_httpx)",
            "",
        ]
        for name in sorted(corpus.corpus_for(provider)):
            case = corpus.corpus_for(provider)[name]
            result = req._v2(provider, case)
            same = result.is_ok() and req._norm(result.ok) == req._norm(
                corpus.run_v1_request_transform(provider, case)
            )
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
        for name in sorted(k for k in req.V1_RAISES if k.startswith(f"{provider}:")):
            p, case, reason = req.V1_RAISES[name]
            result = req._v2(p, case)
            try:
                corpus.run_v1_request_transform(p, case)
                raised = False
            except UnsupportedParamsError:
                raised = True
            ok = result.is_error() and reason in result.error.summary and raised
            failures += 0 if ok else 1
            label = "FALLBACK (v1 raises UnsupportedParamsError)" if ok else "DIVERGENT"
            lines.append(f"- {label}: {name} ({reason})")
        for name in sorted(
            k for k in req.EXPECTED_FALLBACKS if k.startswith(f"{provider}:")
        ):
            p, case, reason = req.EXPECTED_FALLBACKS[name]
            result = req._v2(p, case)
            ok = result.is_error() and reason in result.error.summary
            failures += 0 if ok else 1
            label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
            lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## compat_httpx family: responses (v1's LIVE transform_response over a"
        " FRESH ModelResponse — no seam preset; cdr style for heroku/minimax/"
        "ovhcloud, ModelResponse(**json) style for the rest — vs v2 family"
        " parser + the per-style seam arm; includes the request-model prefix"
        " pins and the usage-null row)",
        "",
    ]
    for provider, name in resp._rows():
        raw = resp._RESPONSES[name]
        v1 = corpus.run_v1_response_transform(
            provider, raw, corpus.SPECS[provider].model
        ).model_dump()
        same = resp._norm(resp._v2_model_response(provider, raw)) == resp._norm(v1)
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider} {name}")
    for provider in corpus.PROVIDERS:
        v1 = corpus.run_v1_response_transform(
            provider, resp._NULL_USAGE_BODY, corpus.SPECS[provider].model
        ).model_dump()
        same = resp._norm(
            resp._v2_model_response(provider, resp._NULL_USAGE_BODY)
        ) == resp._norm(v1)
        failures += 0 if same else 1
        lines.append(
            f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider} usage_null_tokens"
        )
    lines += [
        "",
        "## compat_httpx family: non-string wire model (v1's OpenAILike"
        " construction raises pydantic ValidationError BEFORE the prefix"
        " overwrite; v2's family parser fails closed so the typed fallback"
        " reproduces the raise — verifier-longtail F2)",
        "",
    ]
    from pydantic import ValidationError

    for provider in sorted(
        p for p in corpus.PROVIDERS if corpus.SPECS[p].prefix is not None
    ):
        try:
            corpus.run_v1_response_transform(
                provider, resp._NON_STRING_MODEL_BODY, corpus.SPECS[provider].model
            )
            raised = False
        except ValidationError:
            raised = True
        result = resp._v2_parse_result(provider, resp._NON_STRING_MODEL_BODY)
        ok = (
            raised
            and result.is_error()
            and "non-string wire model" in result.error.summary
        )
        failures += 0 if ok else 1
        label = "FALLBACK (v1 raises ValidationError)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {provider} non_string_wire_model")
    lines += [
        "",
        "## compat_httpx family: streams (v1 base"
        " OpenAIChatCompletionStreamingHandler + CustomStreamWrapper(provider)"
        " over SSE lines vs v2 family parser with the xai chunk dialect)",
        "",
    ]
    for provider, name in stream._rows():
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            corpus.replay_v1_sse_lines(provider, events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider} {name}")
    for provider in corpus.PROVIDERS:
        v1 = corpus.replay_v1_sse_lines(
            provider, stream.USAGE_STREAM, stream_options={"include_usage": True}
        )
        v2 = stream._v2_chunks(stream.USAGE_STREAM)
        tail_ok = (
            len(v1) == len(v2)
            and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
            and v2[-1]["choices"] == []
            and all(
                v1[-1]["usage"][k] == v2[-1]["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        )
        failures += 0 if tail_ok else 1
        lines.append(
            ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
            + f"{provider} usage tail (v2 passes the wire choices=[] usage"
            " chunk through; the streaming seam owns v1's synthesized final"
            " chunk)"
        )
    import json as _json

    from litellm.translation.engine.stream import fold_lines as _fold_lines
    from litellm.translation.inbound.openai_chat.stream import (
        initial_state as _initial_state,
    )
    from litellm.translation.providers.compat_httpx.stream import (
        parse_line as _httpx_parse_line,
    )

    error_events = stream._ERROR_CHUNK_STREAM
    v1_swallow = corpus.replay_v1_sse_lines("heroku", error_events)
    error_lines = [f"data: {_json.dumps(e)}" for e in error_events]
    v2_loud = _fold_lines(
        error_lines,
        _httpx_parse_line,
        _initial_state(stream.STREAM_MODEL, dialect="xai"),
    )
    divergence_pinned = (
        "error" not in _json.dumps(v1_swallow)
        and v2_loud.is_error()
        and "provider stream error" in v2_loud.error.summary
    )
    failures += 0 if divergence_pinned else 1
    lines.append(
        (
            "- PINNED DIVERGENCE (fail-closed on a failure path): "
            if divergence_pinned
            else "- DIVERGENT: "
        )
        + "mid-stream {'error': ...} chunks — v1's BASE handler silently"
        " swallows them (no error surface in the emitted sequence; asserted"
        " in-process for all nine base-handler members), v2's family parser"
        " surfaces a LOUD typed boundary error naming the chunk"
        " (test_error_chunk_divergence_two_sided; cometapi differs: its v1"
        " handler RAISES and its policy row mirrors the raise — see the"
        " cometapi stream rows below)"
    )
    return failures


def _cometapi_rows(lines: list) -> int:
    from . import test_differential_cometapi_response as resp
    from . import test_differential_cometapi_stream as stream

    failures = 0
    lines += [
        "",
        "## cometapi: responses (v1 CometAPIConfig.transform_response over"
        " httpx — LIVE on the dedicated elif, main.py:2547 — vs v2 shared"
        " openai parser with NO model preset; bare wire model, the xai R4 pin)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        v1 = resp._v1_model_response(raw)
        v2 = resp._v2_model_response(raw)
        same = (
            resp._norm(v2) == resp._norm(v1)
            and v2["model"] == raw["model"]
            and not str(v2["model"]).startswith("cometapi/")
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name} (no prefix)")
    lines += [
        "",
        "## cometapi: streams (v1 line-seam replay through"
        " CometAPIChatCompletionStreamingHandler + CustomStreamWrapper"
        "('cometapi') vs v2 cometapi parser + the shared xai chunk dialect)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    return failures


def _cohere_rows(lines: list) -> int:
    """wave-2b-beta: cohere/cohere_chat (the v2 chat wire, the DEFAULT
    route at HEAD)."""
    import copy

    from . import test_differential_cohere_request as req
    from . import test_differential_cohere_response as resp
    from . import test_differential_cohere_stream as stream

    failures = 0
    lines += [
        "",
        "## cohere v2 (wave-2b-beta): requests (v1 get_optional_params +"
        " CohereV2ChatConfig.transform_request vs v2 providers/cohere — both"
        " provider names, cohere and cohere_chat)",
        "",
    ]
    for provider, name in req._rows(req.CASES):
        case = req.CASES[name]
        result = req._v2(provider, case)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req.run_v1_request_transform(provider, case)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {provider}/{name}")
    for provider, name in req._rows(req.V1_RAISES):
        case, fragment = req.V1_RAISES[name]
        result = req._v2(provider, case)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                req.run_v1_request_transform(provider, case)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises UnsupportedParamsError)' if ok else 'DIVERGENT'}:"
            f" {provider}/{name}"
        )
    for provider, name in req._rows(req.V1_SERVES_FALLBACKS):
        case, fragment = req.V1_SERVES_FALLBACKS[name]
        result = req._v2(provider, case)
        ok = result.is_error() and fragment in result.error.summary
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: {provider}/{name}"
        )
    lines += [
        "",
        "## cohere v2: responses (v1 CohereV2ChatConfig.transform_response —"
        " fresh-ModelResponse mutation, request model verbatim, finish always"
        " stop — vs v2 cohere parser + the seam's openai construction arm)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        v1 = resp._v1_model_response(raw)
        v2 = resp._v2_model_response(raw)
        same = (
            resp._norm(v2) == resp._norm(v1)
            and v2["model"] == resp.MODEL
            and v2["choices"][0]["finish_reason"] == "stop"
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(resp._LOUD):
        raw, fragment = resp._LOUD[name]
        result = resp._v2_parse(raw)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                resp._v1_model_response(copy.deepcopy(raw))
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 raises)' if ok else 'DIVERGENT'}: {name}")
    for name in sorted(resp._V1_SERVES_FALLBACKS):
        raw, fragment = resp._V1_SERVES_FALLBACKS[name]
        result = resp._v2_parse(raw)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            v1 = resp._v1_model_response(copy.deepcopy(raw))
            ok = bool(v1["choices"][0]["message"].get("annotations"))
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 serves the unvalidated annotation)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    lines += [
        "",
        "## cohere v2: streams (v1 bare-JSON line replay through"
        " CohereV2ModelResponseIterator + CustomStreamWrapper('cohere_chat')"
        " generic arm vs v2 cohere parser + the generic chunk dialect; ids"
        " normalized — v1 mints a fresh chatcmpl id per chunk)",
        "",
    ]
    for name in sorted(stream.REAL_WIRE_STREAMS):
        events, synth_finish = stream.REAL_WIRE_STREAMS[name]
        v1 = stream._v1_chunks(events)
        v2 = stream._v2_chunks(events)
        ok = (
            len(v1) == len(v2) + 1
            and stream._norm(v2) == stream._norm(v1[:-1])
            and v1[-1]["choices"][0]["finish_reason"] == synth_finish
        )
        failures += 0 if ok else 1
        lines.append(
            ("- SEAM CONTRACT: " if ok else "- DIVERGENT: ")
            + f"real-wire {name} (v2 == v1 minus the wrapper's synthesized"
            f" {synth_finish} tail — the generic streaming seam owns it)"
        )
    for name in sorted(stream.EVENT_KEYED_STREAMS):
        events = stream.EVENT_KEYED_STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: event-keyed {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    divergence_event = {
        "type": "content-delta",
        "delta": {"message": {"content": {"text": 5}}},
    }
    v1_swallows = all(
        choice["delta"]["content"] in (None, "")
        for chunk in stream._v1_chunks([copy.deepcopy(divergence_event)])
        for choice in chunk["choices"]
    )
    v2_loud = stream.parse_event(copy.deepcopy(divergence_event)).is_error()
    pinned = v1_swallows and v2_loud
    failures += 0 if pinned else 1
    lines.append(
        (
            "- PINNED DIVERGENCE (fail-closed on a failure path): "
            if pinned
            else "- DIVERGENT: "
        )
        + "non-str content.text — v1 silently swallows the chunk, v2 errors"
        " loudly naming the shape (re-decide if either half stops holding)"
    )
    for name in sorted(stream._LOUD_CHUNKS):
        event, fragment = stream._LOUD_CHUNKS[name]
        result = stream.parse_event(copy.deepcopy(event))
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                stream._v1_chunks([copy.deepcopy(event)])
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 raises)' if ok else 'DIVERGENT'}: {name}")
    str_tokens_event = {
        "event": "message-end",
        "data": {
            "delta": {
                "finish_reason": "COMPLETE",
                "usage": {"tokens": {"input_tokens": "5", "output_tokens": "3"}},
            }
        },
    }
    result = stream.parse_event(copy.deepcopy(str_tokens_event))
    served = stream._v1_chunks(
        [
            {
                "type": "content-delta",
                "delta": {"message": {"content": {"text": "Hi"}}},
            },
            copy.deepcopy(str_tokens_event),
        ],
        stream_options={"include_usage": True},
    )
    ok = (
        result.is_error()
        and "v1 SERVES" in result.error.summary
        and served[-1]["usage"]["total_tokens"] == 8
    )
    failures += 0 if ok else 1
    lines.append(
        f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: str+str"
        " message-end token counts (v1 concatenates then re-sums in its"
        " include_usage chunk; deliberately left to v1 — the response-side"
        " _int_token decision, critic M1)"
    )
    return failures


def _mistral_rows(lines: list) -> int:
    """wave-2b-beta: mistral (httpx path, bare wire model)."""
    import copy

    from . import test_differential_mistral_request as req
    from . import test_differential_mistral_response as resp
    from . import test_differential_mistral_stream as stream

    failures = 0
    lines += [
        "",
        "## mistral (wave-2b-beta): requests (v1 get_optional_params +"
        " MistralConfig.transform_request — the two-branch message munge —"
        " vs v2 providers/mistral)",
        "",
    ]
    for name in sorted(req.CASES):
        case = req.CASES[name]
        result = req._v2(case)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req.run_v1_request_transform(case)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.V1_RAISES):
        case, fragment = req.V1_RAISES[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                req.run_v1_request_transform(case)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises UnsupportedParamsError)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    for name in sorted(req.V1_SERVES_FALLBACKS):
        case, fragment = req.V1_SERVES_FALLBACKS[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## mistral: responses (v1 transform_response pre-steps + cdr vs v2"
        " mistral pre-steps + the shared openai parser; bare wire model)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        v1 = resp._v1_model_response(raw)
        v2 = resp._v2_model_response(raw)
        same = (
            resp._norm(v2) == resp._norm(v1)
            and v2["model"] == resp.WIRE_MODEL
            and not str(v2["model"]).startswith("mistral/")
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(resp._LOUD):
        raw, fragment = resp._LOUD[name]
        result = resp._v2_parse(raw)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                resp._v1_model_response(copy.deepcopy(raw))
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 raises)' if ok else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## mistral: streams (v1 SSE line replay through"
        " MistralChatResponseIterator + CustomStreamWrapper('mistral') vs v2"
        " mistral pre-step + the httpx_chunk factory (rename +"
        " passthrough thinking_blocks) + the xai chunk dialect)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    return failures


def _watsonx_rows(lines: list) -> int:
    """wave-2b-beta: watsonx (the OpenAILikeChatHandler route; live
    watsonx/{wire} response prefix; generic stream dialect)."""
    import copy

    import pytest

    from . import test_differential_watsonx_request as req
    from . import test_differential_watsonx_response as resp
    from . import test_differential_watsonx_stream as stream

    failures = 0
    lines += [
        "",
        "## watsonx (wave-2b-beta): requests (v1 get_optional_params +"
        " _get_api_params/_prepare_payload + the openai_like body assembly"
        " vs v2 providers/watsonx with deps-borne project/space ids)",
        "",
    ]
    for name in sorted(req.CASES):
        case = req.CASES[name]
        result = req._v2(case)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req.run_v1_request_transform(case)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.V1_RAISES):
        case, fragment = req.V1_RAISES[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                req.run_v1_request_transform(case)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises UnsupportedParamsError)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    for label, case_fn in (
        (
            "top_k (v1 raises the legacy watsonx_text ValueError)",
            lambda: req._v2({"model": req.MODEL, "messages": req._U, "top_k": 5}),
        ),
        (
            "missing project/space ids (v1 raises WatsonXAIError 401)",
            lambda: req._v2(
                req.CASES["plain"], deps=req._deps(project_id=None, space_id=None)
            ),
        ),
        (
            "deployment/ model (envelope routing; v1 serves)",
            lambda: req._v2({"model": "deployment/dep-1", "messages": req._U}),
        ),
    ):
        ok = case_fn().is_error()
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK' if ok else 'DIVERGENT'}: {label}")
    for name in sorted(req.V1_SERVES_FALLBACKS):
        case, fragment = req.V1_SERVES_FALLBACKS[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## watsonx: responses (v1 OpenAILike _transform_response with the"
        " LIVE watsonx/{wire_model} prefix vs v2 watsonx parser + the seam's"
        " openai_like construction arm)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        same = resp._norm(resp._v2_model_response(raw)) == resp._norm(
            resp._v1_model_response(raw)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    f2 = resp._v2_parse({**resp._RESPONSES["text"], "model": 7})
    ok = f2.is_error() and "non-string wire model" in f2.error.summary
    if ok:
        with pytest.raises(Exception):
            resp._v1_model_response({**resp._RESPONSES["text"], "model": 7})
    failures += 0 if ok else 1
    lines.append(
        f"- {'FALLBACK (v1 raises ValidationError)' if ok else 'DIVERGENT'}:"
        " non-string wire model"
    )
    lines += [
        "",
        "## watsonx: streams (v1 line replay through the databricks"
        " ModelResponseIterator + CustomStreamWrapper('watsonx') generic arm"
        " vs v2 watsonx parser + the generic chunk dialect; ids normalized)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    no_finish = [stream._chunk({"role": "assistant", "content": "Hi"})]
    v1 = stream._v1_chunks(no_finish)
    v2 = stream._v2_chunks(no_finish)
    synth_ok = (
        len(v1) == len(v2) + 1
        and stream._norm(v2) == stream._norm(v1[:-1])
        and v1[-1]["choices"][0]["finish_reason"] == "stop"
    )
    failures += 0 if synth_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if synth_ok else "- DIVERGENT: ")
        + "no-wire-finish stream (v2 == v1 minus the wrapper's synthesized"
        " stop tail — the generic streaming seam owns it)"
    )
    for name in sorted(stream._PINNED_DIVERGENCES):
        events, raw_lines, fragment = stream._PINNED_DIVERGENCES[name]
        v1 = stream._v1_chunks(events, raw_lines=raw_lines)
        swallows = all(
            choice["delta"]["content"] in (None, "")
            and not choice["delta"]["tool_calls"]
            for chunk in v1
            for choice in chunk["choices"]
        )
        if raw_lines is not None:
            result = stream.parse_line(raw_lines[0])
        else:
            result = stream.parse_event(copy.deepcopy(events[0]))
        pinned = swallows and result.is_error() and fragment in result.error.summary
        failures += 0 if pinned else 1
        lines.append(
            (
                "- PINNED DIVERGENCE (fail-closed on a failure path): "
                if pinned
                else "- DIVERGENT: "
            )
            + f"{name} — v1's iterator silently swallows it, v2 errors loudly"
        )
    for name in sorted(stream._V1_RAISES_LOUD):
        events, fragment = stream._V1_RAISES_LOUD[name]
        result = stream.parse_event(copy.deepcopy(events[0]))
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                stream._v1_chunks(events)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises MidStreamFallbackError)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    for falsy, label in (("", "empty-string"), ({}, "empty-object")):
        events = [
            stream._chunk({"role": "assistant", "content": "Hi"}),
            stream._chunk({}, finish=falsy),
        ]
        v1 = stream._v1_chunks(events)
        v2 = stream._v2_chunks(events)
        ok = (
            len(v1) == len(v2) + 1
            and stream._norm(v2) == stream._norm(v1[:-1])
            and v1[-1]["choices"][0]["finish_reason"] == "stop"
        )
        failures += 0 if ok else 1
        lines.append(
            ("- SEAM CONTRACT: " if ok else "- DIVERGENT: ")
            + f"falsy ({label}) finish_reason — no finish rides (v1's truthy"
            " gate); v2 == v1 minus the wrapper's synthesized stop tail"
        )
    return failures


def _sagemaker_chat_rows(lines: list) -> int:
    """wave-2b-beta: sagemaker_chat (base GPT config over SigV4 transport;
    sagemaker_nova deliberately unregistered)."""
    import pytest

    from . import test_differential_sagemaker_chat_request as req
    from . import test_differential_sagemaker_chat_response as resp
    from . import test_differential_sagemaker_chat_stream as stream

    failures = 0
    lines += [
        "",
        "## sagemaker_chat (wave-2b-beta): requests (v1 get_optional_params"
        " + the base GPT transform_request vs v2 providers/sagemaker_chat;"
        " SigV4 signs after assembly — envelope)",
        "",
    ]
    for name in sorted(req.CASES):
        case = req.CASES[name]
        result = req._v2(case)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req.run_v1_request_transform(case)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.V1_RAISES):
        case, fragment = req.V1_RAISES[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                req.run_v1_request_transform(case)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises UnsupportedParamsError)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    for name in sorted(req.V1_SERVES_FALLBACKS):
        case, fragment = req.V1_SERVES_FALLBACKS[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## sagemaker_chat: responses (v1 base transform_response/cdr vs the"
        " shared openai parser; bare wire model, no seam preset)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        v1 = resp._v1_model_response(raw)
        v2 = resp._v2_model_response(raw)
        same = resp._norm(v2) == resp._norm(v1) and v2["model"] == resp.WIRE_MODEL
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## sagemaker_chat: streams (v1 AWS event-stream PARSED-event replay"
        " through AWSEventStreamDecoder(is_messages_api) +"
        " CustomStreamWrapper('sagemaker_chat') vs v2 openai parser + the"
        " litellm-validation post-step, 'openai' dialect)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    for name in sorted(stream._V1_RAISES):
        bad, fragment = stream._V1_RAISES[name]
        result = stream.parse_event(dict(bad))
        loud = result.is_error() and fragment in result.error.summary
        if loud:
            with pytest.raises(Exception):
                stream._v1_chunks([bad])
        failures += 0 if loud else 1
        lines.append(
            f"- {'FALLBACK (v1 raises ValidationError)' if loud else 'DIVERGENT'}:"
            f" {name} (loud on both sides)"
        )
    for falsy, label in (("", "empty-string"), ({}, "empty-object")):
        events = [
            stream._chunk({"role": "assistant", "content": "x"}),
            stream._chunk({}, finish=falsy),
        ]
        v1 = stream._v1_chunks(events)
        v2 = stream._v2_chunks(events)
        ok = (
            len(v1) == len(v2) + 1
            and stream._norm(v2) == stream._norm(v1[:-1])
            and v1[-1]["choices"][0]["finish_reason"] == "stop"
        )
        failures += 0 if ok else 1
        lines.append(
            ("- SEAM CONTRACT: " if ok else "- DIVERGENT: ")
            + f"falsy ({label}) finish_reason — no finish rides (v1's truthy"
            " gate); v2 == v1 minus the wrapper's synthesized stop tail"
        )
    return failures


def _groq_rows(lines: list) -> int:
    """wave-2b-beta: groq (httpx path; bare wire model + service_tier
    clamp; the json_schema fork rows)."""
    import copy

    import pytest

    from . import test_differential_groq_request as req
    from . import test_differential_groq_response as resp
    from . import test_differential_groq_stream as stream

    failures = 0
    lines += [
        "",
        "## groq (wave-2b-beta): requests (v1 get_optional_params +"
        " GroqChatConfig.transform_request + hh's extra_body merge vs v2"
        " providers/groq; the json_schema three-way fork)",
        "",
    ]
    for name in sorted(req.CASES):
        case = req.CASES[name]
        result = req._v2(case)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req.run_v1_request_transform(case)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.V1_RAISES):
        case, fragment = req.V1_RAISES[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        if ok:
            try:
                req.run_v1_request_transform(case)
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises UnsupportedParamsError)' if ok else 'DIVERGENT'}:"
            f" {name}"
        )
    schema_tools_case = {
        "model": req.MODEL,
        "messages": req._U,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "s",
                "schema": {"type": "object", "properties": {}},
                "strict": True,
            },
        },
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    bad_request = req._v2(schema_tools_case)
    ok = bad_request.is_error() and "BadRequestError" in bad_request.error.summary
    if ok:
        try:
            req.run_v1_request_transform(schema_tools_case)
            ok = False
        except Exception:
            pass
    failures += 0 if ok else 1
    lines.append(
        f"- {'FALLBACK (v1 raises BadRequestError)' if ok else 'DIVERGENT'}:"
        " response_format json_schema + tools on a non-native model"
    )
    for name in sorted(req.V1_SERVES_FALLBACKS):
        case, fragment = req.V1_SERVES_FALLBACKS[name]
        result = req._v2(case)
        ok = result.is_error() and fragment in result.error.summary
        failures += 0 if ok else 1
        lines.append(f"- {'FALLBACK (v1 serves)' if ok else 'DIVERGENT'}: {name}")
    lines += [
        "",
        "## groq: responses (v1 OpenAILike direct construction + the"
        " service_tier clamp vs v2 groq parser + the seam's openai_like"
        " arm; bare wire model)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        raw = resp._RESPONSES[name]
        v1 = resp._v1_model_response(raw)
        v2 = resp._v2_model_response(raw)
        same = resp._norm(v2) == resp._norm(v1) and v2["model"] == resp.MODEL
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    missing_tier = {
        **{k: v for k, v in resp._BASE.items() if k != "service_tier"},
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    }
    tier_result = resp._v2_parse(missing_tier)
    ok = tier_result.is_error() and "service_tier" in tier_result.error.summary
    if ok:
        with pytest.raises(AttributeError):
            resp._v1_model_response(missing_tier)
    failures += 0 if ok else 1
    lines.append(
        f"- {'FALLBACK (v1 raises AttributeError)' if ok else 'DIVERGENT'}:"
        " response without service_tier (v1's clamp post-step crashes)"
    )
    f2 = resp._v2_parse({**resp._RESPONSES["text_tier_clamped_to_auto"], "model": 7})
    ok = f2.is_error() and "non-string wire model" in f2.error.summary
    failures += 0 if ok else 1
    lines.append(
        f"- {'FALLBACK (v1 raises ValidationError)' if ok else 'DIVERGENT'}:"
        " non-string wire model"
    )
    lines += [
        "",
        "## groq: streams (v1 SSE line replay through"
        " GroqChatCompletionStreamingHandler + CustomStreamWrapper('groq')"
        " vs v2 httpx_chunk factory (rename) + the xai chunk dialect)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        events = stream.STREAMS[name]
        same = stream._norm(stream._v2_chunks(events)) == stream._norm(
            stream._v1_chunks(events)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    v1 = stream._v1_chunks(stream.USAGE_STREAM, stream_options={"include_usage": True})
    v2 = stream._v2_chunks(stream.USAGE_STREAM)
    tail_ok = (
        len(v1) == len(v2)
        and stream._norm(v2[:-1]) == stream._norm(v1[: len(v2) - 1])
        and v2[-1]["choices"] == []
        and all(
            v1[-1]["usage"][k] == v2[-1]["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    )
    failures += 0 if tail_ok else 1
    lines.append(
        ("- SEAM CONTRACT: " if tail_ok else "- DIVERGENT: ")
        + "usage tail (v2 passes the wire choices=[] usage chunk through;"
        " the streaming seam owns v1's synthesized final chunk)"
    )
    err = {"error": {"message": "boom", "code": 500}}
    loud = stream.parse_event(dict(err)).is_error()
    failures += 0 if loud else 1
    lines.append(
        f"- {'FALLBACK (v1 raises MidStreamFallbackError)' if loud else 'DIVERGENT'}:"
        " error chunk (loud on both sides — the truthy-value check)"
    )
    for key in ("reasoning", "reasoning_content"):
        bad = stream._chunk({"role": "assistant", key: 5})
        result = stream.parse_event(copy.deepcopy(bad))
        ok = result.is_error() and "is not a string" in result.error.summary
        if ok:
            try:
                stream._v1_chunks([bad, stream._chunk({}, finish="stop")])
                ok = False
            except Exception:
                pass
        failures += 0 if ok else 1
        lines.append(
            f"- {'FALLBACK (v1 raises APIError)' if ok else 'DIVERGENT'}:"
            f" non-str delta {key} (the F6 groq-local pre-step; the wrapper"
            " epilogue join TypeErrors in v1)"
        )
    refusal_events = [
        stream._chunk({"role": "assistant", "refusal": 7, "content": "x"}),
        stream._chunk({}, finish="stop"),
    ]
    v1 = stream._v1_chunks(refusal_events)
    v2 = stream._v2_chunks(refusal_events)
    handoff = (
        v1[0]["choices"][0]["delta"]["refusal"] == 7
        and v2[0]["choices"][0]["delta"]["refusal"] is None
    )
    failures += 0 if handoff else 1
    lines.append(
        (
            "- INTEGRATOR-FLIP HANDOFF (current behavior guarded): "
            if handoff
            else "- DIVERGENT: "
        )
        + "non-str refusal — v1 forwards 7, the SHARED httpx_chunk factory"
        " nulls it; the fix belongs to the alpha fix round's concurrent"
        " httpx_chunk edit (verifier-wave2b-alpha F1) — the sibling-merge"
        " integrator flips this row and the gate test to v1 parity"
    )
    return failures


def _azure_rows(lines: list) -> int:
    import os

    from litellm.translation import translate_chat_request

    from . import _azure_corpus as corpus
    from . import test_differential_azure_request as req
    from . import test_differential_azure_response as resp
    from . import test_differential_azure_stream as stream
    from .conftest import build_real_deps

    os.environ.pop("AZURE_API_VERSION", None)  # the corpus seam's default chain
    failures = 0
    lines += [
        "",
        "## azure: request bodies (v1 api-version-aware map_openai_params + transform_request vs v2)",
        "",
    ]
    for name in sorted(req.CORPUS):
        request, api_version, base_model = req._row(name)
        result = req._v2_body(request, api_version, base_model)
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req._v1_body(request, api_version, base_model)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.EXPECTED_FALLBACKS):
        case, api_version, base_model, reason = req.EXPECTED_FALLBACKS[name]
        result = req._v2_body(case, api_version, base_model)
        ok = result.is_error() and reason in result.error.summary
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## azure: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)",
        "",
    ]
    for case_id in sorted(req.CHAR_CASES):
        case = req.CHAR_CASES[case_id]
        if "azure" in case["skip"]:
            lines.append(f"- SKIPPED (corpus): {case_id} ({case['skip']['azure']})")
            continue
        snapshot_path = corpus.SNAPSHOTS_DIR / "requests" / "azure" / f"{case_id}.json"
        v1_same = (
            corpus.canonical_json(corpus.run_v1_request_transform(case))
            == snapshot_path.read_text()
        )
        raw = {
            "model": corpus.MODEL,
            "messages": [dict(m) for m in case["messages"]],
            **dict(case["params"]),
        }
        result = translate_chat_request(
            raw, "azure", build_real_deps(api_version=req.DEFAULT_API_VERSION)
        )
        if case_id in req.CHAR_EXPECTED_FALLBACKS:
            ok = result.is_error() and v1_same
            failures += 0 if ok else 1
            label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
            lines.append(
                f"- {label}: {case_id} ({req.CHAR_EXPECTED_FALLBACKS[case_id]})"
            )
            continue
        expected = corpus.v2_comparable(corpus.load_json(snapshot_path))
        v2_same = result.is_ok() and corpus.canonical_json(
            result.ok
        ) == corpus.canonical_json(expected)
        same = v1_same and v2_same
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {case_id}")
    lines += [
        "",
        "## azure: responses (v1 convert_to_model_response_object with azure.py's args vs v2)",
        "",
    ]
    for name in sorted(resp._RESPONSES):
        same = resp._norm(resp._v2_model_response(resp._RESPONSES[name])) == resp._norm(
            resp._v1_model_response(resp._RESPONSES[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for fixture_id in resp._FIXTURES:
        raw = corpus.load_json(
            corpus.FIXTURES_DIR / "responses" / "azure" / f"{fixture_id}.json"
        )
        snapshot = (
            corpus.SNAPSHOTS_DIR / "responses" / "azure" / f"{fixture_id}.json"
        ).read_text()
        same = (
            corpus.canonical_json(corpus.run_v1_response_transform(raw)) == snapshot
            and corpus.canonical_json(resp._v2_model_response(raw)) == snapshot
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: corpus {fixture_id}")
    same = resp._norm(
        resp._v2_azure_ai_model_response(resp._AZURE_AI_RESPONSE)
    ) == resp._norm(resp._v1_azure_ai_model_response(resp._AZURE_AI_RESPONSE))
    failures += 0 if same else 1
    lines.append(
        f"- {'IDENTICAL' if same else 'DIVERGENT'}: azure_ai model rename (v1 preset + convert re-prefix)"
    )
    lines += [
        "",
        "## azure: streams (v1 CustomStreamWrapper('azure') over SDK chunks vs v2 azure dialect)",
        "",
    ]
    for name in sorted(stream.STREAMS):
        same = stream._norm(stream._v2_chunks(stream.STREAMS[name])) == stream._norm(
            corpus.replay_azure_sdk_chunks(stream.STREAMS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for fixture_id in stream._FIXTURES:
        events = corpus.load_json(
            corpus.FIXTURES_DIR / "streams" / "azure" / f"{fixture_id}.json"
        )
        expected = corpus.canonical_json(
            corpus.stream_snapshot_chunks(
                corpus.SNAPSHOTS_DIR / "streams" / "azure" / f"{fixture_id}.json"
            )
        )
        same = (
            corpus.canonical_json(corpus.replay_azure_sdk_chunks(events)) == expected
            and corpus.canonical_json(stream._v2_chunks(events)) == expected
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: corpus {fixture_id}")
    return failures


def _azure_ai_rows(lines: list) -> int:
    from . import test_differential_azure_ai_request as req

    failures = 0
    lines += [
        "",
        "## azure_ai: request bodies (v1 AzureAIStudioConfig chain vs v2)",
        "",
    ]
    for name in sorted(req.FOUNDRY_CORPUS):
        result = req._v2_body(req.FOUNDRY_CORPUS[name], "azure_ai")
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req._v1_foundry_body(req.FOUNDRY_CORPUS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.FOUNDRY_EXPECTED_FALLBACKS):
        case, reason = req.FOUNDRY_EXPECTED_FALLBACKS[name]
        result = req._v2_body(case, "azure_ai")
        ok = result.is_error() and reason in result.error.summary
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    lines += [
        "",
        "## azure_ai_anthropic: request bodies (v1 AzureAnthropicConfig chain vs v2, no model spoof)",
        "",
    ]
    for name in sorted(req.CLAUDE_CORPUS):
        result = req._v2_body(req.CLAUDE_CORPUS[name], "azure_ai_anthropic")
        same = result.is_ok() and req._norm(result.ok) == req._norm(
            req._v1_claude_body(req.CLAUDE_CORPUS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    for name in sorted(req.CLAUDE_EXPECTED_FALLBACKS):
        case, reason = req.CLAUDE_EXPECTED_FALLBACKS[name]
        result = req._v2_body(case, "azure_ai_anthropic")
        ok = result.is_error() and reason in result.error.summary
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(f"- {label}: {name} ({reason})")
    return failures


def _bedrock_request_rows(lines: list) -> int:
    from litellm.translation import translate_chat_request

    from . import _bedrock_corpus as corpus
    from . import test_differential_bedrock_request as req
    from .conftest import build_real_deps

    failures = 0
    cases = corpus.cases()
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: request bodies "
            "(characterization snapshot == v1-at-HEAD == v2, canonical JSON)",
            "",
        ]
        for case_id in sorted(cases):
            case = cases[case_id]
            if provider_key in case["skip"]:
                lines.append(
                    f"- SKIPPED (corpus): {case_id} ({case['skip'][provider_key]})"
                )
                continue
            snapshot = (
                corpus.SNAPSHOTS_DIR / "requests" / provider_key / f"{case_id}.json"
            ).read_text()
            v1_same = (
                corpus.canonical_json(
                    corpus.run_v1_request_transform(provider_key, case)
                )
                == snapshot
            )
            result = translate_chat_request(
                req._v2_raw(provider_key, case), provider_key, build_real_deps()
            )
            if case_id in req.EXPECTED_FALLBACKS:
                ok = result.is_error() and v1_same
                failures += 0 if ok else 1
                label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
                lines.append(
                    f"- {label}: {case_id} ({req.EXPECTED_FALLBACKS[case_id]})"
                )
                continue
            v2_same = result.is_ok() and corpus.canonical_json(result.ok) == snapshot
            same = v1_same and v2_same
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {case_id}")
        quirk_sets = [("quirk", req.QUIRKS, corpus.PROVIDERS[provider_key])]
        if provider_key == "bedrock_converse":
            quirk_sets.append(
                ("quirk", req.QUIRKS_CONVERSE_ONLY, corpus.PROVIDERS[provider_key])
            )
        for label, quirks, model_alias in quirk_sets:
            for name in sorted(quirks):
                case = quirks[name]
                v1 = corpus.run_v1_request_transform_for_model(model_alias, case)
                result = translate_chat_request(
                    req._v2_raw(provider_key, case), provider_key, build_real_deps()
                )
                same = result.is_ok() and corpus.canonical_json(
                    result.ok
                ) == corpus.canonical_json(v1)
                failures += 0 if same else 1
                lines.append(
                    f"- {'IDENTICAL' if same else 'DIVERGENT'}: {label} {name} (v1 in-process)"
                )
    return failures


def _bedrock_response_rows(lines: list) -> int:
    from . import _bedrock_corpus as corpus
    from . import test_differential_bedrock_response as resp

    failures = 0
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: responses (snapshot == v1 transform_response == v2)",
            "",
        ]
        for fixture_id in resp._fixture_ids(provider_key):
            payload = corpus.load_json(
                corpus.FIXTURES_DIR / "responses" / provider_key / f"{fixture_id}.json"
            )
            v1 = corpus.run_v1_response_transform(
                provider_key, dict(payload), [dict(m) for m in resp._MESSAGES]
            ).model_dump()
            v2 = resp._v2_model_response(provider_key, payload)
            snapshot = corpus.load_json(
                corpus.SNAPSHOTS_DIR / "responses" / provider_key / f"{fixture_id}.json"
            )
            same = resp._norm(v2) == resp._norm(v1) == resp._norm(snapshot)
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {fixture_id}")
    return failures


def _bedrock_stream_rows(lines: list) -> int:
    from . import _bedrock_corpus as corpus
    from . import test_differential_bedrock_stream as stream

    failures = 0
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: streams (snapshot == real decoder replay == v2 fold, parsed-event seam)",
            "",
        ]
        replay = (
            corpus.replay_v1_converse_events
            if provider_key == "bedrock_converse"
            else corpus.replay_v1_invoke_events
        )
        for fixture_id in stream._fixture_ids(provider_key):
            events = corpus.load_json(
                corpus.FIXTURES_DIR / "streams" / provider_key / f"{fixture_id}.json"
            )
            v1 = replay([dict(e) for e in events])
            v2 = stream._v2_chunks(provider_key, events)
            snapshot = corpus.load_json(
                corpus.SNAPSHOTS_DIR / "streams" / provider_key / f"{fixture_id}.json"
            )
            same = stream._norm(v2) == stream._norm(v1) == stream._norm(snapshot)
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {fixture_id}")
    return failures


def _stub_vertex_token() -> None:
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    VertexBase.get_access_token = (  # type: ignore[method-assign]
        lambda self, credentials, project_id: (
            "char-vertex-token",
            project_id or "char-test-project",
        )
    )


def _google_request_rows(lines: list) -> int:
    from litellm.translation import translate_chat_request
    from litellm.translation_seam_google import build_google_deps

    from . import _google_corpus as corpus
    from . import test_differential_google_request as req

    failures = 0
    cases = corpus.cases()
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: request bodies "
            "(characterization snapshot == v1-at-HEAD == v2, canonical JSON)",
            "",
        ]
        deps = build_google_deps(corpus.V2_PROVIDERS[provider_key])
        for case_id in sorted(cases):
            case = cases[case_id]
            if provider_key in case["skip"]:
                lines.append(
                    f"- SKIPPED (corpus): {case_id} ({case['skip'][provider_key]})"
                )
                continue
            snapshot = (
                corpus.SNAPSHOTS_DIR / "requests" / provider_key / f"{case_id}.json"
            ).read_text()
            v1_same = (
                corpus.canonical_json(
                    corpus.run_v1_request_transform(provider_key, case)
                )
                == snapshot
            )
            result = translate_chat_request(
                req._v2_raw(provider_key, case),
                corpus.V2_PROVIDERS[provider_key],  # type: ignore[arg-type]
                deps,
            )
            if case_id in req.EXPECTED_FALLBACKS:
                ok = result.is_error() and v1_same
                failures += 0 if ok else 1
                label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
                lines.append(
                    f"- {label}: {case_id} ({req.EXPECTED_FALLBACKS[case_id]})"
                )
                continue
            v2_same = result.is_ok() and corpus.canonical_json(result.ok) == snapshot
            same = v1_same and v2_same
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {case_id}")
    lines += ["", "## google quirk corpus (v1 in-process reference)", ""]
    import copy as _copy

    for name in sorted(req.QUIRKS):
        alias, case, drop_params = req.QUIRKS[name]
        v1 = corpus.run_v1_request_transform_for_model(
            alias, _copy.deepcopy(case), drop_params=drop_params
        )
        model, custom_llm_provider, _ = corpus.resolve_model(alias)
        provider_key = {"vertex_ai": "vertex_gemini", "gemini": "gemini"}[
            custom_llm_provider
        ]
        raw = {
            "model": model,
            "messages": _copy.deepcopy(case["messages"]),
            **_copy.deepcopy(case["params"]),
        }
        result = req._v2_translate(provider_key, raw, drop_params=drop_params)
        same = result.is_ok() and corpus.canonical_json(
            result.ok
        ) == corpus.canonical_json(v1)
        failures += 0 if same else 1
        lines.append(
            f"- {'IDENTICAL' if same else 'DIVERGENT'}: quirk {name} ({alias})"
        )
    for name in sorted(req.CACHE_GATE_FALLBACKS):
        raw = _copy.deepcopy(req.CACHE_GATE_FALLBACKS[name])
        result = req._v2_translate("vertex_gemini", raw)
        ok = result.is_error()
        failures += 0 if ok else 1
        label = "FALLBACK (v1 serves it)" if ok else "DIVERGENT"
        lines.append(
            f"- {label}: cache-marker token bound {name}"
            " (v1's check_and_create_cache may create the context cache;"
            " the byte+margin bound fails closed)"
        )
    return failures


def _google_response_rows(lines: list) -> int:
    from . import _google_corpus as corpus
    from . import test_differential_google_response as resp

    failures = 0
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: responses (snapshot == v1 transform_response == v2)",
            "",
        ]
        for fixture_id in resp._fixture_ids(provider_key):
            payload = corpus.load_json(
                corpus.FIXTURES_DIR / "responses" / provider_key / f"{fixture_id}.json"
            )
            v1 = corpus.run_v1_response_transform(
                provider_key, dict(payload), [dict(m) for m in resp._MESSAGES]
            ).model_dump()
            v2 = resp._v2_model_response(provider_key, payload)
            snapshot = corpus.load_json(
                corpus.SNAPSHOTS_DIR / "responses" / provider_key / f"{fixture_id}.json"
            )
            same = resp._norm(v2) == resp._norm(v1) == resp._norm(snapshot)
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {fixture_id}")
    return failures


def _google_stream_rows(lines: list) -> int:
    from . import _google_corpus as corpus
    from . import test_differential_google_stream as stream

    failures = 0
    for provider_key in sorted(corpus.PROVIDERS):
        lines += [
            "",
            f"## {provider_key}: streams (snapshot == real decoder replay == v2 fold)",
            "",
        ]
        for fixture_id in stream._fixture_ids(provider_key):
            lines_raw = stream._read_lines(provider_key, fixture_id)
            if provider_key == "vertex_anthropic":
                v1 = corpus.replay_v1_vertex_anthropic_sse(list(lines_raw))
                v2 = stream._v2_vertex_anthropic_chunks(lines_raw)
                normalize = True
            else:
                v1 = corpus.replay_v1_gemini_sse(provider_key, list(lines_raw))
                v2 = stream._v2_gemini_chunks(provider_key, lines_raw)
                normalize = False
            snapshot = corpus.load_json(
                corpus.SNAPSHOTS_DIR / "streams" / provider_key / f"{fixture_id}.json"
            )
            same = (
                stream._norm(v2, normalize)
                == stream._norm(v1, normalize)
                == stream._norm(snapshot, normalize)
            )
            failures += 0 if same else 1
            lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {fixture_id}")
    return failures


def main() -> None:
    _freeze_ambient()
    _stub_vertex_token()

    lines = [
        "# Translation v2 differential report (anthropic + bedrock + openai + google + azure + xai + the compat_sdk family (waves 1a+1b+2a) + the wave-1b compat_httpx family + the wave-2b-beta own modules)",
        "",
        "v1 and v2 run over the same corpus; every row must be IDENTICAL (or an",
        "explained FALLBACK that v1 serves) for a provider's flag to turn on.",
        "Bedrock and google rows additionally pin the characterization-corpus",
        "snapshot, so each row proves snapshot == v1-at-HEAD == v2. Regenerate with:",
        "`python -m tests.test_litellm.translation.generate_differential_report`",
        "",
        f"- commit: {_git_sha()}",
        "",
    ]
    failures = _anthropic_rows(lines)
    failures += _openai_rows(lines)
    failures += _xai_rows(lines)
    failures += _compat_sdk_rows(lines)
    failures += _compat_httpx_rows(lines)
    failures += _cometapi_rows(lines)
    failures += _cohere_rows(lines)
    failures += _mistral_rows(lines)
    failures += _watsonx_rows(lines)
    failures += _sagemaker_chat_rows(lines)
    failures += _groq_rows(lines)
    failures += _azure_rows(lines)
    failures += _azure_ai_rows(lines)
    failures += _bedrock_request_rows(lines)
    failures += _bedrock_response_rows(lines)
    failures += _bedrock_stream_rows(lines)
    failures += _google_request_rows(lines)
    failures += _google_response_rows(lines)
    failures += _google_stream_rows(lines)
    lines += [
        "",
        f"Result: {failures} divergent rows."
        " Shapes outside the corpus fall back to v1 (fail-closed), so this"
        " table is the complete flag-on surface.",
        "",
    ]
    (_HERE / "DIFFERENTIAL_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    sys.exit(1 if failures else 0)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
