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


def main() -> None:
    _freeze_ambient()

    lines = [
        "# Translation v2 differential report (anthropic + bedrock + openai)",
        "",
        "v1 and v2 run over the same corpus; every row must be IDENTICAL (or an",
        "explained FALLBACK that v1 serves) for a provider's flag to turn on.",
        "Bedrock rows additionally pin the characterization-corpus snapshot, so",
        "each row proves snapshot == v1-at-HEAD == v2. Regenerate with:",
        "`python -m tests.test_litellm.translation.generate_differential_report`",
        "",
        f"- commit: {_git_sha()}",
        "",
    ]
    failures = _anthropic_rows(lines)
    failures += _openai_rows(lines)
    failures += _bedrock_request_rows(lines)
    failures += _bedrock_response_rows(lines)
    failures += _bedrock_stream_rows(lines)
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
