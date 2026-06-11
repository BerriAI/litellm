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


def main() -> None:
    from . import test_differential_anthropic_request as req
    from . import test_differential_anthropic_response as resp
    from . import test_differential_anthropic_stream as stream

    counter = iter(range(1, 1_000_000))
    uuid.uuid4 = lambda: uuid.UUID(int=next(counter))  # type: ignore[assignment]
    time.time = lambda: resp.FROZEN_TIME  # type: ignore[assignment]

    lines = [
        "# Translation v2 differential report (anthropic)",
        "",
        "v1 and v2 run over the same corpus; every row must be IDENTICAL for",
        "the anthropic flag to turn on. Regenerate with:",
        "`python -m tests.test_litellm.translation.generate_differential_report`",
        "",
        f"- commit: {_git_sha()}",
        "",
        "## Request bodies (v1 map_openai_params + transform_request vs v2)",
        "",
    ]
    failures = 0
    for name in sorted(req.CORPUS):
        same = req._norm(req._v2_body(req.CORPUS[name])) == req._norm(
            req._v1_body(req.CORPUS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += ["", "## Responses (v1 transform_response vs v2)", ""]
    for name in sorted(resp._REQUESTS):
        same = resp._norm(resp._v2_model_response(name)) == resp._norm(
            resp._v1_model_response(name)
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
    lines += ["", "## Streams (v1 CustomStreamWrapper replay vs v2 engine/stream)", ""]
    for name in sorted(stream.STREAMS):
        same = stream._norm(stream._v2_chunks(stream.STREAMS[name])) == stream._norm(
            stream._v1_chunks(stream.STREAMS[name])
        )
        failures += 0 if same else 1
        lines.append(f"- {'IDENTICAL' if same else 'DIVERGENT'}: {name}")
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
