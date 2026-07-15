"""pdf_input x Vertex AI.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to GCP Vertex AI, write a tiny valid PDF to disk, allow
the built-in `Read` tool, and ask Claude to read the PDF and report
what it contains.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/pdf_input/test_vertex_ai.py
                       ^^^^^^^^^      ^^^^^^^^^
                       feature_id     provider
"""

from __future__ import annotations

import pytest

from claude_code._env import require_compat_cli_credentials
from claude_code.conftest import _compat_cli_key_provider
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]

PDF_MARKER = "PONG"


def _build_minimal_pdf(marker: str) -> bytes:
    """Return a single-page PDF whose only visible text is `marker`."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        (
            b"<< /Length %d >>\nstream\nBT /F1 24 Tf 50 100 Td ("
            + marker.encode("ascii")
            + b") Tj ET\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body_open = objects[3].index(b"\nstream\n") + len(b"\nstream\n")
    body_close = objects[3].index(b"\nendstream")
    body_len = body_close - body_open
    objects[3] = (
        b"<< /Length "
        + str(body_len).encode("ascii")
        + b" >>\nstream\n"
        + objects[3][body_open:body_close]
        + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    return bytes(out)


def test_pdf_input_vertex_ai(compat_result, tmp_path):
    base_url, api_key = require_compat_cli_credentials(
        compat_result, cli_key_provider=_compat_cli_key_provider
    )

    pdf_path = tmp_path / "marker.pdf"
    pdf_path.write_bytes(_build_minimal_pdf(PDF_MARKER))

    outcomes = run_claude_models_parallel(
        models=VERTEX_AI_MODELS,
        prompt=(
            f"Use the Read tool to read the file at {pdf_path}. "
            "Report the single word that appears in the document."
        ),
        base_url=base_url,
        api_key=api_key,
        extra_args=["--allowed-tools", "Read"],
    )

    failures = []
    for model in VERTEX_AI_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if PDF_MARKER not in outcome.text.upper():
            error = (
                f"[{model}] reply did not reference the PDF marker {PDF_MARKER!r}; "
                f"got: {outcome.text.strip()!r}"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
