"""pdf_input x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, write a tiny valid PDF to disk, allow the built-in `Read`
tool, and ask Claude to read the PDF and report what it contains.

The Read tool inlines the PDF bytes as `document` content blocks on the
next assistant turn, which is exactly the gateway path we want to
exercise: it's distinct from image content blocks (which are tested in
`vision/`) and uses a different transformation in LiteLLM's Anthropic
provider. We assert the upstream produces a non-empty reply that
references the contents of the PDF — proving the proxy preserved the
document content block end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/pdf_input/test_anthropic.py
                       ^^^^^^^^^      ^^^^^^^^^
                       feature_id     provider
"""

from __future__ import annotations

import os

import pytest

from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Smallest valid PDF that renders a single visible word ("PONG"). Built
# inline rather than checked in as a binary fixture so the test stays
# self-contained and the marker word is easy to grep for in CI logs.
# The structure is a hand-crafted single-page PDF with one Helvetica
# text show; offsets are computed at write time so the xref table
# stays consistent regardless of platform line endings.
PDF_MARKER = "PONG"


def _build_minimal_pdf(marker: str) -> bytes:
    """Return a single-page PDF whose only visible text is `marker`.

    We construct the PDF imperatively because `pypdf`/`reportlab` are
    not in the test deps and we want the cell to work in a clean
    environment. The xref offsets are recomputed for each `marker`
    length so the file stays well-formed.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        # Page content stream: position the text and show the marker.
        # `BT ... ET` is a text object; `Tf` selects font, `Td` moves
        # the cursor, `Tj` paints a string.
        (
            b"<< /Length %d >>\nstream\nBT /F1 24 Tf 50 100 Td ("
            + marker.encode("ascii")
            + b") Tj ET\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    # Fix up the /Length on the content stream to match its body.
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


@pytest.mark.covers("llm.messages.anthropic.pdf.nonstream.works")
def test_pdf_input_anthropic(compat_result, tmp_path):
    """Drive the `claude` CLI against the LiteLLM proxy with a PDF
    attached via the Read tool and assert the reply references it."""
    base_url = os.environ.get(PROXY_BASE_URL_ENV)
    api_key = os.environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured", pytrace=False
        )

    pdf_path = tmp_path / "marker.pdf"
    pdf_path.write_bytes(_build_minimal_pdf(PDF_MARKER))

    outcomes = run_claude_models_parallel(
        models=ANTHROPIC_MODELS,
        prompt=(
            f"Use the Read tool to read the file at {pdf_path}. "
            "Report the single word that appears in the document."
        ),
        base_url=base_url,
        api_key=api_key,
        extra_args=["--allowed-tools", "Read"],
    )

    failures = []
    for model in ANTHROPIC_MODELS:
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

        # The strongest gateway-level signal we can assert without
        # parsing every event type: the model's final user-visible
        # reply names the marker word that only the PDF carries.
        # If the proxy dropped the `document` content block, the
        # model has no way to produce this token.
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
