import datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pypdf import PdfReader

from litellm.integrations.email_alerting import (
    _fetch_custom_logo_bytes,
    build_cost_savings_report_email,
    build_cost_savings_report_pdf,
    send_cost_savings_report_email,
)
from litellm.proxy.spend_tracking.cost_savings_report import CostSavingsReport
from litellm.proxy.utils import EmailAttachment


def _tiny_png_bytes(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (40, 10), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _report(**overrides) -> CostSavingsReport:
    defaults = dict(
        start_date=datetime.date(2026, 1, 1),
        end_date=datetime.date(2026, 1, 8),
        total_spend=100.0,
        autorouter_savings_spend=10.5,
        compression_savings_spend=2.25,
        prompt_caching_savings_spend=1.25,
    )
    defaults.update(overrides)
    return CostSavingsReport(**defaults)


def test_build_cost_savings_report_email_contains_each_savings_driver_and_total():
    subject, html = build_cost_savings_report_email(_report())

    assert "Jan 01, 2026" in subject and "Jan 08, 2026" in subject
    assert "$100.0" in html
    assert "$10.5" in html
    assert "$2.25" in html
    assert "$1.25" in html
    assert "$14.0" in html  # total_savings = 10.5 + 2.25 + 1.25


def test_build_cost_savings_report_email_reflects_savings_changes():
    _, html_before = build_cost_savings_report_email(_report(autorouter_savings_spend=10.5))
    _, html_after = build_cost_savings_report_email(_report(autorouter_savings_spend=20.5))

    assert html_before != html_after
    assert "$20.5" in html_after
    assert "$10.5" not in html_after


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages)


def test_build_cost_savings_report_pdf_contains_each_savings_driver_and_total():
    pdf_bytes = build_cost_savings_report_pdf(_report())

    assert pdf_bytes.startswith(b"%PDF-")
    text = _pdf_text(pdf_bytes)
    assert "100.0000" in text
    assert "10.5000" in text
    assert "2.2500" in text
    assert "1.2500" in text
    assert "14.0000" in text  # total_savings = 10.5 + 2.25 + 1.25


def test_build_cost_savings_report_pdf_reflects_savings_changes():
    text_before = _pdf_text(build_cost_savings_report_pdf(_report(autorouter_savings_spend=10.5)))
    text_after = _pdf_text(build_cost_savings_report_pdf(_report(autorouter_savings_spend=20.5)))

    assert "20.5000" in text_after
    assert "20.5000" not in text_before


def test_build_cost_savings_report_pdf_embeds_custom_logo_when_given():
    default_pdf = build_cost_savings_report_pdf(_report())
    custom_logo_pdf = build_cost_savings_report_pdf(_report(), logo_bytes=_tiny_png_bytes((255, 0, 0)))

    assert custom_logo_pdf.startswith(b"%PDF-")
    assert custom_logo_pdf != default_pdf


def test_build_cost_savings_report_pdf_preserves_custom_logo_aspect_ratio():
    # a very wide, short logo should not be squeezed into a square box
    wide_logo_pdf = build_cost_savings_report_pdf(_report(), logo_bytes=_tiny_png_bytes((0, 255, 0)))
    assert wide_logo_pdf.startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_fetch_custom_logo_bytes_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("SMTP_SENDER_LOGO", raising=False)
    monkeypatch.delenv("EMAIL_LOGO_URL", raising=False)

    assert await _fetch_custom_logo_bytes() is None


@pytest.mark.asyncio
async def test_fetch_custom_logo_bytes_returns_content_on_success(monkeypatch):
    monkeypatch.setenv("EMAIL_LOGO_URL", "https://example.com/logo.png")
    logo_bytes = _tiny_png_bytes((0, 0, 255))

    mock_response = MagicMock()
    mock_response.content = logo_bytes
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await _fetch_custom_logo_bytes()

    assert result == logo_bytes
    mock_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_custom_logo_bytes_falls_back_to_none_on_failure(monkeypatch):
    monkeypatch.setenv("EMAIL_LOGO_URL", "https://example.com/logo.png")

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=RuntimeError("network down"))

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await _fetch_custom_logo_bytes()

    assert result is None


@pytest.mark.asyncio
async def test_send_cost_savings_report_email_sends_built_content_to_recipients():
    report = _report()
    expected_subject, expected_html = build_cost_savings_report_email(report)

    with patch("litellm.proxy.utils.send_email", new=AsyncMock()) as mock_send_email:
        sent = await send_cost_savings_report_email(
            recipient_emails=["a@example.com", "b@example.com"],
            report=report,
        )

    assert sent is True
    mock_send_email.assert_awaited_once()
    _, kwargs = mock_send_email.await_args
    assert kwargs["receiver_email"] == "a@example.com,b@example.com"
    assert kwargs["subject"] == expected_subject
    assert kwargs["html"] == expected_html
    assert len(kwargs["attachments"]) == 1
    attachment: EmailAttachment = kwargs["attachments"][0]
    assert attachment.filename == f"litellm_cost_savings_report_{report.start_date}_{report.end_date}.pdf"
    assert attachment.mime_type == "application/pdf"
    assert attachment.content.startswith(b"%PDF-")
    assert "14.0000" in _pdf_text(attachment.content)  # total_savings


@pytest.mark.asyncio
async def test_send_cost_savings_report_email_noop_when_no_recipients():
    with patch("litellm.proxy.utils.send_email", new=AsyncMock()) as mock_send_email:
        sent = await send_cost_savings_report_email(recipient_emails=[], report=_report())

    assert sent is False
    mock_send_email.assert_not_awaited()
