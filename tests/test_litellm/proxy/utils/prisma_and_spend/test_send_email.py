"""Pin ``send_email``.

Symbols pinned here:
  - ``send_email``
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm.proxy.utils import send_email


@pytest.fixture(autouse=True)
def _smtp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "u")
    monkeypatch.setenv("SMTP_PASSWORD", "p")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "from@invalid")
    monkeypatch.setenv("SMTP_TLS", "True")
    monkeypatch.setenv("SMTP_USE_SSL", "False")


@pytest.mark.asyncio
async def test_send_email_dispatches_via_smtp(in_memory_smtp: Any) -> None:
    await send_email(
        receiver_email="to@invalid",
        subject="Hello",
        html="<p>body</p>",
    )
    assert len(in_memory_smtp.sent) == 1
    sent = in_memory_smtp.sent[0]
    pinned = {
        "from_addr": sent.from_addr,
        "to_addrs": sent.to_addrs,
        "subject": sent.subject,
        "starttls": sent.starttls_called,
        "login": sent.login_args,
    }
    assert pinned == {
        "from_addr": "from@invalid",
        "to_addrs": "to@invalid",
        "subject": "Hello",
        "starttls": True,
        "login": ("u", "p"),
    }
    assert "<p>body</p>" in sent.body


@pytest.mark.asyncio
async def test_send_email_starttls_uses_ssl(
    in_memory_smtp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_USE_SSL", "True")
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.sent[0].starttls_called is False


@pytest.mark.asyncio
async def test_send_email_skips_starttls_when_tls_disabled(
    in_memory_smtp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_TLS", "False")
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.sent[0].starttls_called is False


@pytest.mark.asyncio
async def test_send_email_error_missing_sender_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMTP_SENDER_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SMTP_SENDER_EMAIL"):
        await send_email(
            receiver_email="x@y", subject="s", html="<p>h</p>"
        )


@pytest.mark.asyncio
async def test_send_email_error_missing_receiver() -> None:
    with pytest.raises(ValueError, match="receiver email"):
        await send_email(receiver_email=None, subject="s", html="<p>h</p>")


@pytest.mark.asyncio
async def test_send_email_error_missing_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        await send_email(receiver_email="x@y", subject=None, html="<p>h</p>")


@pytest.mark.asyncio
async def test_send_email_error_missing_html() -> None:
    with pytest.raises(ValueError, match="HTML"):
        await send_email(receiver_email="x@y", subject="s", html=None)


@pytest.mark.asyncio
async def test_send_email_smtp_failure_is_swallowed(
    in_memory_smtp: Any,
) -> None:
    """SMTP send_message errors are caught and logged; ``send_email`` itself
    does not raise so a failing email never blocks the proxy.
    """
    in_memory_smtp.raise_on_send = RuntimeError("smtp boom")
    await send_email(
        receiver_email="to@invalid", subject="Hi", html="<p>x</p>"
    )
    assert in_memory_smtp.sent == []
