"""Pin ``send_email``.

Symbols pinned here:
  - ``send_email``
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock

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
    delivered = await send_email(
        receiver_email="to@invalid",
        subject="Hello",
        html="<p>body</p>",
    )
    assert delivered is True
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
async def test_send_email_starttls_uses_ssl(in_memory_smtp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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
        await send_email(receiver_email="x@y", subject="s", html="<p>h</p>")


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
async def test_send_email_sets_connection_timeout(in_memory_smtp: Any) -> None:
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.connection_kwargs[0].get("timeout") == 30.0


@pytest.mark.asyncio
async def test_send_email_timeout_env_override(in_memory_smtp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_TIMEOUT", "5")
    monkeypatch.setenv("SMTP_USE_SSL", "True")
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.connection_kwargs[0].get("timeout") == 5.0


@pytest.mark.asyncio
async def test_send_email_malformed_timeout_is_swallowed(in_memory_smtp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_TIMEOUT", "30s")
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.sent == []


@pytest.mark.asyncio
async def test_send_email_runs_off_event_loop_thread(in_memory_smtp: Any) -> None:
    await send_email(
        receiver_email="to@invalid",
        subject="Hi",
        html="<p>x</p>",
    )
    assert in_memory_smtp.sent[0].thread_ident != threading.get_ident()


@pytest.mark.asyncio
async def test_send_email_smtp_failure_is_swallowed(
    in_memory_smtp: Any,
) -> None:
    """SMTP send_message errors are caught and logged; ``send_email`` itself
    does not raise so a failing email never blocks the proxy.
    """
    in_memory_smtp.raise_on_send = RuntimeError("smtp boom")
    delivered = await send_email(receiver_email="to@invalid", subject="Hi", html="<p>x</p>")
    assert delivered is False
    assert in_memory_smtp.sent == []


@pytest.mark.asyncio
async def test_should_send_plain_text_and_html_as_multipart_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = Mock()
    monkeypatch.setattr("litellm.proxy.utils._send_smtp_message", transport)
    assert (
        await send_email(
            receiver_email="to@invalid", subject="Retirement", html="<p>Model retirement</p>", text="Model retirement"
        )
        is True
    )
    message = transport.call_args.kwargs["email_message"]
    assert message.get_content_type() == "multipart/alternative"
    parts = tuple(part for part in message.walk() if not part.is_multipart())
    assert tuple(part.get_content_type() for part in parts) == ("text/plain", "text/html")
    assert parts[0].get_payload(decode=True).decode() == "Model retirement"
    assert parts[1].get_payload(decode=True).decode() == "<p>Model retirement</p>"
