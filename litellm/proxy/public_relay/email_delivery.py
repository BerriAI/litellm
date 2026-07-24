from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import resend

from litellm.proxy.utils import send_email


@dataclass(frozen=True, slots=True)
class VerificationEmail:
    receiver: str
    code: str
    purpose: str


async def send_verification_email(message: VerificationEmail) -> None:
    subject = "Verify your email" if message.purpose == "register" else "Reset your password"
    html = f"<p>Your verification code is <strong>{message.code}</strong>.</p><p>It expires in 10 minutes.</p>"
    resend_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("RESEND_FROM_EMAIL") or os.getenv("SMTP_SENDER_EMAIL")
    if resend_key and sender:
        resend.api_key = resend_key
        await asyncio.to_thread(
            resend.Emails.send,
            {
                "from": sender,
                "to": [message.receiver],
                "subject": subject,
                "html": html,
            },
        )
        return
    await send_email(receiver_email=message.receiver, subject=subject, html=html)
