"""
Functions for sending Email Alerts
"""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.proxy._types import WebhookEvent
from litellm.repositories.team_repository import TeamRepository

if TYPE_CHECKING:
    from reportlab.platypus import Image as ReportlabImage

    from litellm.proxy.spend_tracking.cost_savings_report import CostSavingsReport

# we use this for the email header, please send a test email if you change this. verify it looks good on email
LITELLM_LOGO_URL: Final = "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
LITELLM_SUPPORT_CONTACT: Final = "support@berri.ai"


async def get_all_team_member_emails(team_id: str | None = None) -> list:
    verbose_logger.debug("Email Alerting: Getting all team members for team_id=%s", team_id)
    if team_id is None:
        return []
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise Exception("Not connected to DB!")

    team_row: Final = await TeamRepository(prisma_client).table.find_unique(
        where={
            "team_id": team_id,
        }
    )

    if team_row is None:
        return []

    _team_members: Final = team_row.members_with_roles
    verbose_logger.debug(
        "Email Alerting: Got team members for team_id=%s Team Members: %s",
        team_id,
        _team_members,
    )
    _team_member_user_ids: Final[list[str]] = []
    for member in _team_members:
        if member and isinstance(member, dict):
            _user_id = member.get("user_id")
            if _user_id and isinstance(_user_id, str):
                _team_member_user_ids.append(_user_id)

    sql_query: Final = """
        SELECT user_email
        FROM "LiteLLM_UserTable"
        WHERE user_id = ANY($1::TEXT[]);
    """

    _result: Final = await prisma_client.db.query_raw(sql_query, _team_member_user_ids)

    verbose_logger.debug("Email Alerting: Got all Emails for team, emails=%s", _result)

    if _result is None:
        return []

    emails: Final = []
    for user in _result:
        if user and isinstance(user, dict) and user.get("user_email", None) is not None:
            emails.append(user.get("user_email"))
    return emails


async def send_team_budget_alert(webhook_event: WebhookEvent) -> bool:
    """
    Send an Email Alert to All Team Members when the Team Budget is crossed
    Returns -> True if sent, False if not.
    """
    from litellm.proxy.utils import send_email

    _team_id: Final = webhook_event.team_id
    team_alias: Final = webhook_event.team_alias
    verbose_logger.debug("Email Alerting: Sending Team Budget Alert for team=%s", team_alias)

    email_logo_url = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None))
    email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)

    # await self._check_if_using_premium_email_feature(
    #     premium_user, email_logo_url, email_support_contact
    # )

    if email_logo_url is None:
        email_logo_url = LITELLM_LOGO_URL
    if email_support_contact is None:
        email_support_contact = LITELLM_SUPPORT_CONTACT
    recipient_emails: Final = await get_all_team_member_emails(_team_id)
    recipient_emails_str: Final[str] = ",".join(recipient_emails)
    verbose_logger.debug("Email Alerting: Sending team budget alert to %s", recipient_emails_str)

    event_name: Final = webhook_event.event_message
    max_budget: Final = webhook_event.max_budget
    email_html_content = "Alert from LiteLLM Server"

    if recipient_emails_str is None:
        verbose_proxy_logger.warning(
            "Email Alerting: Trying to send email alert to no recipient, got recipient_emails=%s",
            recipient_emails_str,
        )

    email_html_content = f"""
    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" /> <br/><br/><br/>

    Budget Crossed for Team <b> {team_alias} </b> <br/> <br/>

    Your Teams LLM API usage has crossed it's <b> budget of ${max_budget} </b>, current spend is <b>${webhook_event.spend}</b><br /> <br />

    API requests will be rejected until either (a) you increase your budget or (b) your budget gets reset <br /> <br />

    If you have any questions, please send an email to {email_support_contact} <br /> <br />

    Best, <br />
    The LiteLLM team <br />
    """

    email_event: Final = {
        "to": recipient_emails_str,
        "subject": f"LiteLLM {event_name} for Team {team_alias}",
        "html": email_html_content,
    }

    await send_email(
        receiver_email=email_event["to"],
        subject=email_event["subject"],
        html=email_event["html"],
    )

    return False


def build_cost_savings_report_email(report: "CostSavingsReport") -> tuple[str, str]:
    """
    Returns (subject, html) for a cost-savings report email. Pure function, no I/O.
    """
    email_logo_url = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None)) or LITELLM_LOGO_URL
    email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None) or LITELLM_SUPPORT_CONTACT

    date_range: Final = f"{report.start_date.strftime('%b %d, %Y')} - {report.end_date.strftime('%b %d, %Y')}"
    subject: Final = f"LiteLLM Cost Savings Report ({date_range})"

    html: Final = f"""
    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" /> <br/><br/><br/>

    Cost Savings Report for <b>{date_range}</b> <br/> <br/>

    Total spend: <b>${round(report.total_spend, 4)}</b> <br/> <br/>

    <ul>
        <li>Auto Router savings: <b>${round(report.autorouter_savings_spend, 4)}</b></li>
        <li>Prompt Compression savings: <b>${round(report.compression_savings_spend, 4)}</b></li>
        <li>Prompt Caching savings: <b>${round(report.prompt_caching_savings_spend, 4)}</b></li>
    </ul>

    Total savings: <b>${round(report.total_savings, 4)}</b> <br/> <br/>

    If you have any questions, please send an email to {email_support_contact} <br/> <br/>

    Best, <br/>
    The LiteLLM team <br/>
    """

    return subject, html


LITELLM_LOGO_PATH: Final = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "proxy", "_experimental", "out", "assets", "logos", "litellm.jpg")
)
CUSTOM_LOGO_FETCH_TIMEOUT_SECONDS: Final = 5.0
_LOGO_MAX_WIDTH_INCHES: Final = 2.0
_LOGO_MAX_HEIGHT_INCHES: Final = 0.6


def _custom_logo_url() -> str | None:
    return os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None))


async def _fetch_custom_logo_bytes() -> bytes | None:
    """
    Fetches the proxy admin's custom logo (same SMTP_SENDER_LOGO / EMAIL_LOGO_URL env
    vars the HTML email already uses), for embedding in the cost savings report PDF.
    Returns None when no custom logo is configured, or on any fetch failure, so the
    caller falls back to the bundled LiteLLM logo.
    """
    custom_logo_url: Final = _custom_logo_url()
    if not custom_logo_url or not custom_logo_url.startswith(("http://", "https://")):
        return None

    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        httpxSpecialProvider,
    )

    try:
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.EmailReporting)
        response: Final = await client.get(custom_logo_url, timeout=CUSTOM_LOGO_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.content
    except Exception as e:  # noqa: BLE001  # a broken custom logo must fall back, not break the report
        verbose_proxy_logger.warning("Email Alerting: failed to fetch custom logo %s: %s", custom_logo_url, e)
        return None


def _sized_logo_image(image_bytes: bytes) -> "ReportlabImage":
    """
    Loads an ``Image`` flowable scaled to fit within the report header's logo box,
    preserving the source image's aspect ratio (custom logos can be any shape).
    """
    from io import BytesIO

    from PIL import Image as PILImage
    from reportlab.lib.units import inch
    from reportlab.platypus import Image

    with PILImage.open(BytesIO(image_bytes)) as pil_image:
        intrinsic_width, intrinsic_height = pil_image.size

    scale: Final = min(
        (_LOGO_MAX_WIDTH_INCHES * inch) / intrinsic_width,
        (_LOGO_MAX_HEIGHT_INCHES * inch) / intrinsic_height,
    )
    return Image(
        BytesIO(image_bytes),
        width=intrinsic_width * scale,
        height=intrinsic_height * scale,
        hAlign="LEFT",
    )


def build_cost_savings_report_pdf(report: "CostSavingsReport", logo_bytes: bytes | None = None) -> bytes:
    """
    Renders the cost-savings report as a one-page, left-aligned PDF, for attachment
    to the email. Pass ``logo_bytes`` for a proxy admin's custom logo; falls back to
    the bundled LiteLLM logo asset (no network I/O) when not given.
    """
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    date_range: Final = f"{report.start_date.strftime('%b %d, %Y')} - {report.end_date.strftime('%b %d, %Y')}"

    with open(LITELLM_LOGO_PATH, "rb") as f:
        fallback_logo_bytes: Final = f.read()
    logo: Final = _sized_logo_image(logo_bytes or fallback_logo_bytes)

    base_styles: Final = getSampleStyleSheet()
    title_style: Final = ParagraphStyle("LeftTitle", parent=base_styles["Title"], alignment=TA_LEFT)
    date_style: Final = ParagraphStyle(
        "LeftDate", parent=base_styles["Normal"], alignment=TA_LEFT, textColor=colors.grey
    )

    rows: Final = (
        ("Metric", "Amount"),
        ("Total spend", f"${report.total_spend:,.4f}"),
        ("Auto Router savings", f"${report.autorouter_savings_spend:,.4f}"),
        ("Prompt Compression savings", f"${report.compression_savings_spend:,.4f}"),
        ("Prompt Caching savings", f"${report.prompt_caching_savings_spend:,.4f}"),
        ("Total savings", f"${report.total_savings:,.4f}"),
    )
    table: Final = Table(rows, colWidths=(3 * inch, 2 * inch), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            (
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), (colors.white, colors.HexColor("#f5f5f5"))),
            )
        )
    )

    buffer: Final = BytesIO()
    doc: Final = SimpleDocTemplate(buffer, pagesize=letter, title="LiteLLM Cost Savings Report")
    # reportlab's SimpleDocTemplate.build() mutates (pops from) the flowables list it
    # is given, so a tuple isn't accepted here.
    flowables: Final = [  # mutable-ok: reportlab's build() deletes items from this list in place
        logo,
        Spacer(1, 0.25 * inch),
        Paragraph("LiteLLM Cost Savings Report", title_style),
        Paragraph(date_range, date_style),
        Spacer(1, 0.3 * inch),
        table,
    ]
    doc.build(flowables)
    return buffer.getvalue()


async def send_cost_savings_report_email(recipient_emails: Sequence[str], report: "CostSavingsReport") -> bool:
    """
    Send the periodic cost-savings report email to the configured recipients,
    with a PDF copy of the report attached.
    Returns True if sent, False if not.
    """
    from litellm.proxy.utils import EmailAttachment, send_email

    if not recipient_emails:
        verbose_proxy_logger.warning("Email Alerting: No recipients configured for cost savings report")
        return False

    subject, html = build_cost_savings_report_email(report)
    custom_logo_bytes: Final = await _fetch_custom_logo_bytes()
    pdf_bytes: Final = build_cost_savings_report_pdf(report, logo_bytes=custom_logo_bytes)

    await send_email(
        receiver_email=",".join(recipient_emails),
        subject=subject,
        html=html,
        attachments=(
            EmailAttachment(
                filename=f"litellm_cost_savings_report_{report.start_date}_{report.end_date}.pdf",
                content=pdf_bytes,
                mime_type="application/pdf",
            ),
        ),
    )

    return True
