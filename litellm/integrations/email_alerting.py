"""
Functions for sending Email Alerts
"""

import os
from typing import List, Optional

from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.proxy._types import WebhookEvent

# we use this for the email header, please send a test email if you change this. verify it looks good on email
LITELLM_LOGO_URL = "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
LITELLM_SUPPORT_CONTACT = "support@berri.ai"


async def get_all_team_member_emails(team_id: Optional[str] = None) -> list:
    verbose_logger.debug(
        "Email Alerting: Getting all team members for team_id=%s", team_id
    )
    if team_id is None:
        return []
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise Exception("Not connected to DB!")

    team_row = await prisma_client.db.litellm_teamtable.find_unique(
        where={
            "team_id": team_id,
        }
    )

    if team_row is None:
        return []

    _team_members = team_row.members_with_roles
    verbose_logger.debug(
        "Email Alerting: Got team members for team_id=%s Team Members: %s",
        team_id,
        _team_members,
    )
    _team_member_user_ids: List[str] = []
    for member in _team_members:
        if member and isinstance(member, dict):
            _user_id = member.get("user_id")
            if _user_id and isinstance(_user_id, str):
                _team_member_user_ids.append(_user_id)

    sql_query = """
        SELECT user_email
        FROM "LiteLLM_UserTable"
        WHERE user_id = ANY($1::TEXT[]);
    """

    _result = await prisma_client.db.query_raw(sql_query, _team_member_user_ids)

    verbose_logger.debug("Email Alerting: Got all Emails for team, emails=%s", _result)

    if _result is None:
        return []

    emails = []
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

    _team_id = webhook_event.team_id
    team_alias = webhook_event.team_alias
    verbose_logger.debug(
        "Email Alerting: Sending Team Budget Alert for team=%s", team_alias
    )

    email_logo_url = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None))
    email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)

    # await self._check_if_using_premium_email_feature(
    #     premium_user, email_logo_url, email_support_contact
    # )

    if email_logo_url is None:
        email_logo_url = LITELLM_LOGO_URL
    if email_support_contact is None:
        email_support_contact = LITELLM_SUPPORT_CONTACT
    recipient_emails = await get_all_team_member_emails(_team_id)
    recipient_emails_str: str = ",".join(recipient_emails)
    verbose_logger.debug(
        "Email Alerting: Sending team budget alert to %s", recipient_emails_str
    )

    event_name = webhook_event.event_message
    max_budget = webhook_event.max_budget
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

    email_event = {
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
