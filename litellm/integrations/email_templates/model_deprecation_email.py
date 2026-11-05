"""Table-based, responsive model retirement emails in LiteLLM's cobalt visual style."""

from html import escape
from typing import Final
from urllib.parse import urlencode, urlsplit

from litellm.types.integrations.model_deprecation_email import (
    DeprecationEmailMilestone,
    ModelDeprecationEmail,
    RenderedDeprecationEmail,
)
from litellm.types.proxy.model_deprecation import SuccessorModelSource

_EMAIL_LOGO_URL: Final = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/ui/litellm-dashboard/public/assets/logos/litellm_logo.png"
)


def _recommendation_source(source: SuccessorModelSource | None) -> str:
    if source is SuccessorModelSource.DEPLOYMENT_CONFIG:
        return "Deployment configuration"
    if source is SuccessorModelSource.MODEL_COST_MAP:
        return "LiteLLM model cost map"
    return "Not provided"


def _http_url(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed: Final = urlsplit(value)
    except ValueError:
        return None
    return value if parsed.scheme in ("http", "https") and parsed.netloc and not parsed.username else None


def render_model_deprecation_email(email: ModelDeprecationEmail, *, base_url: str) -> RenderedDeprecationEmail:
    gateway: Final = _http_url(base_url)
    if gateway is None:
        raise ValueError("Model deprecation emails require a public HTTP(S) PROXY_BASE_URL")
    urgent: Final = email.milestone is DeprecationEmailMilestone.ONE_DAY
    days: Final = int(email.milestone)
    timing: Final = "tomorrow" if urgent else "in 30 days"
    subject: Final = f"[LiteLLM] Model being deprecated {timing}: {email.usage_model}"
    heading: Final = "Model being deprecated"
    preheader: Final = f"{email.usage_model} is being deprecated {timing}. Review your affected keys and teams before the deprecation date."
    query: Final = urlencode(
        {
            "tab": "models",
            "from": email.start_date.isoformat(),
            "to": email.end_date.isoformat(),
        }
    )
    review_url: Final = f"{gateway.rstrip('/')}/ui/usage?{query}"
    replacement: Final = email.model.successor_model or "No replacement provided"
    source: Final = _recommendation_source(email.model.successor_model_source)
    notice: Final = _http_url(email.model.deprecation_source)
    source_link: Final = (
        f'<a href="{escape(notice, quote=True)}" style="color:#263cdb;text-decoration:underline;">Provider deprecation notice &#8599;</a>'
        if notice
        else ""
    )
    team_names: Final = ", ".join(email.team_names[:3]) or "Your personal keys"
    more_teams: Final = f" + {len(email.team_names) - 3} more" if len(email.team_names) > 3 else ""
    day_label: Final = "DAY" if urgent else "DAYS"
    retirement: Final = email.model.deprecation_date.strftime("%d %b %Y")
    label: Final = "FINAL REMINDER" if urgent else "DEPRECATION NOTICE"
    advice: Final = (
        "Prioritize the remaining keys and coordinate the switch with your team before the provider deprecates this model."
        if urgent
        else "Choose a replacement and validate it against your workload. Plan the rollout with your team before the provider deprecates this model."
    )
    html: Final = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(subject)}</title>
<style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
@media only screen and (max-width:620px){{.shell{{width:100%!important}}.pad{{padding:24px!important}}.headline{{font-size:30px!important;line-height:36px!important}}.brand{{padding:20px 24px!important}}.brand img{{width:96px!important}}}}
</style></head>
<body style="margin:0;padding:0;background:#f7f8fa;color:#17213b;font-family:'DM Sans',Arial,sans-serif;">
<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;">{escape(preheader)}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"><tr><td align="center" style="padding:32px 12px;">
<table role="presentation" class="shell" width="600" cellspacing="0" cellpadding="0" border="0" style="width:600px;max-width:600px;background:#ffffff;border:1px solid #e2e6ee;">
<tr><td class="brand" style="padding:24px 36px;border-bottom:1px solid #e2e6ee;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td><img src="{_EMAIL_LOGO_URL}" width="112" alt="LiteLLM" style="display:block;border:0;height:auto;"></td>
<td align="right" style="font:11px 'Courier New',monospace;letter-spacing:1.5px;color:#526079;">/ MODEL DEPRECATIONS</td></tr></table></td></tr>
<tr><td class="pad" style="padding:32px 36px;background:#263cdb;color:#ffffff;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td style="font:11px 'Courier New',monospace;letter-spacing:1.8px;color:#ffffff;">{label}</td>
<td align="right" style="font:11px 'Courier New',monospace;color:#ffffff;">{days:02d} {day_label} UNTIL DEPRECATION</td></tr></table>
<h1 class="headline" style="margin:22px 0 20px;max-width:420px;font-size:36px;line-height:42px;font-weight:600;letter-spacing:-1px;color:#ffffff;">{heading}</h1>
<p style="margin:0;padding-top:18px;border-top:1px solid #7a89ec;font-size:14px;line-height:22px;color:#ffffff;"><span style="font-family:'Courier New',monospace;overflow-wrap:anywhere;word-break:break-word;">{escape(email.usage_model)}</span></p>
</td></tr>
<tr><td class="pad" style="padding:30px 36px;">
<p style="margin:0 0 20px;font-size:14px;line-height:22px;color:#526079;">You own keys or administer teams that used this model in the last 30 days.</p>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border:1px solid #e2e6ee;border-collapse:collapse;">
<tr><td style="padding:14px 18px;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td style="font-size:12px;color:#526079;">Deprecation date</td><td align="right" style="font:12px 'Courier New',monospace;color:#17213b;">{retirement} &middot; UTC</td></tr></table>
</td></tr></table>
<h2 style="margin:26px 0 12px;font-size:17px;font-weight:600;color:#17213b;">Your affected usage</h2>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-top:1px solid #e2e6ee;border-bottom:1px solid #e2e6ee;">
<tr><td width="50%" style="padding:16px 0;"><div style="font:24px 'Courier New',monospace;color:#17213b;">{email.key_count:,}</div><div style="margin-top:5px;font-size:12px;color:#526079;">Affected keys</div></td>
<td width="50%" style="padding:16px 0 16px 20px;border-left:1px solid #e2e6ee;"><div style="font:24px 'Courier New',monospace;color:#17213b;">{email.requests:,}</div><div style="margin-top:5px;font-size:12px;color:#526079;">Requests &middot; last 30 days</div></td></tr></table>
<p style="margin:12px 0 24px;font-size:12px;line-height:20px;color:#526079;">{escape(team_names + more_teams)}<br>Scope: your keys and teams you administer.</p>
<div style="border-left:3px solid #263cdb;padding:2px 0 2px 16px;">
<div style="font:10px 'Courier New',monospace;letter-spacing:1.2px;color:#526079;">RECOMMENDED REPLACEMENT</div>
<div style="margin:8px 0;font:14px 'Courier New',monospace;line-height:22px;overflow-wrap:anywhere;color:#17213b;">{escape(replacement)}</div>
<p style="margin:0;font-size:12px;line-height:20px;color:#526079;">Source: {escape(source)}<br>{source_link}</p></div>
<p style="margin:18px 0 24px;font-size:12px;line-height:20px;color:#526079;">Evaluate the replacement on your use case before changing production routing.</p>
<table role="presentation" cellspacing="0" cellpadding="0"><tr><td bgcolor="#263cdb" style="border-radius:4px;mso-padding-alt:14px 20px;">
<a href="{escape(review_url, quote=True)}" style="display:inline-block;padding:14px 20px;color:#ffffff;font-size:13px;font-weight:600;line-height:18px;text-decoration:none;">Review model usage &#8599;</a></td></tr></table>
<p style="margin:14px 0 0;font-size:12px;line-height:20px;color:#526079;">Opens the Models usage view for this reporting period. Sign in to see usage you have access to.</p>
</td></tr>
<tr><td class="pad" style="padding:22px 36px;background:#fafbfc;border-top:1px solid #e2e6ee;font-size:11px;line-height:18px;color:#526079;">
Two scheduled reminders per model deprecation: 30 days before and 1 day before. Your gateway admin enabled these emails. Recipients follow current key ownership and team admin membership.<br><br>LiteLLM &middot; AI Gateway</td></tr>
</table></td></tr></table></body></html>"""
    text: Final = (
        f"{subject}\n\n{advice}\n\nModel: {email.usage_model}\nDeprecation date: {retirement} UTC\n"
        f"Affected keys: {email.key_count}\nRequests in the last 30 days: {email.requests}\n"
        f"Teams: {team_names}{more_teams}\nReplacement: {replacement}\nSource: {source}\n"
        f"Provider notice: {notice or 'Not provided'}\nReview model usage: {review_url}\n\n"
        "You received this because you own affected keys or administer affected teams. "
        "Your gateway admin enabled two reminders: 30 days and 1 day before deprecation."
    )
    return RenderedDeprecationEmail(subject=subject, html=html, text=text)
