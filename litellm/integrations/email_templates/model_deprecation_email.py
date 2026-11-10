"""Model deprecation emails with provider identity, a migration comparison and next steps."""

from collections.abc import Mapping
from html import escape
from types import MappingProxyType
from typing import Final
from urllib.parse import urlencode, urlsplit

from litellm.types.integrations.model_deprecation_email import (
    DeprecationEmailMilestone,
    ModelDeprecationEmail,
    RenderedDeprecationEmail,
)
from litellm.types.proxy.model_deprecation import SuccessorModelSource

_LOGO_BASE_URL: Final = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/ui/litellm-dashboard/public/assets/logos"
)
_EMAIL_LOGO_URL: Final = f"{_LOGO_BASE_URL}/litellm_logo.png"
# Reuse dashboard brand assets; PNGs also render in email clients that block SVG.
_PROVIDER_BRANDS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "gemini": ("Google", "google_pse.png"),
        "vertex_ai": ("Google", "google_pse.png"),
        "openai": ("OpenAI", "openai.png"),
        "anthropic": ("Anthropic", "anthropic.png"),
        "azure": ("Microsoft Azure", "microsoft_azure.png"),
        "bedrock": ("AWS Bedrock", "aws.png"),
        "bedrock_converse": ("AWS Bedrock", "aws.png"),
    }
)


def _recommendation_source(source: SuccessorModelSource | None) -> str:
    if source is SuccessorModelSource.DEPLOYMENT_CONFIG:
        return "Gateway configuration"
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


def _provider(model: str, fallback: str | None = None) -> str:
    prefix, separator, _ = model.partition("/")
    return prefix if separator else fallback or ""


def _provider_name(provider: str) -> str:
    brand: Final = _PROVIDER_BRANDS.get(provider)
    if brand:
        return brand[0]
    return provider.replace("_", " ") or "The provider"


def _model_html(model: str, provider: str) -> str:
    brand: Final = _PROVIDER_BRANDS.get(provider)
    logo: Final = (
        f'<td width="38" valign="middle"><img src="{_LOGO_BASE_URL}/{brand[1]}" width="24" height="24" alt="{escape(brand[0], quote=True)}" style="display:block;object-fit:contain;border:0;"></td>'
        if brand
        else ""
    )
    return (
        f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr>{logo}'
        f'<td><div style="font-size:11px;line-height:18px;color:#526079;">{escape(_provider_name(provider))}</div>'
        f"<div style=\"font:14px 'Courier New',monospace;line-height:22px;word-break:break-word;overflow-wrap:anywhere;color:#17213b;\">{escape(model)}</div></td></tr></table>"
    )


def render_model_deprecation_email(email: ModelDeprecationEmail, *, base_url: str) -> RenderedDeprecationEmail:
    gateway: Final = _http_url(base_url)
    if gateway is None:
        raise ValueError("Model deprecation emails require a public HTTP(S) PROXY_BASE_URL")
    urgent: Final = email.milestone is DeprecationEmailMilestone.ONE_DAY
    days: Final = int(email.milestone)
    timing: Final = "tomorrow" if urgent else "in 30 days"
    provider: Final = _provider(email.usage_model, email.model.litellm_provider)
    provider_name: Final = _provider_name(provider)
    subject: Final = f"[LiteLLM] Model being deprecated by {provider_name} {timing}: {email.usage_model}"
    deadline: Final = email.model.deprecation_date.strftime("%d %b %Y")
    announcement: Final = f"{provider_name} is deprecating this model on {deadline} (UTC)."
    preheader: Final = f"{announcement} Review the model change and migrate your affected keys."
    query: Final = urlencode(
        {
            "tab": "models",
            "from": email.start_date.isoformat(),
            "to": email.end_date.isoformat(),
        }
    )
    review_url: Final = f"{gateway.rstrip('/')}/ui/usage?{query}"
    replacement: Final = email.model.successor_model
    after: Final = (
        _model_html(replacement, _provider(replacement, provider))
        if replacement
        else '<p style="margin:0;font-size:14px;line-height:22px;color:#17213b;">No recommended model listed. Choose a supported replacement.</p>'
    )
    source: Final = _recommendation_source(email.model.successor_model_source)
    notice: Final = _http_url(email.model.deprecation_source)
    source_link: Final = (
        f'<a href="{escape(notice, quote=True)}" style="color:#263cdb;text-decoration:underline;">Read the provider notice &#8599;</a>'
        if notice
        else ""
    )
    team_names: Final = ", ".join(email.team_names[:3]) or "Personal keys"
    more_teams: Final = f" + {len(email.team_names) - 3} more" if len(email.team_names) > 3 else ""
    day_label: Final = "DAY" if urgent else "DAYS"
    label: Final = "FINAL REMINDER" if urgent else "DEPRECATION NOTICE"
    test_step: Final = (
        "Test the recommended model with your workload."
        if replacement
        else "Choose a supported replacement and test it with your workload."
    )
    switch_step: Final = (
        f"Switch your app or gateway routing before {deadline}. Then check that requests to the old model reach zero."
    )
    html: Final = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(subject)}</title>
<style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
@media only screen and (max-width:620px){{.pad{{padding:24px!important}}.headline{{font-size:30px!important;line-height:36px!important}}.brand{{padding:20px 24px!important}}.brand img{{width:96px!important}}.comparison-half{{display:block!important;width:auto!important;border-right:0!important}}.comparison-after{{border-top:1px solid #e2e6ee!important}}}}
</style></head>
<body style="margin:0;padding:0;background:#f7f8fa;color:#17213b;font-family:'DM Sans',Arial,sans-serif;">
<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;">{escape(preheader)}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"><tr><td style="padding:0;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;background:#ffffff;">
<tr><td class="brand" style="padding:24px 36px;border-bottom:1px solid #e2e6ee;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td><img src="{_EMAIL_LOGO_URL}" width="112" alt="LiteLLM" style="display:block;border:0;height:auto;"></td>
<td align="right" style="font:11px 'Courier New',monospace;letter-spacing:1.5px;color:#526079;">/ MODEL DEPRECATIONS</td></tr></table></td></tr>
<tr><td class="pad" style="padding:32px 36px;background:#263cdb;color:#ffffff;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td style="font:11px 'Courier New',monospace;letter-spacing:1.8px;color:#ffffff;">{label}</td>
<td align="right" style="font:11px 'Courier New',monospace;color:#ffffff;">{days:02d} {day_label} UNTIL DEPRECATION</td></tr></table>
<h1 class="headline" style="margin:22px 0 16px;font-size:36px;line-height:42px;font-weight:600;letter-spacing:-1px;color:#ffffff;">Model being deprecated</h1>
<p style="margin:0;font-size:14px;line-height:22px;color:#ffffff;">{escape(announcement)}</p>
</td></tr>
<tr><td class="pad" style="padding:28px 36px;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border:1px solid #e2e6ee;border-collapse:collapse;">
<tr><td class="comparison-half" width="50%" valign="top" style="padding:16px 18px;border-right:1px solid #e2e6ee;">
<div style="margin-bottom:8px;font:10px 'Courier New',monospace;letter-spacing:1px;color:#526079;">BEFORE &middot; MODEL TO REPLACE</div>
{_model_html(email.usage_model, provider)}
</td>
<td class="comparison-half comparison-after" width="50%" valign="top" style="padding:16px 18px;background:#f5f7ff;">
<div style="margin-bottom:8px;font:10px 'Courier New',monospace;letter-spacing:1px;color:#263cdb;">AFTER &middot; RECOMMENDED MODEL</div>
{after}
</td></tr></table>
<p style="margin:10px 0 24px;font-size:12px;line-height:20px;color:#526079;">Recommendation source: {escape(source)}<br>{source_link}</p>
<h2 style="margin:0 0 12px;font-size:17px;font-weight:600;color:#17213b;">Your affected usage</h2>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-top:1px solid #e2e6ee;border-bottom:1px solid #e2e6ee;">
<tr><td width="50%" style="padding:14px 0;"><div style="font:24px 'Courier New',monospace;color:#17213b;">{email.key_count:,}</div><div style="margin-top:5px;font-size:12px;color:#526079;">Affected keys</div></td>
<td width="50%" style="padding:14px 0 14px 20px;border-left:1px solid #e2e6ee;"><div style="font:24px 'Courier New',monospace;color:#17213b;">{email.requests:,}</div><div style="margin-top:5px;font-size:12px;color:#526079;">Requests &middot; last 30 days</div></td></tr></table>
<p style="margin:10px 0 24px;font-size:12px;line-height:20px;color:#526079;">{escape(team_names + more_teams)}<br>Your keys and teams you administer.</p>
<h2 style="margin:0 0 10px;font-size:17px;font-weight:600;color:#17213b;">What to do</h2>
<ol style="margin:0 0 24px;padding-left:20px;font-size:13px;line-height:22px;color:#526079;">
<li style="padding:0 0 8px 4px;">Review the affected keys in your usage dashboard.</li>
<li style="padding:0 0 8px 4px;">{escape(test_step)}</li>
<li style="padding-left:4px;">{escape(switch_step)}</li></ol>
<table role="presentation" cellspacing="0" cellpadding="0"><tr><td bgcolor="#263cdb" style="border-radius:4px;mso-padding-alt:14px 20px;">
<a href="{escape(review_url, quote=True)}" style="display:inline-block;padding:14px 20px;color:#ffffff;font-size:13px;font-weight:600;line-height:18px;text-decoration:none;">Review affected usage &#8599;</a></td></tr></table>
</td></tr>
<tr><td class="pad" style="padding:20px 36px;background:#fafbfc;border-top:1px solid #e2e6ee;font-size:11px;line-height:18px;color:#526079;">
Your gateway admin enabled these emails. Reminder schedule: 30 days and 1 day before deprecation.<br><br>LiteLLM &middot; AI Gateway</td></tr>
</table></td></tr></table></body></html>"""
    text: Final = (
        f"{subject}\n\n{announcement}\n\n"
        f"BEFORE - MODEL TO REPLACE\n{email.usage_model}\n\n"
        f"AFTER - RECOMMENDED MODEL\n{replacement or 'No recommended model listed. Choose a supported replacement.'}\n"
        f"Recommendation source: {source}\nProvider notice: {notice or 'Not provided'}\n\n"
        f"YOUR AFFECTED USAGE\n{email.key_count} keys | {email.requests} requests in the last 30 days\n"
        f"{team_names}{more_teams}\nYour keys and teams you administer.\n\n"
        f"WHAT TO DO\n1. Review the affected keys in your usage dashboard.\n2. {test_step}\n3. {switch_step}\n\n"
        f"Review affected usage: {review_url}\n\n"
        "Your gateway admin enabled these emails. Reminder schedule: 30 days and 1 day before deprecation."
    )
    return RenderedDeprecationEmail(subject=subject, html=html, text=text)
