from collections.abc import Sequence
from html import escape
from typing import Final

from litellm.constants import CLI_JWT_EXPIRATION_HOURS

_CONSENT_STYLES: Final = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background-color: #f8fafc;
    margin: 0;
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    color: #1e293b;
}
.container {
    background-color: #fff;
    padding: 40px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    width: 450px;
    max-width: 100%;
}
h1 { margin: 0 0 16px; font-size: 24px; font-weight: 600; }
p { margin: 0 0 12px; line-height: 1.5; }
code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
label { display: block; margin: 16px 0 6px; font-weight: 600; }
select { width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
.actions { display: flex; gap: 12px; margin-top: 24px; }
button { flex: 1; padding: 10px; border-radius: 6px; font-size: 15px; cursor: pointer; border: 1px solid #cbd5e1; }
.approve { background: #2563eb; color: #fff; border-color: #2563eb; }
.deny { background: #fff; color: #1e293b; }
"""


def render_native_client_consent_page(
    *,
    client_origin: str,
    user_id: str,
    teams: Sequence[tuple[str, str]],
    flow_handle: str,
    complete_url: str,
) -> str:
    """The consent page a native client's sign-in lands on: who is signed in, which
    loopback client asked, which team the credential is attributed to, and an explicit
    Approve or Deny that POSTs back to ``complete_url``. Every value is client- or
    user-influenced and HTML-escaped; the flow handle travels only in the form body."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="referrer" content="no-referrer">
<title>Authorize CLI access - LiteLLM</title>
<style>
{_CONSENT_STYLES}
</style>
</head>
<body>
<div class="container">
<h1>Authorize CLI access</h1>
<p>A command-line client at <code>{escape(client_origin)}</code> wants to call LiteLLM as <strong>{escape(user_id)}</strong>.</p>
<p>Approving issues it a personal credential that expires within {CLI_JWT_EXPIRATION_HOURS} hours. <code>lite logout</code> stops it from being renewed. Only approve if you started this sign-in yourself.</p>
<form method="post" action="{escape(complete_url)}">
<input type="hidden" name="flow" value="{escape(flow_handle)}">
{_team_field(teams)}
<div class="actions">
<button type="submit" name="decision" value="deny" class="deny">Deny</button>
<button type="submit" name="decision" value="approve" class="approve">Approve</button>
</div>
</form>
</div>
</body>
</html>
"""


def _team_field(teams: Sequence[tuple[str, str]]) -> str:
    if not teams:
        return ""
    if len(teams) == 1:
        team_id, team_label = teams[0]
        return (
            f'<input type="hidden" name="team_id" value="{escape(team_id)}">'
            f"<p>Requests are attributed to team <strong>{escape(team_label)}</strong>.</p>"
        )
    options: Final = "".join(
        f'<option value="{escape(team_id)}">{escape(team_label)}</option>' for team_id, team_label in teams
    )
    return (
        f'<label for="team_id">Attribute requests to team</label><select id="team_id" name="team_id">{options}</select>'
    )


def render_mcp_connection_consent_page(
    *,
    client_name: str,
    server_name: str,
    scopes: Sequence[str],
    redirect_uri: str,
    flow_handle: str,
    complete_url: str,
    style_nonce: str,
) -> str:
    permissions: Final = (
        '<ul class="permissions">' + "".join(f"<li><code>{escape(scope)}</code></li>" for scope in scopes) + "</ul>"
        if scopes
        else '<p class="secondary">Default permissions configured by the server.</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="referrer" content="no-referrer">
<title>Connect to {escape(server_name)} - LiteLLM</title>
<style nonce="{escape(style_nonce)}">
{_CONSENT_STYLES}
body {{ box-sizing: border-box; padding: 24px; }}
.container {{ box-sizing: border-box; width: 480px; padding: 32px; border: 1px solid #e2e8f0; border-radius: 12px; }}
.brand {{ font-size: 18px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 28px; }}
h1 {{ line-height: 1.3; overflow-wrap: anywhere; }}
h2 {{ font-size: 13px; font-weight: 600; margin: 0 0 10px; }}
.secondary {{ color: #475569; font-size: 14px; }}
.permissions {{ list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }}
code {{ font-size: 12px; line-height: 1.7; overflow-wrap: anywhere; word-break: break-word; }}
section {{ padding: 20px 0; border-top: 1px solid #e2e8f0; }}
.return-address {{ display: block; background: #f8fafc; padding: 10px 12px; }}
.notice {{ padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 0; }}
button {{ font-weight: 600; min-height: 44px; }}
button:hover {{ filter: brightness(0.97); }}
button:focus-visible {{ outline: 3px solid #93c5fd; outline-offset: 3px; }}
.next-step {{ margin: 16px 0 0; text-align: center; font-size: 12px; color: #64748b; }}
@media (max-width: 480px) {{ body {{ padding: 16px; }} .container {{ padding: 24px; }} }}
</style>
</head>
<body>
<main class="container">
<div class="brand">LiteLLM</div>
<h1>Connect to <bdi>{escape(server_name)}</bdi></h1>
<p class="secondary"><strong><bdi>{escape(client_name)}</bdi></strong> is requesting access through LiteLLM.</p>
<section aria-labelledby="permissions-heading">
<h2 id="permissions-heading">Requested permissions</h2>
{permissions}
</section>
<section aria-labelledby="return-heading">
<h2 id="return-heading">Return to client</h2>
<code class="return-address">{escape(redirect_uri)}</code>
</section>
<p class="secondary notice">Only continue if you started this connection and recognize the client and return address.</p>
<form method="post" action="{escape(complete_url)}">
<input type="hidden" name="flow" value="{escape(flow_handle)}">
<div class="actions">
<button type="submit" name="decision" value="deny" class="deny">Cancel</button>
<button type="submit" name="decision" value="approve" class="approve">Continue</button>
</div>
</form>
<p class="next-step">Next, authorize access on the server's sign-in page.</p>
</main>
</body>
</html>
"""
