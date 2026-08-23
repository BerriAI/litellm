

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.proxy.common_utils.html_forms.native_client_consent import render_native_client_consent_page


def _render(teams=(), **overrides) -> str:
    arguments = {
        "client_origin": "http://127.0.0.1:51234",
        "user_id": "u1",
        "teams": teams,
        "flow_handle": "handle-123",
        "complete_url": "https://llm.example.com/authorize/complete",
    }
    return render_native_client_consent_page(**{**arguments, **overrides})


def test_consent_page_posts_the_flow_handle_and_both_decisions_to_the_complete_url():
    page = _render()
    assert '<meta name="referrer" content="no-referrer">' in page
    assert '<form method="post" action="https://llm.example.com/authorize/complete">' in page
    assert '<input type="hidden" name="flow" value="handle-123">' in page
    assert '<button type="submit" name="decision" value="deny"' in page
    assert '<button type="submit" name="decision" value="approve"' in page
    assert "<code>http://127.0.0.1:51234</code>" in page
    assert "<strong>u1</strong>" in page
    assert 'name="team_id"' not in page


def test_consent_page_pins_a_single_team_without_a_chooser():
    page = _render(teams=(("team-a", "Team A"),))
    assert '<input type="hidden" name="team_id" value="team-a">' in page
    assert "<strong>Team A</strong>" in page
    assert "<select" not in page


def test_consent_page_offers_a_chooser_for_several_teams():
    page = _render(teams=(("team-a", "Team A"), ("team-b", "team-b")))
    assert '<select id="team_id" name="team_id">' in page
    assert '<option value="team-a">Team A</option>' in page
    assert '<option value="team-b">team-b</option>' in page
    assert 'type="hidden" name="team_id"' not in page


def test_consent_page_escapes_every_untrusted_value():
    page = _render(
        teams=(('t"><script>', "<b>x</b>"),),
        client_origin="http://127.0.0.1:1/<svg>",
        user_id='<img src=x onerror="y">',
        flow_handle='h" onmouseover="z',
        complete_url="https://llm.example.com/authorize/complete?x=<y>",
    )
    assert "<script>" not in page
    assert "<b>x</b>" not in page
    assert "<svg>" not in page
    assert "<img" not in page
    assert 'onmouseover="z' not in page
    assert "?x=<y>" not in page
    assert "&lt;b&gt;x&lt;/b&gt;" in page
    assert 'value="h&quot; onmouseover=&quot;z"' in page


def test_consent_page_promises_only_what_logout_can_deliver():
    page = _render()
    assert f"expires within {CLI_JWT_EXPIRATION_HOURS} hours" in page
    assert "<code>lite logout</code> stops it from being renewed" in page
    assert "revoked" not in page
