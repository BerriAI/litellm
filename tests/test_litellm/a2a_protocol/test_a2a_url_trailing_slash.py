"""
XCT fork: A2AConfig.get_complete_url must use the registered agent URL verbatim.

Agent servers (e.g. XCity's FastAPI-mounted agent gateway) serve the A2A
endpoint at the exact registered path and 307-redirect the slash-less URL to the
slash version. httpx drops the POST body across that redirect, so the agent sees
an empty request and replies with empty content. Stripping the trailing slash
therefore silently breaks every such agent — keep the URL as-is.
"""

import litellm


def _cfg():
    return litellm.A2AConfig()


def test_trailing_slash_is_preserved():
    url = _cfg().get_complete_url(
        api_base="https://agents.example.com/agents/ppc-campaign-strategist/",
        api_key=None,
        model="a2a/xct-ppc-campaign-strategist",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://agents.example.com/agents/ppc-campaign-strategist/"


def test_no_slash_url_left_untouched():
    url = _cfg().get_complete_url(
        api_base="https://agents.example.com/a2a",
        api_key=None,
        model="a2a/x",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://agents.example.com/a2a"
