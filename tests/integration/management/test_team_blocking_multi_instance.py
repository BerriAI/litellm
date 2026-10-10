from typing import Final

import httpx

from tests.integration._support.client import Gateway, eventually, object_value

_PROMPT: Final = {"messages": [{"role": "user", "content": "team blocking"}]}


def _chat(proxy: Gateway, model: str, key: str) -> httpx.Response:
    return proxy.request("POST", "/v1/chat/completions", {"model": model, **_PROMPT}, key=key)


def _rejected(proxy: Gateway, model: str, key: str) -> httpx.Response:
    return eventually(lambda: _chat(proxy, model, key), lambda response: response.status_code == 401, seconds=70)


def _blocked_flag(proxy: Gateway, team_id: str) -> object:
    return object_value(proxy.get("/team/info", {"team_id": team_id})["team_info"])["blocked"]


def test_a_team_blocked_on_one_instance_is_rejected_on_both(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert _blocked_flag(peer, team) is False
        allowed: Final = _chat(gateway, model, key)
        assert allowed.status_code == 200, allowed.text

        peer.post("/team/update", {"team_id": team, "blocked": True})

        assert _blocked_flag(gateway, team) is True
        assert _blocked_flag(peer, team) is True
        rejections: Final = tuple(_rejected(proxy, model, key) for proxy in (gateway, peer))

    for rejection in rejections:
        error: Final = object_value(rejection.json()["error"])
        assert error["type"] == "auth_error", rejection.text
        assert f"Team={team} is blocked" in str(error["message"]), rejection.text
