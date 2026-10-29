import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support import device_login as dl
from integration._support import responses_vendor as rv
from integration._support.client import gateway_from_environment
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Request
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

_CREDENTIAL: Final = "copilot-ghe"
_MODEL: Final = "copilot-ghe-per-user"
_CLIENT_SECONDS: Final = 60


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    peers: dl.Peers

    def post(self, path: str, key: str, body: Mapping[str, JsonValue]) -> httpx.Response:
        with httpx.Client(
            base_url=str(self.owned.gateway.client.base_url), timeout=_CLIENT_SECONDS, trust_env=False
        ) as client:
            return client.post(path, json=dict(body), headers={"Authorization": f"Bearer {key}"})

    def disconnect(self, key: str) -> None:
        with httpx.Client(
            base_url=str(self.owned.gateway.client.base_url), timeout=_CLIENT_SECONDS, trust_env=False
        ) as client:
            response: Final = client.delete(
                f"/credentials/{_CREDENTIAL}/user_connection", headers={"Authorization": f"Bearer {key}"}
            )
        assert response.status_code == 200, response.text


def _config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["credential_list"] = [
        {
            "credential_name": _CREDENTIAL,
            "credential_values": {"github_copilot_auth_type": "per_user_oauth", "github_copilot_host": dl.GHE_HOST},
            "credential_info": {"custom_llm_provider": "github_copilot"},
        }
    ]
    config["model_list"] = [
        {
            "model_name": _MODEL,
            "litellm_params": {
                "model": "github_copilot/gpt-5.4",
                "litellm_credential_name": _CREDENTIAL,
                "github_copilot_auth_type": "per_user_oauth",
            },
        }
    ]
    path: Final = directory / "copilot-ghe-per-user.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("copilot-ghe-per-user")
    copilot_dir: Final = directory / "copilot"
    copilot_dir.mkdir()
    with (
        gateway_from_environment() as gateway,
        dl.device_login_peers(directory) as peers,
        owned_proxy_process(
            gateway,
            directory,
            {**peers.environment(), "GITHUB_COPILOT_TOKEN_DIR": str(copilot_dir)},
            config=_config(directory),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, peers)


def _outbound(request: Request) -> tuple[str, str, str, str]:
    return (
        request.headers.get("host", ""),
        request.method,
        urlsplit(request.target).path,
        request.headers.get("authorization", ""),
    )


def test_a_ghe_user_connects_and_chats_with_their_own_token_on_the_tenant_hosts(rig: _Rig) -> None:
    rig.peers.reset()
    rig.peers.switches.grant.set()
    with rig.owned.gateway.scenario() as scenario:
        user_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
        scenario.cleanups.callback(rig.disconnect, user_key)

        started: Final = rig.post(f"/credentials/{_CREDENTIAL}/user_connection/start", user_key, {})
        assert started.status_code == 200, started.text
        flow: Final = rv.JSON_OBJECT.validate_json(started.text)
        assert flow["verification_uri"] == f"https://{dl.GHE_HOST}/login/device", started.text

        polled: Final = rig.post(
            f"/credentials/{_CREDENTIAL}/user_connection/poll", user_key, {"flow_handle": flow["flow_handle"]}
        )
        assert polled.status_code == 200, polled.text
        connection: Final = rv.JSON_OBJECT.validate_json(polled.text)
        assert (connection["status"], connection["github_login"]) == ("connected", dl.GHE_LOGIN), polled.text

        marker: Final = uuid.uuid4().hex
        chat: Final = rig.post(
            "/v1/chat/completions",
            user_key,
            {
                "model": _MODEL,
                "messages": [{"role": "user", "content": f"Reply to marker-{marker}"}],
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        assert chat.status_code == 200, chat.text
        assert set(rv.MARKER.findall(chat.text)) == {marker}, chat.text

    received: Final = rig.peers.auth.drain()
    outbound: Final = tuple(_outbound(request) for request in received)
    (forwarded,) = (request for request in received if marker.encode() in request.body)
    assert _outbound(forwarded) == (
        dl.GHE_COPILOT_HOST,
        "POST",
        "/chat/completions",
        f"Bearer {dl.GHE_ACCESS}",
    ), outbound
    assert {(host, method, path) for host, method, path, _ in outbound} == {
        (dl.GHE_HOST, "POST", "/login/device/code"),
        (dl.GHE_HOST, "POST", "/login/oauth/access_token"),
        (dl.GHE_API_HOST, "GET", "/copilot_internal/user"),
        (dl.GHE_COPILOT_HOST, "GET", "/models"),
        (dl.GHE_API_HOST, "GET", "/user"),
        (dl.GHE_COPILOT_HOST, "POST", "/chat/completions"),
    }, outbound
    assert {authorization for host, _, _, authorization in outbound if host == dl.GHE_API_HOST} == {
        f"token {dl.GHE_ACCESS}"
    }, outbound
    assert {authorization for host, _, _, authorization in outbound if host == dl.GHE_COPILOT_HOST} == {
        f"Bearer {dl.GHE_ACCESS}"
    }, outbound
    assert rig.peers.api.drain() == (), "the github.com GITHUB_COPILOT_API_BASE took a ghe.com call"
