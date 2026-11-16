import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
_ORGANIZATIONS: Final = "/management/v1/customer_organizations"
_CHAT_ROUTE: Final = "/v1/chat/completions"
_CHAT_RESPONSE: Final = (
    b'{"id":"chatcmpl-scripted","object":"chat.completion","created":1700000000,"model":"scripted",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"scripted response"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}'
)

CUSTOM_AUTH_DOUBLE: Final = """
import os

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    if api_key == os.environ["LITELLM_MASTER_KEY"]:
        return UserAPIKeyAuth(api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN)
    if api_key != os.environ["INTEGRATION_CUSTOM_AUTH_KEY"]:
        raise Exception("unknown caller")
    return UserAPIKeyAuth(api_key=api_key, user_id="integration-custom-caller")
"""


def _reply(request: Request) -> Reply:
    return Reply(body=_CHAT_RESPONSE if request.method == "POST" else b'{"object":"list","data":[]}')


def _custom_auth_config(directory: Path, public_routes: list[str]) -> Path:
    (directory / "customer_custom_auth.py").write_text(CUSTOM_AUTH_DOUBLE)
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "customer-custom-auth.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "general_settings": {
                    **object_value(base["general_settings"]),
                    "custom_auth": "customer_custom_auth.user_api_key_auth",
                    "public_routes": public_routes,
                },
            }
        )
    )
    return path


@pytest.mark.timeout(480)
@pytest.mark.parametrize("public_routes", [[], [_CHAT_ROUTE]], ids=["route_not_public", "route_in_public_routes"])
def test_custom_auth_caller_of_an_over_budget_customer_org_is_refused(
    gateway: Gateway, tmp_path: Path, public_routes: list[str]
) -> None:
    organization: Final = f"integration-org-{uuid.uuid4().hex}"
    caller: Final = "sk-custom-" + uuid.uuid4().hex
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        created: Final = gateway.request(
            "POST",
            _ORGANIZATIONS,
            {"customer_org_id": organization, "alias": organization, "budget": {"max_budget": 0}},
        )
        assert created.status_code == 201, created.text
        scenario.cleanups.callback(
            gateway.request, "DELETE", f"{_ORGANIZATIONS}/{organization}", params={"cascade": "true"}
        )
        with owned_proxy(
            gateway,
            tmp_path,
            {"INTEGRATION_CUSTOM_AUTH_KEY": caller},
            config=_custom_auth_config(tmp_path, public_routes),
            state=OwnedState.SHARED_WITH_WORKER,
        ) as candidate:
            refused: Final = candidate.request(
                "POST",
                _CHAT_ROUTE,
                {"model": model, "messages": [{"role": "user", "content": "hi"}]},
                key=caller,
                headers={_ORG_ALIAS_HEADER: organization},
            )
            assert "budget_exceeded" in refused.text, f"{refused.status_code} {refused.text}"
            assert f"customer_org={organization}" in refused.text, refused.text
