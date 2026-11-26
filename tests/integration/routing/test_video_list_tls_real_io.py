from pathlib import Path
from typing import Final

import pytest
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, wire_server

import litellm
from litellm.proxy.route_llm_request import route_request


@pytest.mark.asyncio
async def test_a_video_list_reaches_a_self_signed_backend_whose_deployment_turns_tls_checks_off(tmp_path: Path) -> None:
    certificate, key = write_self_signed_cert(tmp_path)
    request: Final = {
        "model": "dep-b",
        "query_params": {},
        "custom_llm_provider": "volcengine",
        "litellm_metadata": {"user_api_key_team_id": "team-b"},
    }

    with wire_server(lambda _: Reply(body=b'{"items": [], "total": 0}'), tls=server_context(certificate, key)) as wire:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "team-b-seedance",
                    "litellm_params": {
                        "model": "volcengine/doubao-seedance-2-0-mini-260615",
                        "api_key": "dep-b-key",
                        "api_base": wire.url,
                        "ssl_verify": False,
                    },
                    "model_info": {"id": "dep-b", "team_id": "team-b", "team_public_model_name": "seedance"},
                }
            ]
        )
        listed: Final = await (await route_request(request, router, None, "avideo_list"))

    assert listed["data"] == [], listed
