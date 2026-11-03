import json
from collections.abc import Awaitable
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.realtime_api import main as realtime_main

UPSTREAM_PATHS: Final = MappingProxyType(
    {
        "client_secrets": "/v1/realtime/client_secrets",
        "translation_client_secrets": "/v1/realtime/translations/client_secrets",
        "calls": "/v1/realtime/calls",
        "translation_calls": "/v1/realtime/translations/calls",
    }
)


def _logging(route: str) -> Logging:
    return Logging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="acreate_realtime_client_secret",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id=f"test-realtime-{route}",
        function_id=f"test-realtime-{route}",
    )


def _route_call(route: str, api_base: str) -> Awaitable[httpx.Response]:
    match route:
        case "client_secrets":
            return realtime_main.acreate_realtime_client_secret.__wrapped__(
                model="gpt-realtime",
                custom_llm_provider="openai",
                api_key="sk-test",
                api_base=api_base,
                litellm_logging_obj=_logging(route),
            )
        case "translation_client_secrets":
            return realtime_main.acreate_realtime_translation_client_secret.__wrapped__(
                model="gpt-realtime-translate",
                custom_llm_provider="openai",
                api_key="sk-test",
                api_base=api_base,
                litellm_logging_obj=_logging(route),
            )
        case "calls":
            return realtime_main.arealtime_calls.__wrapped__(
                openai_ephemeral_key="ek_test",
                sdp_body=b"v=0",
                model="gpt-realtime",
                custom_llm_provider="openai",
                api_base=api_base,
                litellm_logging_obj=_logging(route),
            )
        case _:
            return realtime_main.arealtime_translation_calls.__wrapped__(
                openai_ephemeral_key="ek_test",
                sdp_body=b"v=0",
                model="gpt-realtime-translate",
                custom_llm_provider="openai",
                api_base=api_base,
                litellm_logging_obj=_logging(route),
            )


def _ephemeral_key(_: Request) -> Reply:
    return Reply(body=json.dumps({"value": "ek_test"}).encode())


@pytest.mark.asyncio
@pytest.mark.parametrize("route", tuple(UPSTREAM_PATHS))
async def test_openai_realtime_http_routes_trust_the_ca_bundle_set_in_litellm_ssl_verify(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate, key = write_self_signed_cert(tmp_path)
    monkeypatch.setattr(litellm, "ssl_verify", str(certificate))

    with wire_server(_ephemeral_key, tls=server_context(certificate, key)) as upstream:
        response: Final = await _route_call(route, upstream.url)
        targets: Final = tuple(request.target for request in upstream.drain())

    assert response.status_code == 200
    assert response.json() == {"value": "ek_test"}
    assert targets == (UPSTREAM_PATHS[route],)
