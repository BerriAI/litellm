import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
_ORGANIZATIONS: Final = "/management/v1/customer_organizations"
_BURST: Final = 150
_CHAT_RESPONSE: Final = (
    b'{"id":"chatcmpl-scripted","object":"chat.completion","created":1700000000,"model":"scripted",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"scripted response"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}'
)


def _reply(request: Request) -> Reply:
    return Reply(body=_CHAT_RESPONSE if request.method == "POST" else b'{"object":"list","data":[]}')


def _chat(gateway: Gateway, key: str, model: str, alias: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        key=key,
        headers={_ORG_ALIAS_HEADER: alias},
    )


def _registered_aliases(gateway: Gateway, prefix: str) -> frozenset[object]:
    rows: Final = gateway.rows(
        'SELECT customer_org_id FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id LIKE %s',
        (f"{prefix}%",),
    )
    return frozenset(row["customer_org_id"] for row in rows)


def test_a_burst_of_first_requests_on_new_org_aliases_is_served_and_registers_every_alias(gateway: Gateway) -> None:
    prefix: Final = f"integration-burst-{uuid.uuid4().hex}-"
    aliases: Final = tuple(f"{prefix}{index}" for index in range(_BURST))
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        for alias in aliases:
            scenario.cleanups.callback(
                gateway.request, "DELETE", f"{_ORGANIZATIONS}/{alias}", params={"cascade": "true"}
            )

        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            responses: Final = tuple(pool.map(lambda alias: _chat(gateway, key, model, alias), aliases))

        refused: Final = tuple(response.text for response in responses if response.status_code != 200)
        assert Counter(response.status_code for response in responses) == {200: _BURST}, refused[:1]
        assert _registered_aliases(gateway, prefix) == frozenset(aliases)
