import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
_ORGANIZATIONS: Final = "/management/v1/customer_organizations"
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


def _organization_ids(gateway: Gateway, customer_org_id: str) -> list[object]:
    rows: Final = gateway.rows(
        'SELECT customer_org_id FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = %s',
        (customer_org_id,),
    )
    return [row["customer_org_id"] for row in rows]


def test_a_key_pinned_to_one_customer_org_cannot_register_another_by_alias(gateway: Gateway) -> None:
    own: Final = f"integration-org-{uuid.uuid4().hex}"
    other: Final = f"integration-org-{uuid.uuid4().hex}"
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], metadata={"customer_org_id": own})
        for customer_org_id in (own, other):
            scenario.cleanups.callback(
                gateway.request, "DELETE", f"{_ORGANIZATIONS}/{customer_org_id}", params={"cascade": "true"}
            )

        refused: Final = _chat(gateway, key, model, other)
        assert refused.status_code == 403, refused.text
        assert _organization_ids(gateway, other) == []

        served: Final = _chat(gateway, key, model, own)
        assert served.status_code == 200, served.text
        assert _organization_ids(gateway, own) == [own]
