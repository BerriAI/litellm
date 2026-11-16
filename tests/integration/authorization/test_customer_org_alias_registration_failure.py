import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import psycopg
from psycopg import sql

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
_CHAT_RESPONSE: Final = (
    b'{"id":"chatcmpl-scripted","object":"chat.completion","created":1700000000,"model":"scripted",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"scripted response"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}'
)


def _reply(request: Request) -> Reply:
    return Reply(body=_CHAT_RESPONSE if request.method == "POST" else b'{"object":"list","data":[]}')


@contextmanager
def _organization_insert_refused(gateway: Gateway, customer_org_id: str) -> Iterator[None]:
    name: Final = sql.Identifier(f"refuse-org-{uuid.uuid4().hex}")
    with psycopg.connect(gateway.database_url or os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            sql.SQL(
                "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $fault$ "
                "BEGIN RAISE EXCEPTION 'synthetic customer organization insert failure'; END $fault$"
            ).format(name)
        )
        connection.execute(
            sql.SQL(
                'CREATE TRIGGER {} BEFORE INSERT ON "LiteLLM_CustomerOrganizationTable" '
                "FOR EACH ROW WHEN (NEW.customer_org_id = {}) EXECUTE FUNCTION {}()"
            ).format(name, sql.Literal(customer_org_id), name)
        )
        try:
            yield
        finally:
            connection.execute(sql.SQL('DROP TRIGGER IF EXISTS {} ON "LiteLLM_CustomerOrganizationTable"').format(name))
            connection.execute(sql.SQL("DROP FUNCTION IF EXISTS {}()").format(name))


def test_a_new_customer_org_alias_is_served_untracked_when_its_registration_insert_fails(gateway: Gateway) -> None:
    alias: Final = f"integration-org-{uuid.uuid4().hex}"
    with wire_server(_reply) as wire, gateway.scenario() as scenario, _organization_insert_refused(gateway, alias):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])

        served: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
            key=key,
            headers={_ORG_ALIAS_HEADER: alias},
        )

        assert served.status_code == 200, served.text
        assert served.json()["choices"][0]["message"]["content"] == "scripted response"
        assert (
            gateway.rows(
                'SELECT customer_org_id FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = %s',
                (alias,),
            )
            == []
        )
