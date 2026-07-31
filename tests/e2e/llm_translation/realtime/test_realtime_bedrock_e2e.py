"""Live e2e: Bedrock Nova Sonic realtime (LIT-2239).

Customer path: open /v1/realtime, session.update, conversation.item.create,
response.create, and receive a completed response. A hang with no response.done
is the regression.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from realtime_client import (
    RealtimeClient,
    ResponseCreate,
    ResponseDone,
    SessionConfig,
    SessionUpdate,
    parse_last,
    transcript,
    user_message,
)

pytestmark = pytest.mark.e2e

NOVA_SONIC = "bedrock/amazon.nova-sonic-v1:0"


class TestNovaSonicRealtime:
    @pytest.mark.covers(
        "llm.realtime.bedrock_converse.basic.stream.works",
        exercised_on=["realtime"],
    )
    def test_nova_sonic_response_create_completes(
        self, client: RealtimeClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = f"e2e-nova-sonic-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=NOVA_SONIC,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
            mode="realtime",
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        with client.connect(key=scoped_key, model=model) as session:
            created = session.collect_until("session.created", timeout=30)
            assert created[-1].type == "session.created"

            session.send(
                SessionUpdate(
                    session=SessionConfig(
                        instructions="You are a terse assistant. Reply in one short sentence."
                    )
                )
            )
            session.collect_until("session.updated", timeout=30)

            session.send(user_message("Say the single word hello."))
            session.send(ResponseCreate())
            events = session.collect_until("response.done", timeout=90)

            types = {e.type for e in events}
            assert "response.created" in types, (
                f"Nova Sonic never emitted response.created; types={sorted(types)}"
            )
            assert transcript(events).strip() != "" or "response.done" in types, (
                "Nova Sonic response.create produced no transcript (LIT-2239 hang)"
            )
            done = parse_last(events, "response.done", ResponseDone)
            assert done is not None, (
                f"Nova Sonic never completed response.done within timeout; types={sorted(types)}"
            )
