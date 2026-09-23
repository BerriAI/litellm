"""A database outage during JWT team resolution is a 503, never a model-access denial."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
from db_outage_gateway import DbOutageGateway, owned_db_outage_gateway
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import Success, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, TeamNewBody
from other_client import OtherClient
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.e2e

MODEL: Final = CHEAP_OPENAI_MODEL
CACHE_TTL_SECONDS: Final = 2


class ErrorPayload(BaseModel):
    message: str = ""
    type: str = ""


class ErrorEnvelope(BaseModel):
    error: ErrorPayload


@pytest.fixture
def outage_gateway(client: OtherClient, tmp_path: Path) -> Iterator[DbOutageGateway]:
    with ExitStack() as cleanup:
        gateway: Final = owned_db_outage_gateway(client.idp, MODEL, tmp_path, cleanup)
        cleanup.callback(gateway.relay.restore)
        yield gateway


def _ping() -> ChatBody:
    return ChatBody(
        model=MODEL,
        messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
        max_tokens=16,
    )


class TestJwtTeamReadDuringDbOutage:
    @pytest.mark.covers("other.auth.jwt.db_outage_is_not_model_denial")
    def test_db_outage_during_team_read_is_503_not_model_denial(
        self, client: OtherClient, resources: ResourceManager, outage_gateway: DbOutageGateway
    ) -> None:
        marker: Final = unique_marker()
        identity: Final = client.idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}", defer=resources.defer)
        resources.defer(lambda: client.proxy.delete_user(identity.user_id))
        team_id: Final = outage_gateway.proxy.create_team(
            TeamNewBody(team_alias=f"e2e-jwt-outage-{marker}", team_id=identity.group, models=[MODEL])
        )
        resources.defer(lambda: client.proxy.delete_team(team_id))
        token: Final = client.idp.access_token(identity)

        healthy: Final = outage_gateway.proxy.chat(token, _ping())
        assert isinstance(healthy, Success), (
            f"precondition: a valid JWT for a team that owns {MODEL} must get a 200, got {healthy}"
        )
        assert unwrap(healthy).choices, f"healthy call returned no completion: {healthy}"

        outage_gateway.relay.cut()
        time.sleep(CACHE_TTL_SECONDS + 1)

        result: Final = outage_gateway.proxy.chat(token, _ping())
        match result:
            case UnknownApiError(status_code=503, body=body):
                try:
                    error: Final = ErrorEnvelope.model_validate_json(body).error
                except ValidationError:
                    pytest.fail(f"503 during database outage was not the proxy error shape: {body[:500]}")
                assert error.type == "no_db_connection", (
                    f"a database outage during team resolution must surface as no_db_connection, "
                    f"got error type {error.type!r}: {body[:500]}"
                )
            case _:
                pytest.fail(
                    "database outage during team resolution was reported as a model-access denial or "
                    f"another error instead of 503 no_db_connection: {result}"
                )
