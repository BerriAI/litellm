"""Live e2e: a native Vertex AI generateContent call over the proxy's /vertex_ai
passthrough is forwarded to Vertex and still logged as a costed SpendLogs row.

Ports the de-flake of the SDK-based spend test (#31689). That test configured the
vertexai SDK with an api_endpoint override pointing at the proxy, but the SDK
intermittently ignored the override and billed the public Vertex endpoint directly,
so the request never reached LiteLLM and no spend was recorded; the bypass, not
logging lag, was the flake. Driving raw HTTP through the shared transport always
reaches the proxy, which the harness already guarantees, so the only residual
nondeterminism is the ~60s async spend flush the poll absorbs.

The vertex deployment is added at runtime through the management endpoint rather than
declared in the gateway config: the test POSTs /model/new with use_in_pass_through so
the proxy registers that deployment's service account for the /vertex_ai route, then
deletes it on teardown. The credential is the one the proxy already holds (read from
the same VERTEXAI_CREDENTIALS the deployment uses), so the passthrough call sends only
its litellm virtual key in x-litellm-api-key and no upstream bearer, and the proxy
mints the Vertex token itself. The test never mints a token.

Asserts both sides of the promise: the forward succeeds (2xx with a candidate) and
the costed row lands (call_type pass_through_endpoint, vertex_ai provider, a gemini
model, spend > 0), correlated by the x-litellm-call-id header.
"""

import os

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import SpendLogRow
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

VERTEX_MODEL = "gemini-2.5-flash"
# The added deployment's region and the passthrough URL's region are the same constant,
# so they always agree; the proxy registers passthrough credentials per project+region.
VERTEX_LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-central1")


@pytest.fixture(scope="session")
def vertex_project() -> str:
    """The Vertex project to bill, read from the same VERTEXAI_PROJECT the proxy uses.
    Skip when unset, since that is an environment gap rather than a behavior failure."""
    project = os.environ.get("VERTEXAI_PROJECT")
    if not project:
        pytest.skip("set VERTEXAI_PROJECT (the project the vertex deployment bills)")
    return project


@pytest.fixture(scope="session")
def vertex_credentials() -> str:
    """The service-account JSON the added deployment authenticates with, the same
    VERTEXAI_CREDENTIALS the proxy holds. Skip when unset."""
    credentials = os.environ.get("VERTEXAI_CREDENTIALS")
    if not credentials:
        pytest.skip("set VERTEXAI_CREDENTIALS (the vertex service-account JSON)")
    return credentials


class _VertexDeploymentParams(BaseModel):
    model: str
    vertex_project: str
    vertex_location: str
    vertex_credentials: str
    use_in_pass_through: bool


class _ModelInfoId(BaseModel):
    id: str


class _ModelNewBody(BaseModel):
    model_name: str
    litellm_params: _VertexDeploymentParams
    model_info: _ModelInfoId


class _ModelNewResponse(BaseModel):
    model_id: str


class _ModelDeleteBody(BaseModel):
    id: str


def _add_vertex_passthrough_model(
    client: PassthroughClient, model_name: str, project: str, credentials: str
) -> str:
    return unwrap(
        client.gateway.transport.post(
            "/model/new",
            headers=client.gateway.transport.master,
            json=_ModelNewBody(
                model_name=model_name,
                litellm_params=_VertexDeploymentParams(
                    model=f"vertex_ai/{VERTEX_MODEL}",
                    vertex_project=project,
                    vertex_location=VERTEX_LOCATION,
                    vertex_credentials=credentials,
                    use_in_pass_through=True,
                ),
                model_info=_ModelInfoId(id=model_name),
            ),
            response_type=_ModelNewResponse,
        )
    ).model_id


def _delete_model(client: PassthroughClient, model_id: str) -> None:
    _ = client.gateway.transport.post(
        "/model/delete",
        headers=client.gateway.transport.master,
        json=_ModelDeleteBody(id=model_id),
        response_type=NoBody,
    )


def _costed_row(client: PassthroughClient, call_id: str | None) -> SpendLogRow:
    """The passthrough call's SpendLogs row, polled until it carries a cost.

    A 2xx passthrough call that produced no costed row is a hard failure, not a skip:
    a billed Vertex call that LiteLLM did not track is the exact regression #31689
    guards against."""
    assert call_id, "vertex passthrough response had no x-litellm-call-id header"
    rows = client.gateway.poll_logs_for_request_id(
        call_id,
        predicate=lambda rs: (rs[0].spend or 0) > 0,
    )
    assert rows, f"no SpendLogs row for vertex passthrough call_id {call_id}"
    row = rows[0]
    assert row.call_type == "pass_through_endpoint", f"unexpected call_type: {row}"
    assert (row.spend or 0) > 0, f"vertex passthrough call was not costed: {row}"
    assert row.status == "success", f"unexpected status: {row}"
    return row


class TestVertexPassthroughSpendTracking:
    def test_vertex_passthrough_via_managed_model_logs_cost(
        self,
        client: PassthroughClient,
        scoped_key: str,
        resources: ResourceManager,
        vertex_project: str,
        vertex_credentials: str,
    ) -> None:
        model_name = f"e2e-vertex-pt-{unique_marker()}"
        model_id = _add_vertex_passthrough_model(client, model_name, vertex_project, vertex_credentials)
        resources.defer(lambda: _delete_model(client, model_id))

        result = client.vertex_generate(
            key=scoped_key,
            project=vertex_project,
            location=VERTEX_LOCATION,
            model=VERTEX_MODEL,
            text=f"reply with one word {unique_marker()}",
        )
        require_successful_call(result)
        assert '"candidates"' in result.body, f"vertex passthrough returned no candidates: {result.body[:300]}"

        row = _costed_row(client, result.call_id)
        assert row.custom_llm_provider == "vertex_ai", f"passthrough spend logged under the wrong provider: {row}"
        assert "gemini" in (row.model or ""), f"unexpected model in spend log: {row}"
