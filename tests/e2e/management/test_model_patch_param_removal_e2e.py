"""Live e2e for removing a litellm_params key from a DB-backed deployment.

The Admin UI edit form sends the whole LiteLLM Params JSON on PATCH
/model/{id}/update. A key the operator deleted from that JSON is absent from the
body, and the stored deployment must lose it too; otherwise a param such as
max_tokens can never be taken off a deployment without editing the database.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import LiteLLMParamsBody
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

_BACKEND: Final = "gpt-4o-mini"
_STORED_MAX_TOKENS: Final = 256


class _ModelInfoParams(BaseModel):
    litellm_model_id: str


class _StoredParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature: float | None = None
    max_tokens: int | None = None


class _StoredEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    litellm_params: _StoredParams = _StoredParams()


class _StoredInfo(BaseModel):
    data: list[_StoredEntry] = []


class _PatchParams(BaseModel):
    model: str
    mock_response: str
    temperature: float


class _PatchBody(BaseModel):
    litellm_params: _PatchParams


def _stored(client: ManagementClient, model_id: str) -> _StoredParams | None:
    entries = unwrap(
        client.proxy.transport.get(
            "/model/info",
            headers=client.proxy.management_headers(),
            params=_ModelInfoParams(litellm_model_id=model_id),
            response_type=_StoredInfo,
        )
    ).data
    return entries[0].litellm_params if entries else None


class TestModelPatchParamRemoval:
    def test_key_dropped_from_edited_params_is_removed_from_stored_deployment(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        model_name = f"e2e-param-removal-{unique_marker()}"
        model_id = client.proxy.create_model(
            model_name,
            LiteLLMParamsBody(
                model=_BACKEND,
                mock_response="ok",
                max_tokens=_STORED_MAX_TOKENS,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        before = _stored(client, model_id)
        assert before is not None and before.max_tokens == _STORED_MAX_TOKENS, (
            f"/model/info does not report the registered max_tokens: {before}"
        )

        unwrap(
            client.proxy.transport.patch(
                f"/model/{model_id}/update",
                headers=client.proxy.management_headers(),
                json=_PatchBody(litellm_params=_PatchParams(model=_BACKEND, mock_response="ok", temperature=0.3)),
                response_type=NoBody,
            )
        )

        deadline = time.monotonic() + client.proxy.poll_timeout
        after = _stored(client, model_id)
        while time.monotonic() < deadline and (after is None or after.temperature != 0.3):
            time.sleep(client.proxy.poll_interval)
            after = _stored(client, model_id)
        assert after is not None and after.temperature == 0.3, f"the PATCH never reached the stored deployment: {after}"
        assert after.max_tokens is None, (
            f"max_tokens was dropped from the edited params but the stored deployment still has "
            f"max_tokens={after.max_tokens}"
        )
