from collections.abc import Mapping

import pytest
from fastapi import HTTPException

from litellm.proxy.agent_endpoints.identity import has_legacy_identity, reject_legacy_identity

TENANT = "11111111-1111-4111-8111-111111111111"
CLIENT = "22222222-2222-4222-8222-222222222222"


@pytest.mark.parametrize("params", [None, {}, {"model": "gpt-4o", "api_key": "sk-test"}])
def test_runtime_params_without_identity_are_accepted(params: Mapping[str, object] | None) -> None:
    assert has_legacy_identity(params) is False
    reject_legacy_identity(params)


@pytest.mark.parametrize(
    "identity", [None, {}, {"provider": "microsoft_entra", "tenant_id": TENANT, "client_id": CLIENT}]
)
def test_legacy_litellm_params_identity_is_rejected(identity: object) -> None:
    params: Mapping[str, object] = {"model": "gpt-4o", "identity": identity}
    assert has_legacy_identity(params) is True
    with pytest.raises(HTTPException) as failure:
        reject_legacy_identity(params)
    assert failure.value.status_code == 400
    assert "top-level identity field" in failure.value.detail
