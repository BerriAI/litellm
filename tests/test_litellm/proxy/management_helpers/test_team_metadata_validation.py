import asyncio
import os
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_helpers.team_metadata_validation import (
    DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE,
    DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS,
    DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE,
    TeamMetadataRequester,
    TeamMetadataValidationPayload,
    TeamMetadataValidationResult,
    TeamMetadataValidatorRegistry,
    _read_timeout_seconds,
    _read_unavailable_message,
    run_team_metadata_validation,
    validate_team_metadata_if_configured,
)


def _registry_with(validator):
    registry = TeamMetadataValidatorRegistry()
    registry.set(validator)
    return registry

UNAVAILABLE_MESSAGE = "validation system down, contact ops"


def _payload(**overrides):
    values = {
        "operation": "create",
        "metadata": {"cost_center": "CC-1001"},
        "existing_metadata": None,
        "team_id": "team-1",
        "team_alias": "alias-1",
        "requester": TeamMetadataRequester(user_id="u1"),
    }
    values.update(overrides)
    return TeamMetadataValidationPayload(**values)


async def _run(validator, payload=None, premium_user=True, timeout_seconds=1.0):
    await run_team_metadata_validation(
        validator=validator,
        payload=payload or _payload(),
        premium_user=premium_user,
        timeout_seconds=timeout_seconds,
        unavailable_message=UNAVAILABLE_MESSAGE,
    )


@pytest.mark.asyncio
async def test_valid_result_passes():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    await _run(validator)


@pytest.mark.asyncio
async def test_rejection_raises_400_with_validator_message():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=False, error_message="cost center rejected, contact FinOps")

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": "cost center rejected, contact FinOps"}


@pytest.mark.asyncio
async def test_rejection_without_message_uses_default():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=False)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE}


@pytest.mark.asyncio
async def test_dict_shaped_return_is_accepted():
    async def validator(payload):
        return {"valid": False, "error_message": "rejected via dict"}

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": "rejected via dict"}


@pytest.mark.asyncio
async def test_validator_exception_fails_closed_with_generic_message():
    async def validator(payload):
        raise RuntimeError("internal validation service is down")

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_validator_timeout_fails_closed():
    async def validator(payload):
        await asyncio.sleep(1.0)
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator, timeout_seconds=0.01)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_malformed_return_shape_fails_closed():
    async def validator(payload):
        return "not-a-validation-result"

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_non_premium_user_is_rejected():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator, premium_user=False)
    assert exc_info.value.status_code == 400
    assert CommonProxyErrors.not_premium_user.value in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_non_coroutine_validator_is_rejected():
    def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 500
    assert "async" in exc_info.value.detail["error"]


@pytest.mark.parametrize(
    "general_settings, expected",
    [
        ({}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": 2}, 2.0),
        ({"team_metadata_validation_timeout": 0.5}, 0.5),
        ({"team_metadata_validation_timeout": True}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": -1}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": "3"}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
    ],
)
def test_read_timeout_seconds(general_settings, expected):
    assert _read_timeout_seconds(general_settings) == expected


@pytest.mark.parametrize(
    "general_settings, expected",
    [
        ({}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
        (
            {"team_metadata_validation_error_message": "call the help desk"},
            "call the help desk",
        ),
        ({"team_metadata_validation_error_message": "  "}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
        ({"team_metadata_validation_error_message": None}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
    ],
)
def test_read_unavailable_message(general_settings, expected):
    assert _read_unavailable_message(general_settings) == expected


@pytest.mark.asyncio
async def test_adapter_is_noop_when_unconfigured():
    calls = []

    async def validator(payload):
        calls.append(payload)
        return TeamMetadataValidationResult(valid=True)

    await validate_team_metadata_if_configured(
        operation="create",
        metadata={"cost_center": "CC-1001"},
        existing_metadata=None,
        team_id="team-1",
        team_alias="alias-1",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="u1"),
        registry=TeamMetadataValidatorRegistry(),
    )
    assert calls == []


@pytest.mark.asyncio
async def test_adapter_builds_payload_and_reads_settings():
    recorded = []

    async def validator(payload):
        recorded.append(payload)
        return TeamMetadataValidationResult(valid=True)

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"team_metadata_validation_timeout": 3, "team_metadata_validation_error_message": "ops msg"},
        ),
    ):
        await validate_team_metadata_if_configured(
            operation="update",
            metadata={"cost_center": "CC-2001"},
            existing_metadata={"cost_center": "CC-1001", "keep": 1},
            team_id="team-9",
            team_alias="alias-9",
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                user_id="user-9",
                user_email="user-9@example.com",
            ),
            registry=_registry_with(validator),
        )

    assert len(recorded) == 1
    payload = recorded[0]
    assert payload.operation == "update"
    assert payload.metadata == {"cost_center": "CC-2001"}
    assert payload.existing_metadata == {"cost_center": "CC-1001", "keep": 1}
    assert payload.team_id == "team-9"
    assert payload.team_alias == "alias-9"
    assert payload.requester.user_id == "user-9"
    assert payload.requester.user_email == "user-9@example.com"
    assert payload.requester.user_role == LitellmUserRoles.INTERNAL_USER.value


@pytest.mark.asyncio
async def test_adapter_normalizes_non_dict_metadata_to_empty_dict():
    recorded = []

    async def validator(payload):
        recorded.append(payload)
        return TeamMetadataValidationResult(valid=True)

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        await validate_team_metadata_if_configured(
            operation="create",
            metadata=None,
            existing_metadata=None,
            team_id="team-1",
            team_alias=None,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="u1"),
            registry=_registry_with(validator),
        )

    assert len(recorded) == 1
    assert recorded[0].metadata == {}
    assert recorded[0].existing_metadata is None


# ---------------------------------------------------------------------------
# Validator implementation matrix: three independent implementations
# (allowlist function, HTTP-service-backed function, immutability class
# instance) driven through the real team write endpoints.
# ---------------------------------------------------------------------------

import json as _json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock, Mock

import team_metadata_validator_impls as impls

from litellm.proxy._types import ProxyException
from litellm.proxy.management_helpers.team_metadata_validation import (
    TEAM_METADATA_VALIDATOR_REGISTRY,
)


class _CostCenterServiceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = _json.loads(self.rfile.read(length) or b"{}")
        cost_center = (body.get("metadata") or {}).get("cost_center")
        if cost_center is None:
            resp = {"ok": False, "reason": "cost_center missing per cost center service"}
        elif cost_center not in impls.ALLOWED_COST_CENTERS:
            resp = {"ok": False, "reason": f"cost center {cost_center} rejected by cost center service"}
        else:
            resp = {"ok": True}
        payload = _json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def cost_center_service_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CostCenterServiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/validate"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _closed_port_url():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}/validate"


@contextmanager
def _configured(validator):
    TEAM_METADATA_VALIDATOR_REGISTRY.set(validator)
    try:
        with (
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            yield
    finally:
        TEAM_METADATA_VALIDATOR_REGISTRY.set(None)


async def _drive_create(metadata, mock_sink=None):
    from fastapi import Request

    from litellm.proxy._types import LiteLLM_TeamTable, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as pc,
        patch("litellm.proxy.proxy_server._license_check") as lic,
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
    ):
        team_row = MagicMock(team_id="matrix-team-1")
        team_row.model_dump.return_value = {"team_id": "matrix-team-1"}
        pc.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
        pc.get_data = AsyncMock(return_value=None)
        pc.update_data = AsyncMock(return_value=MagicMock())
        pc.db.litellm_teamtable.create = AsyncMock(return_value=team_row)
        pc.db.litellm_teamtable.count = AsyncMock(return_value=0)
        pc.db.litellm_teamtable.update = AsyncMock(return_value=team_row)
        pc.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())
        pc.db.litellm_modeltable.create = AsyncMock(return_value=MagicMock(id="model-1"))
        lic.is_team_count_over_limit.return_value = False
        if mock_sink is not None:
            mock_sink["team_create"] = pc.db.litellm_teamtable.create
            mock_sink["model_create"] = pc.db.litellm_modeltable.create

        request_kwargs = {"team_alias": "matrix-team"}
        if metadata is not None:
            request_kwargs["metadata"] = metadata
        return await new_team(
            data=NewTeamRequest(**request_kwargs),
            http_request=MagicMock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )


async def _drive_update(kind, existing_metadata, payload):
    from fastapi import Request

    from litellm.proxy._types import LiteLLM_TeamTable, UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import patch_team, update_team

    team_id = "matrix-team-upd"
    existing = LiteLLM_TeamTable(
        team_id=team_id,
        team_alias="matrix",
        metadata=existing_metadata,
        organization_id=None,
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as pc,
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._refresh_cached_team",
            new=AsyncMock(),
        ),
    ):
        pc.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing)
        pc.db.litellm_teamtable.update = AsyncMock(
            return_value=LiteLLM_TeamTable(team_id=team_id, team_alias="matrix")
        )
        pc.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)

        req = Mock(spec=Request)
        if kind == "post":
            return await update_team(
                data=UpdateTeamRequest(team_id=team_id, **payload),
                http_request=req,
                user_api_key_dict=auth,
                litellm_changed_by=None,
            )
        req.json = AsyncMock(return_value=dict(payload))
        return await patch_team(
            team_id=team_id,
            http_request=req,
            user_api_key_dict=auth,
            litellm_changed_by=None,
        )


OK = ("ok", None)

_MATRIX_IMPLS = {
    "allowlist": lambda: impls.validate_allowlist,
    "http": lambda: impls.validate_via_http,
    "immutable_class": lambda: impls.IMMUTABLE_COST_CENTER_VALIDATOR,
}

# (scenario, kind, existing_metadata, request payload, {impl: expected})
# expected is ("ok", None) or ("reject", <message substring>)
_MATRIX_SCENARIOS = [
    (
        "create-valid-cost-center",
        "create",
        None,
        {"cost_center": "CC-1001"},
        {"allowlist": OK, "http": OK, "immutable_class": OK},
    ),
    (
        "create-missing-cost-center",
        "create",
        None,
        None,
        {
            "allowlist": ("reject", "cost_center is required"),
            "http": ("reject", "cost_center missing per cost center service"),
            "immutable_class": ("reject", "cost_center is required"),
        },
    ),
    (
        "create-unknown-cost-center",
        "create",
        None,
        {"cost_center": "CC-9999"},
        {
            "allowlist": ("reject", "is not recognized"),
            "http": ("reject", "rejected by cost center service"),
            "immutable_class": OK,
        },
    ),
    (
        "patch-change-cost-center",
        "patch",
        {"cost_center": "CC-1001"},
        {"metadata": {"cost_center": "CC-1002"}},
        {
            "allowlist": OK,
            "http": OK,
            "immutable_class": ("reject", "immutable once set"),
        },
    ),
    (
        "patch-unrelated-key-preserves-cost-center",
        "patch",
        {"cost_center": "CC-1001"},
        {"metadata": {"notes": "hello"}},
        {"allowlist": OK, "http": OK, "immutable_class": OK},
    ),
    (
        "patch-null-deletes-cost-center",
        "patch",
        {"cost_center": "CC-1001"},
        {"metadata": {"cost_center": None}},
        {
            "allowlist": ("reject", "cost_center is required"),
            "http": ("reject", "cost_center missing per cost center service"),
            "immutable_class": ("reject", "cost_center is required"),
        },
    ),
    (
        "post-replace-drops-cost-center",
        "post",
        {"cost_center": "CC-1001"},
        {"metadata": {"notes": "only-notes"}},
        {
            "allowlist": ("reject", "cost_center is required"),
            "http": ("reject", "cost_center missing per cost center service"),
            "immutable_class": ("reject", "cost_center is required"),
        },
    ),
    (
        "update-without-metadata-skips-validation",
        "post",
        {"cost_center": "CC-1001"},
        {"tpm_limit": 5},
        {"allowlist": OK, "http": OK, "immutable_class": OK},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("impl_name", sorted(_MATRIX_IMPLS))
@pytest.mark.parametrize(
    "scenario, kind, existing_metadata, request_payload, expectations",
    _MATRIX_SCENARIOS,
    ids=[row[0] for row in _MATRIX_SCENARIOS],
)
async def test_validator_implementation_matrix(
    monkeypatch,
    cost_center_service_url,
    impl_name,
    scenario,
    kind,
    existing_metadata,
    request_payload,
    expectations,
):
    monkeypatch.setenv("TEAM_METADATA_VALIDATION_SERVICE_URL", cost_center_service_url)
    validator = _MATRIX_IMPLS[impl_name]()
    outcome, message_part = expectations[impl_name]

    async def drive():
        if kind == "create":
            return await _drive_create(metadata=request_payload)
        return await _drive_update(kind, existing_metadata, request_payload)

    with _configured(validator):
        if outcome == "ok":
            await drive()
        else:
            with pytest.raises(ProxyException) as exc_info:
                await drive()
            assert str(exc_info.value.code) == "400", f"{scenario} x {impl_name}"
            assert message_part in str(exc_info.value.message), f"{scenario} x {impl_name}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, existing_metadata, request_payload",
    [
        ("create", None, {"cost_center": "CC-1001"}),
        ("patch", {"cost_center": "CC-1001"}, {"metadata": {"notes": "x"}}),
    ],
)
async def test_http_validator_service_outage_fails_closed(monkeypatch, kind, existing_metadata, request_payload):
    monkeypatch.setenv("TEAM_METADATA_VALIDATION_SERVICE_URL", _closed_port_url())

    with _configured(impls.validate_via_http):
        with pytest.raises(ProxyException) as exc_info:
            if kind == "create":
                await _drive_create(metadata=request_payload)
            else:
                await _drive_update(kind, existing_metadata, request_payload)

    assert str(exc_info.value.code) == "503"
    assert DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_class_instance_with_async_call_is_accepted():
    await _run(impls.ImmutableCostCenterValidator(), payload=_payload(metadata={"cost_center": "CC-1"}))


@pytest.mark.asyncio
async def test_class_instance_with_sync_call_is_rejected():
    class SyncValidator:
        def __call__(self, payload):
            return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(SyncValidator())
    assert exc_info.value.status_code == 500
