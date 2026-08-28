"""
Endpoints to control callbacks per team

Use this when each team should control its own callbacks
"""

import asyncio
import copy
import json
import traceback
from datetime import datetime, timezone
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    AddTeamCallback,
    LiteLLM_AuditLogs,
    LiteLLM_TeamTable,
    LitellmTableNames,
    ProxyErrorTypes,
    ProxyException,
    TeamCallbackDeleteResponse,
    TeamCallbackDeleteResponseData,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.callback_config_validation import callback_config_error
from litellm.proxy.common_utils.callback_utils import (
    _CALLBACK_VAR_ENCRYPTED_PREFIX,
    decrypt_callback_vars,
    encrypt_callback_vars,
    is_sensitive_callback_key,
)
from litellm.proxy.litellm_pre_call_utils import (
    _get_validated_callback_metadata,
    convert_key_logging_metadata_to_callback,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    _refresh_cached_team,
    _verify_team_access,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.repositories.team_repository import TeamRepository

router: Final = APIRouter()


_CALLBACK_VARS_REDACTED: Final = "***REDACTED***"


def _callback_config_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": message})  # mutable-ok: FastAPI detail contract


def _validate_team_callback(data: "AddTeamCallback") -> None:
    error: Final = callback_config_error(data.callback_name, data.callback_vars)
    if error is not None:
        raise _callback_config_error(error)


def _redact_callback_secrets(metadata: Any) -> Any:
    """Strip secret values out of a team-metadata snapshot before audit logging.

    Both ``team_metadata["logging"]`` (list of ``AddTeamCallback`` dicts) and
    ``team_metadata["callback_settings"]["callback_vars"]`` carry provider
    credentials such as ``langfuse_secret_key``, ``langsmith_api_key``, and
    ``gcs_path_service_account``.  Persisting them verbatim into
    ``LiteLLM_AuditLogs`` would let anyone with read access to the audit
    table harvest team callback credentials, so we replace each value with
    a fixed marker.  The keys themselves are kept so the audit reader can
    still see *which* fields changed.
    """
    if not isinstance(metadata, dict):
        return metadata
    redacted: Final = copy.deepcopy(metadata)
    logging_entries: Final = redacted.get("logging")
    if isinstance(logging_entries, list):
        for entry in logging_entries:
            if isinstance(entry, dict) and isinstance(entry.get("callback_vars"), dict):
                entry["callback_vars"] = {k: _CALLBACK_VARS_REDACTED for k in entry["callback_vars"]}
    callback_settings: Final = redacted.get("callback_settings")
    if isinstance(callback_settings, dict) and isinstance(callback_settings.get("callback_vars"), dict):
        callback_settings["callback_vars"] = {k: _CALLBACK_VARS_REDACTED for k in callback_settings["callback_vars"]}
    return redacted


def _mask_sensitive_callback_vars(callbacks: TeamCallbackMetadata) -> None:
    """Mask credential-bearing callback vars in place, keeping the rest readable.

    ``callback_vars`` mixes credentials (``langsmith_api_key``,
    ``langfuse_secret_key``, ``gcs_path_service_account``) with plain
    configuration (project names, bucket names, hosts). The configuration is
    what makes a read of this endpoint useful, so only the sensitive keys are
    replaced, using the same marker as the audit-log redaction above.

    A value that still carries the encrypted prefix here failed to decrypt, so
    it is masked too. Handing back a ciphertext blob under a key that is not
    classified as sensitive would give the caller something it cannot use and
    cannot tell apart from a real value.

    Masking in place rather than rebuilding the mapping keeps this under the
    LIT002 mutable-collection-construction budget. It is safe because the only
    caller passes an object it just built from a decrypted deep copy of the
    row, so nothing here is reachable from the team's stored metadata.
    """
    if not callbacks.callback_vars:
        return
    for key in tuple(callbacks.callback_vars):
        value = callbacks.callback_vars[key]
        if is_sensitive_callback_key(key) or str(value).startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX):
            callbacks.callback_vars[key] = _CALLBACK_VARS_REDACTED


def _resolve_team_callbacks(team_metadata: object) -> TeamCallbackMetadata:
    """Report the callbacks that are actually in effect for a team.

    A team's callback config can live in either of two metadata slots.
    ``metadata["logging"]`` holds the ``AddTeamCallback`` entries written by
    ``POST /team/{team_id}/callback`` and by the Admin UI, while
    ``metadata["callback_settings"]`` holds the older ``TeamCallbackMetadata``
    shape. Request-time resolution in ``_get_dynamic_logging_metadata`` treats
    the two as mutually exclusive: a populated ``logging`` slot wins outright
    and ``callback_settings`` is consulted only as the deprecated fallback.
    This reader applies the same precedence so it reports what a request would
    really do. Merging the two instead would report a ``callback_settings``
    entry as active for a team whose requests never fire it.

    Credential ``callback_vars`` are stored encrypted, so they are decrypted
    before being masked by key; a value encrypted under a key that is no longer
    classified as sensitive would otherwise come back as raw ciphertext.
    """
    if not isinstance(team_metadata, dict):
        return TeamCallbackMetadata()

    decrypted: Final = decrypt_callback_vars(team_metadata)
    logging_entries: Final = decrypted.get("logging")

    if logging_entries is not None:
        resolved = TeamCallbackMetadata()
        for entry in logging_entries if isinstance(logging_entries, list) else ():
            if not isinstance(entry, dict):
                continue
            callback = _get_validated_callback_metadata(item=entry, source="team-level read")
            if callback is None:
                continue
            resolved = convert_key_logging_metadata_to_callback(data=callback, team_callback_settings_obj=resolved)
    else:
        callback_settings: Final = decrypted.get("callback_settings")
        resolved = (
            TeamCallbackMetadata(**callback_settings) if isinstance(callback_settings, dict) else TeamCallbackMetadata()
        )

    _mask_sensitive_callback_vars(resolved)
    return resolved


def _log_audit_task_exception(task: "asyncio.Task[None]") -> None:
    """Surface a fire-and-forget audit-log task failure.

    ``asyncio.create_task`` swallows exceptions silently — if the audit
    write fails (transient DB error etc.) we'd otherwise lose the row
    without any signal.  Log at warning level so the operator sees there's
    a gap in the audit trail.
    """
    if task.cancelled():
        return
    exc: Final = task.exception()
    if exc is not None:
        verbose_proxy_logger.warning("Failed to write team-callback audit log: %s", exc)


async def _emit_team_callback_audit_log(
    *,
    team_id: str,
    before_metadata: Any,
    after_metadata: Any,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None,
) -> None:
    """Emit an audit-log row for a team-callback mutation.

    Mirrors the ``store_audit_logs``-gated pattern used in
    ``team_endpoints.py``: the call is async-fire-and-forget and is a no-op
    when audit logging is not enabled on the proxy.  Captured under
    ``LitellmTableNames.TEAM_TABLE_NAME`` so the row co-locates with other
    team mutations in the audit table.

    Callback secrets are redacted before serialization so the audit table
    cannot itself become a credential-harvest sink.
    """
    from litellm.proxy.management_helpers.audit_logs import (
        create_audit_log_for_update,
        is_audit_logging_enabled,
    )
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    if not is_audit_logging_enabled():
        return

    redacted_before: Final = _redact_callback_secrets(before_metadata)
    redacted_after: Final = _redact_callback_secrets(after_metadata)

    task: Final = asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by or user_api_key_dict.user_id or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.TEAM_TABLE_NAME,
                object_id=team_id,
                action="updated",
                updated_values=json.dumps({"metadata": redacted_after}, default=str),
                before_value=json.dumps({"metadata": redacted_before}, default=str),
            )
        )
    )
    task.add_done_callback(_log_audit_task_exception)


def _callback_error(status_code: int, message: str) -> HTTPException:
    """Build the ``{"error": ...}`` failure body the team callback endpoints return."""
    return HTTPException(
        status_code=status_code,
        detail={"error": message},  # mutable-ok: the error response body is a JSON object
    )


@router.post(
    "/team/{team_id:path}/callback",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def add_team_callbacks(
    data: AddTeamCallback,
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Add a success/failure callback to a team

    Use this if if you want different teams to have different success/failure callbacks

    Parameters:
    - callback_name (Literal["langfuse", "langsmith", "gcs"], required): The name of the callback to add
    - callback_type (Literal["success", "failure", "success_and_failure"], required): The type of callback to add. One of:
        - "success": Callback for successful LLM calls
        - "failure": Callback for failed LLM calls
        - "success_and_failure": Callback for both successful and failed LLM calls
    - callback_vars (StandardCallbackDynamicParams, required): A dictionary of variables to pass to the callback
        - langfuse_public_key: The public key for the Langfuse callback
        - langfuse_secret_key: The secret key for the Langfuse callback
        - langfuse_secret: The secret for the Langfuse callback
        - langfuse_host: The host for the Langfuse callback
        - langfuse_environment: The tracing environment for the Langfuse callback (lowercase; falls back to LANGFUSE_TRACING_ENVIRONMENT)
        - gcs_bucket_name: The name of the GCS bucket
        - gcs_path_service_account: The path to the GCS service account
        - langsmith_api_key: The API key for the Langsmith callback
        - langsmith_project: The project for the Langsmith callback
        - langsmith_base_url: The base URL for the Langsmith callback

    Example curl:
    ```
    curl -X POST 'http:/localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer sk-1234' \
        -d '{
        "callback_name": "langfuse",
        "callback_type": "success",
        "callback_vars": {"langfuse_public_key": "pk-lf-xxxx1", "langfuse_secret_key": "sk-xxxxx"}
        
    }'
    ```

    This means for the team where team_id = dbe2f686-a686-4896-864a-4c3924458709, all LLM calls will be logged to langfuse using the public key pk-lf-xxxx1 and the secret key sk-xxxxx

    """
    try:
        from litellm.proxy._types import CommonProxyErrors
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Check if team_id exists already
        _existing_team = await prisma_client.get_data(team_id=team_id, table_name="team", query_type="find_unique")
        if _existing_team is None:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Team id = {team_id} does not exist. Please use a different team id."},
            )

        # IDOR guard: only proxy admins / org admins / team admins of THIS
        # team may write callback credentials. Without this, any
        # authenticated key holder could overwrite another team's logging
        # config (and read back the credentials they wrote).
        await _verify_team_access(
            team_obj=LiteLLM_TeamTable(**_existing_team.model_dump()),
            user_api_key_dict=user_api_key_dict,
        )

        _validate_team_callback(data)

        # store team callback settings in metadata
        team_metadata = _existing_team.metadata
        team_callback_settings: list[dict] = team_metadata.get("logging")  # will be dict of type AddTeamCallback
        if team_callback_settings is None or not isinstance(team_callback_settings, list):
            team_callback_settings = []

        ## check if it already exists, for the same callback event
        for callback in team_callback_settings:
            if (
                callback.get("callback_name") == data.callback_name
                and callback.get("callback_type") == data.callback_type
            ):
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in team_callback_settings, for team_id = {team_id} and event = {data.callback_type}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

        before_metadata: Final = copy.deepcopy(team_metadata)
        team_callback_settings.append(data.model_dump())

        team_metadata["logging"] = team_callback_settings
        team_metadata = encrypt_callback_vars(team_metadata)
        team_metadata_json: Final = json.dumps(team_metadata)  # update team_metadata

        new_team_row: Final = await TeamRepository(prisma_client).table.update(
            where={"team_id": team_id},
            data={"metadata": team_metadata_json},
            # `object_permission` is included so `_refresh_cached_team` doesn't
            # write a cached team with the relation nulled out — see
            # team_model_add for the full rationale.
            include={"object_permission": True},  # mutable-ok: prisma include takes a dict literal
        )

        if new_team_row is None:
            raise _callback_error(400, f"Team id = {team_id} does not exist. Please use a different team id.")

        # Without this a newly registered callback stays dormant for existing keys.
        await _refresh_cached_team(
            team_row=new_team_row,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        await _emit_team_callback_audit_log(
            team_id=team_id,
            before_metadata=before_metadata,
            after_metadata=team_metadata,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

        return {
            "status": "success",
            "data": new_team_row,
        }

    except HTTPException as e:
        raise e
    except ProxyException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.add_team_callbacks(): Exception occured - %s", e)
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.delete(
    "/team/{team_id:path}/callback/{callback_name}",
    tags=["team management"],  # mutable-ok: FastAPI's route decorator takes a list of tags
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI's route decorator takes a list of dependencies
    response_model=TeamCallbackDeleteResponse,
)
@management_endpoint_wrapper
async def delete_team_callback(
    http_request: Request,
    team_id: str,
    callback_name: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    litellm_changed_by: Annotated[
        str | None,
        Header(
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability"
        ),
    ] = None,
):
    """
    Remove a single callback from a team

    The team's other callbacks stay registered and keep firing. Use this instead of
    POST /team/{team_id}/disable_logging, which clears every callback on the team at once.

    Every entry registered under this callback_name is removed, across callback types, so a
    callback registered for both "success" and "failure" is deregistered by one call.

    Parameters:
    - team_id (str, required): The unique identifier for the team
    - callback_name (str, required): The name of the callback to remove, matched exactly as it was
      registered with POST /team/{team_id}/callback (e.g. "langfuse", "langsmith", "gcs")

    Example curl:
    ```
    curl -X DELETE 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback/langsmith' \
        -H 'Authorization: Bearer sk-1234'
    ```

    Covers callbacks registered through POST /team/{team_id}/callback and the Admin UI. Teams still
    on the deprecated callback_settings metadata shape hold no such entries, so this returns 404 for
    them; POST /team/{team_id}/disable_logging remains the way to clear those.

    Returns 404 if the team does not exist, or if callback_name is not registered for the team.
    """
    try:
        from litellm.proxy._types import CommonProxyErrors
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise _callback_error(500, CommonProxyErrors.db_not_connected_error.value)

        _existing_team: Final = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise _callback_error(404, f"Team id = {team_id} does not exist.")

        # IDOR guard: only proxy admins / org admins / team admins of THIS team may
        # deregister its callbacks, otherwise any authenticated key holder could
        # silence another team's observability integration.
        await _verify_team_access(
            team_obj=LiteLLM_TeamTable(**_existing_team.model_dump()),
            user_api_key_dict=user_api_key_dict,
        )

        team_metadata: Final = _existing_team.metadata
        registered_callbacks: Final = team_metadata.get("logging")
        entries: Final = registered_callbacks if isinstance(registered_callbacks, list) else ()

        remaining_callbacks: Final = [  # mutable-ok: metadata["logging"] is isinstance-checked for list downstream
            entry for entry in entries if not (isinstance(entry, dict) and entry.get("callback_name") == callback_name)
        ]
        if len(remaining_callbacks) == len(entries):
            raise _callback_error(404, f"callback_name = {callback_name} is not registered for team_id = {team_id}.")

        updated_metadata: Final = {**team_metadata, "logging": remaining_callbacks}  # mutable-ok: persisted as JSON
        encrypted_metadata: Final = encrypt_callback_vars(updated_metadata)
        team_metadata_json: Final = json.dumps(encrypted_metadata)

        updated_team: Final = await TeamRepository(prisma_client).table.update(
            where={"team_id": team_id},  # mutable-ok: prisma where takes a dict literal
            data={"metadata": team_metadata_json},  # mutable-ok: prisma data takes a dict literal
            # `object_permission` is included so `_refresh_cached_team` doesn't write a
            # cached team with the relation nulled out, see team_model_add for the rationale.
            include={"object_permission": True},  # mutable-ok: prisma include takes a dict literal
        )

        if updated_team is None:
            raise _callback_error(404, f"Team id = {team_id} does not exist. Error removing team callback")

        # Request-time callback resolution reads the cached team, so without this
        # the removed callback keeps firing for live keys until the cache expires.
        await _refresh_cached_team(
            team_row=updated_team,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        await _emit_team_callback_audit_log(
            team_id=team_id,
            before_metadata=team_metadata,
            after_metadata=encrypted_metadata,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

        # Report what survives with the same resolution the GET endpoint uses, so a
        # caller can confirm in one round trip that its other callbacks are intact.
        surviving: Final = _resolve_team_callbacks(encrypted_metadata)

        response: Final = TeamCallbackDeleteResponse(
            status="success",
            message=f"Callback {callback_name} removed for team {team_id}",
            data=TeamCallbackDeleteResponseData(
                team_id=team_id,
                success_callbacks=tuple(surviving.success_callback or ()),
                failure_callbacks=tuple(surviving.failure_callback or ()),
            ),
        )

    except HTTPException:
        # Legitimate 4xx (403 from the access guard, 404 for an unknown team or
        # an unregistered callback). Re-raise without the error-level log noise
        # the catch-all below would produce.
        raise
    except ProxyException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.delete_team_callback(): Exception occurred - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    else:
        return response


@router.post(
    "/team/{team_id}/disable_logging",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def disable_team_logging(
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: str | None = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Disable all logging callbacks for a team

    Callbacks registered through POST /team/{team_id}/callback and the Admin UI are cleared, so
    re-enabling logging means registering them again with their callback_vars

    Parameters:
    - team_id (str, required): The unique identifier for the team

    Example curl:
    ```
    curl -X POST 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/disable_logging' \
        -H 'Authorization: Bearer sk-1234'
    ```


    """
    try:
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team exists
        _existing_team = await prisma_client.get_data(team_id=team_id, table_name="team", query_type="find_unique")
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # IDOR guard: only proxy admins / org admins / team admins of THIS
        # team may disable its logging — otherwise any authenticated key
        # holder can silence audit logging for any team.
        await _verify_team_access(
            team_obj=LiteLLM_TeamTable(**_existing_team.model_dump()),
            user_api_key_dict=user_api_key_dict,
        )

        # Update team metadata to disable logging
        team_metadata = _existing_team.metadata
        before_metadata: Final = copy.deepcopy(team_metadata)
        team_callback_settings: Final = team_metadata.get("callback_settings", {})
        team_callback_settings_obj: Final = TeamCallbackMetadata(**team_callback_settings)

        # Reset callbacks
        team_callback_settings_obj.success_callback = []
        team_callback_settings_obj.failure_callback = []

        # Update metadata
        team_metadata["callback_settings"] = team_callback_settings_obj.model_dump()
        # _get_dynamic_logging_metadata stops at metadata["logging"], where the API
        # and Admin UI register callbacks, without ever reading callback_settings.
        team_metadata["logging"] = []  # mutable-ok: the disabled state is persisted as an empty JSON array
        team_metadata = encrypt_callback_vars(team_metadata)
        team_metadata_json: Final = json.dumps(team_metadata)

        # Update team in database
        updated_team: Final = await TeamRepository(prisma_client).table.update(
            where={"team_id": team_id},
            data={"metadata": team_metadata_json},
            # `object_permission` is included so `_refresh_cached_team` doesn't
            # write a cached team with the relation nulled out — see
            # team_model_add for the full rationale.
            include={"object_permission": True},  # mutable-ok: prisma include takes a dict literal
        )

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist. Error updating team logging"},
            )

        # Request-time callback resolution reads the cached team, so without this
        # the DB says logging is off while live keys keep sending until it expires.
        await _refresh_cached_team(
            team_row=updated_team,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Disabling a team's logging callbacks is itself a logging-control
        # action — emit an audit-log row so the action remains traceable
        # even though the team's own observability is now off.
        await _emit_team_callback_audit_log(
            team_id=team_id,
            before_metadata=before_metadata,
            after_metadata=team_metadata,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

        return {
            "status": "success",
            "message": f"Logging disabled for team {team_id}",
            "data": {
                "team_id": updated_team.team_id,
                "success_callbacks": [],
                "failure_callbacks": [],
            },
        }

    except HTTPException:
        # Legitimate 4xx (e.g. 403 from the access guard, 404 for an
        # unknown team). Re-raise without the error-level log noise that
        # the catch-all branch below would produce.
        raise
    except ProxyException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.disable_team_logging(): Exception occurred - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/team/{team_id:path}/callback",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def get_team_callbacks(
    http_request: Request,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the success/failure callbacks and variables for a team

    Parameters:
    - team_id (str, required): The unique identifier for the team

    Example curl:
    ```
    curl -X GET 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
        -H 'Authorization: Bearer sk-1234'
    ```

    This will return the callback settings for the team with id dbe2f686-a686-4896-864a-4c3924458709

    Covers callbacks registered through POST /team/{team_id}/callback and the Admin UI as well as
    teams still on the deprecated callback_settings shape, resolved from the team's stored metadata
    with the same precedence used at request time. A key-level logging config overrides the team's
    at request time and is not reflected here. Credential-bearing callback_vars are returned masked
    as `***REDACTED***`

    Returns {
            "status": "success",
            "data": {
                "team_id": team_id,
                "success_callbacks": team_callback_settings_obj.success_callback,
                "failure_callbacks": team_callback_settings_obj.failure_callback,
                "callback_vars": team_callback_settings_obj.callback_vars,
            },
        }
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team_id exists
        _existing_team = await prisma_client.get_data(team_id=team_id, table_name="team", query_type="find_unique")
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # IDOR guard: callback metadata holds third-party API credentials
        # (Langfuse / Langsmith / GCS). Only proxy admins / org admins /
        # team admins of THIS team may read them.
        await _verify_team_access(
            team_obj=LiteLLM_TeamTable(**_existing_team.model_dump()),
            user_api_key_dict=user_api_key_dict,
        )

        team_callback_settings_obj: Final = _resolve_team_callbacks(_existing_team.metadata)

        return {
            "status": "success",
            "data": {
                "team_id": team_id,
                "success_callbacks": team_callback_settings_obj.success_callback,
                "failure_callbacks": team_callback_settings_obj.failure_callback,
                "callback_vars": team_callback_settings_obj.callback_vars,
            },
        }

    except HTTPException:
        # Legitimate 4xx (e.g. 403 from the access guard) — re-raise
        # without the error-level log noise that the catch-all below
        # would produce.
        raise
    except ProxyException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.get_team_callbacks(): Exception occurred - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({e})"),
                type=ProxyErrorTypes.internal_server_error.value,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
