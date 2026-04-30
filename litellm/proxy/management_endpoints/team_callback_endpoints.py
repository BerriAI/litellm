"""
Endpoints to control callbacks per team

Use this when each team should control its own callbacks
"""

import asyncio
import copy
import json
import traceback
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    AddTeamCallback,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    ProxyErrorTypes,
    ProxyException,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

router = APIRouter()


_CALLBACK_VARS_REDACTED = "***REDACTED***"


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
    redacted = copy.deepcopy(metadata)
    logging_entries = redacted.get("logging")
    if isinstance(logging_entries, list):
        for entry in logging_entries:
            if isinstance(entry, dict) and isinstance(entry.get("callback_vars"), dict):
                entry["callback_vars"] = {
                    k: _CALLBACK_VARS_REDACTED for k in entry["callback_vars"]
                }
    callback_settings = redacted.get("callback_settings")
    if isinstance(callback_settings, dict) and isinstance(
        callback_settings.get("callback_vars"), dict
    ):
        callback_settings["callback_vars"] = {
            k: _CALLBACK_VARS_REDACTED for k in callback_settings["callback_vars"]
        }
    return redacted


def _log_audit_task_exception(task: "asyncio.Task[None]") -> None:
    """Surface a fire-and-forget audit-log task failure.

    ``asyncio.create_task`` swallows exceptions silently — if the audit
    write fails (transient DB error etc.) we'd otherwise lose the row
    without any signal.  Log at warning level so the operator sees there's
    a gap in the audit trail.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        verbose_proxy_logger.warning("Failed to write team-callback audit log: %s", exc)


async def _emit_team_callback_audit_log(
    *,
    team_id: str,
    before_metadata: Any,
    after_metadata: Any,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
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
    if litellm.store_audit_logs is not True:
        return

    from litellm.proxy.management_helpers.audit_logs import (
        create_audit_log_for_update,
    )
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    redacted_before = _redact_callback_secrets(before_metadata)
    redacted_after = _redact_callback_secrets(after_metadata)

    task = asyncio.create_task(
        create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by
                or user_api_key_dict.user_id
                or litellm_proxy_admin_name,
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
    litellm_changed_by: Optional[str] = Header(
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
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Check if team_id exists already
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Team id = {team_id} does not exist. Please use a different team id."
                },
            )

        # store team callback settings in metadata
        team_metadata = _existing_team.metadata
        team_callback_settings: List[dict] = team_metadata.get(
            "logging"
        )  # will be dict of type AddTeamCallback
        if team_callback_settings is None or not isinstance(
            team_callback_settings, list
        ):
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

        before_metadata = copy.deepcopy(team_metadata)
        team_callback_settings.append(data.model_dump())

        team_metadata["logging"] = team_callback_settings
        team_metadata_json = json.dumps(team_metadata)  # update team_metadata

        new_team_row = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
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
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.add_team_callbacks(): Exception occured - {}".format(
                str(e)
            )
        )
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error.value,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


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
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Disable all logging callbacks for a team

    Parameters:
    - team_id (str, required): The unique identifier for the team

    Example curl:
    ```
    curl -X POST 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/disable_logging' \
        -H 'Authorization: Bearer sk-1234'
    ```


    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

        # Check if team exists
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # Update team metadata to disable logging
        team_metadata = _existing_team.metadata
        before_metadata = copy.deepcopy(team_metadata)
        team_callback_settings = team_metadata.get("callback_settings", {})
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)

        # Reset callbacks
        team_callback_settings_obj.success_callback = []
        team_callback_settings_obj.failure_callback = []

        # Update metadata
        team_metadata["callback_settings"] = team_callback_settings_obj.model_dump()
        team_metadata_json = json.dumps(team_metadata)

        # Update team in database
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team id = {team_id} does not exist. Error updating team logging"
                },
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

    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.proxy_server.disable_team_logging(): Exception occurred - {str(e)}"
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
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
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # Retrieve team callback settings from metadata
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})

        # Convert to TeamCallbackMetadata object for consistent structure
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)

        return {
            "status": "success",
            "data": {
                "team_id": team_id,
                "success_callbacks": team_callback_settings_obj.success_callback,
                "failure_callbacks": team_callback_settings_obj.failure_callback,
                "callback_vars": team_callback_settings_obj.callback_vars,
            },
        }

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_team_callbacks(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
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
