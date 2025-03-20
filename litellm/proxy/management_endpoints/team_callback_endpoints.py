"""
Endpoints to control callbacks per team

Use this when each team should control its own callbacks
"""

import json
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    AddTeamCallback,
    ProxyErrorTypes,
    ProxyException,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

router = APIRouter()


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
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})

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
        team_callback_settings = team_metadata.get("callback_settings", {})
        # expect callback settings to be
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)
        if data.callback_type == "success":
            if team_callback_settings_obj.success_callback is None:
                team_callback_settings_obj.success_callback = []

            if data.callback_name in team_callback_settings_obj.success_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.success_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            team_callback_settings_obj.success_callback.append(data.callback_name)
        elif data.callback_type == "failure":
            if team_callback_settings_obj.failure_callback is None:
                team_callback_settings_obj.failure_callback = []

            if data.callback_name in team_callback_settings_obj.failure_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.failure_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )
            team_callback_settings_obj.failure_callback.append(data.callback_name)
        elif data.callback_type == "success_and_failure":
            if team_callback_settings_obj.success_callback is None:
                team_callback_settings_obj.success_callback = []
            if team_callback_settings_obj.failure_callback is None:
                team_callback_settings_obj.failure_callback = []
            if data.callback_name in team_callback_settings_obj.success_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in success_callback, for team_id = {team_id}. \n Existing success_callback = {team_callback_settings_obj.success_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            if data.callback_name in team_callback_settings_obj.failure_callback:
                raise ProxyException(
                    message=f"callback_name = {data.callback_name} already exists in failure_callback, for team_id = {team_id}. \n Existing failure_callback = {team_callback_settings_obj.failure_callback}",
                    code=status.HTTP_400_BAD_REQUEST,
                    type=ProxyErrorTypes.bad_request_error,
                    param="callback_name",
                )

            team_callback_settings_obj.success_callback.append(data.callback_name)
            team_callback_settings_obj.failure_callback.append(data.callback_name)
        for var, value in data.callback_vars.items():
            if team_callback_settings_obj.callback_vars is None:
                team_callback_settings_obj.callback_vars = {}
            team_callback_settings_obj.callback_vars[var] = value

        team_callback_settings_obj_dict = team_callback_settings_obj.model_dump()

        team_metadata["callback_settings"] = team_callback_settings_obj_dict
        team_metadata_json = json.dumps(team_metadata)  # update team_metadata

        new_team_row = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        return {
            "status": "success",
            "data": new_team_row,
        }

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.add_team_callbacks(): Exception occured - {}".format(
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


@router.delete(
    "/team/{team_id:path}/callback/{callback_name}",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def delete_team_callback(
    http_request: Request,
    team_id: str,
    callback_name: str,
    callback_type: Optional[str] = Query(
        "success_and_failure",
        description="The type of callback to remove. Options: success, failure, success_and_failure"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Remove a success/failure callback from a team
    
    Parameters:
    - team_id (str, required): The unique identifier for the team
    - callback_name (str, required): The name of the callback to remove
    - callback_type (str, optional): The type of callback to remove. Options: "success", "failure", "success_and_failure" (default)
    
    Example curl:
    ```
    curl -X DELETE 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback/langfuse?callback_type=success_and_failure' \
        -H 'Authorization: Bearer sk-1234'
    ```
    
    This will remove the langfuse callback from both success and failure callbacks for the team with id dbe2f686-a686-4896-864a-4c3924458709
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

        # Get team callback settings from metadata
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)

        # Track changes for the response
        changes_made = {
            "success_callback_removed": False,
            "failure_callback_removed": False,
            "vars_removed": []
        }

        # Remove callback based on callback_type
        if callback_type in ["success", "success_and_failure"]:
            if team_callback_settings_obj.success_callback is not None and callback_name in team_callback_settings_obj.success_callback:
                team_callback_settings_obj.success_callback.remove(callback_name)
                changes_made["success_callback_removed"] = True
            
        if callback_type in ["failure", "success_and_failure"]:
            if team_callback_settings_obj.failure_callback is not None and callback_name in team_callback_settings_obj.failure_callback:
                team_callback_settings_obj.failure_callback.remove(callback_name)
                changes_made["failure_callback_removed"] = True

        # If no changes were made, the callback doesn't exist
        if not any([changes_made["success_callback_removed"], changes_made["failure_callback_removed"]]):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Callback '{callback_name}' with type '{callback_type}' not found for team_id = {team_id}"}
            )

        # Also clean up associated callback variables
        if team_callback_settings_obj.callback_vars is not None:
            # Define the variable prefixes for each callback
            callback_var_prefixes = {
                "langfuse": ["langfuse_public_key", "langfuse_secret", "langfuse_secret_key", "langfuse_host"],
                "gcs": ["gcs_bucket_name", "gcs_path_service_account"],
                "langsmith": ["langsmith_api_key", "langsmith_project", "langsmith_base_url"],
                "humanloop": ["humanloop_api_key"],
                "arize": ["arize_api_key", "arize_space_key"],
            }
            
            # Get the variable keys for this callback type
            var_keys_to_remove = callback_var_prefixes.get(callback_name, [])
            
            # Check if this callback is still used in either success or failure lists
            still_in_use = False
            if callback_name in (team_callback_settings_obj.success_callback or []):
                still_in_use = True
            if callback_name in (team_callback_settings_obj.failure_callback or []):
                still_in_use = True
                
            # Only remove vars if the callback is completely removed
            if not still_in_use:
                for var_key in var_keys_to_remove:
                    if var_key in team_callback_settings_obj.callback_vars:
                        changes_made["vars_removed"].append(var_key)
                        team_callback_settings_obj.callback_vars.pop(var_key)

        # Update team in database
        team_metadata["callback_settings"] = team_callback_settings_obj.model_dump()
        team_metadata_json = json.dumps(team_metadata)
        
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist. Error updating team callback"},
            )

        return {
            "status": "success",
            "message": f"Callback '{callback_name}' removed for team {team_id}",
            "data": {
                "team_id": updated_team.team_id,
                "callback_name": callback_name,
                "callback_type": callback_type,
                "changes": changes_made,
                "current_success_callbacks": team_callback_settings_obj.success_callback,
                "current_failure_callbacks": team_callback_settings_obj.failure_callback,
            },
        }

    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.proxy_server.delete_team_callback(): Exception occurred - {str(e)}"
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


@router.patch(
    "/team/{team_id:path}/callback/{callback_name}",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def update_team_callback(
    data: AddTeamCallback,
    http_request: Request,
    team_id: str,
    callback_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Update an existing callback for a team
    
    Parameters:
    - team_id (str, required): The unique identifier for the team
    - callback_name (str, required): The name of the callback to update
    - callback_type (Literal["success", "failure", "success_and_failure"], required): The type of callback to set
    - callback_vars (StandardCallbackDynamicParams, required): A dictionary of variables to pass to the callback
    
    Example curl:
    ```
    curl -X PATCH 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback/langfuse' \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer sk-1234' \
        -d '{
        "callback_name": "langfuse",
        "callback_type": "success_and_failure",
        "callback_vars": {"langfuse_public_key": "pk-lf-updated", "langfuse_secret_key": "sk-updated"}
        }'
    ```
    
    This will update the langfuse callback for the team with id dbe2f686-a686-4896-864a-4c3924458709
    to use the new API keys and set it to trigger on both success and failure events.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})
            
        # Verify callback_name in path matches the one in the request body
        if callback_name != data.callback_name:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Callback name in path '{callback_name}' does not match callback name in request body '{data.callback_name}'"}
            )

        # Check if team exists
        _existing_team = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist."},
            )

        # Get team callback settings from metadata
        team_metadata = _existing_team.metadata
        team_callback_settings = team_metadata.get("callback_settings", {})
        team_callback_settings_obj = TeamCallbackMetadata(**team_callback_settings)
        
        # Track changes for response
        changes = {
            "callback_type_updated": False,
            "callback_vars_updated": [],
            "added_to_success": False,
            "added_to_failure": False,
            "removed_from_success": False,
            "removed_from_failure": False
        }
        
        # Update callback based on callback_type
        # First, check if the callback exists in either list
        callback_in_success = callback_name in (team_callback_settings_obj.success_callback or [])
        callback_in_failure = callback_name in (team_callback_settings_obj.failure_callback or [])
        
        # Then update according to the requested type
        if data.callback_type == "success":
            # Add to success if not already there
            if not callback_in_success:
                if team_callback_settings_obj.success_callback is None:
                    team_callback_settings_obj.success_callback = []
                team_callback_settings_obj.success_callback.append(callback_name)
                changes["added_to_success"] = True
                
            # Remove from failure if it's there
            if callback_in_failure:
                team_callback_settings_obj.failure_callback.remove(callback_name)
                changes["removed_from_failure"] = True
                
        elif data.callback_type == "failure":
            # Add to failure if not already there
            if not callback_in_failure:
                if team_callback_settings_obj.failure_callback is None:
                    team_callback_settings_obj.failure_callback = []
                team_callback_settings_obj.failure_callback.append(callback_name)
                changes["added_to_failure"] = True
                
            # Remove from success if it's there
            if callback_in_success:
                team_callback_settings_obj.success_callback.remove(callback_name)
                changes["removed_from_success"] = True
                
        elif data.callback_type == "success_and_failure":
            # Add to success if not already there
            if not callback_in_success:
                if team_callback_settings_obj.success_callback is None:
                    team_callback_settings_obj.success_callback = []
                team_callback_settings_obj.success_callback.append(callback_name)
                changes["added_to_success"] = True
                
            # Add to failure if not already there
            if not callback_in_failure:
                if team_callback_settings_obj.failure_callback is None:
                    team_callback_settings_obj.failure_callback = []
                team_callback_settings_obj.failure_callback.append(callback_name)
                changes["added_to_failure"] = True
        
        # If any changes were made to the callback lists
        if any([changes["added_to_success"], changes["added_to_failure"], 
                changes["removed_from_success"], changes["removed_from_failure"]]):
            changes["callback_type_updated"] = True
        
        # Update callback variables
        # Define the variable prefixes for each callback to know which ones to update
        callback_var_prefixes = {
            "langfuse": ["langfuse_public_key", "langfuse_secret", "langfuse_secret_key", "langfuse_host"],
            "gcs": ["gcs_bucket_name", "gcs_path_service_account"],
            "langsmith": ["langsmith_api_key", "langsmith_project", "langsmith_base_url"],
            "humanloop": ["humanloop_api_key"],
            "arize": ["arize_api_key", "arize_space_key"],
        }
        
        # Get the variable keys for this callback type
        relevant_var_keys = callback_var_prefixes.get(callback_name, [])
        
        # Initialize callback_vars if it doesn't exist
        if team_callback_settings_obj.callback_vars is None:
            team_callback_settings_obj.callback_vars = {}
            
        # Update only the variables related to this callback
        for var_name, var_value in data.callback_vars.items():
            # Only update if this is a relevant variable for this callback
            if var_name in relevant_var_keys:
                # Check if it's a new variable or an update
                old_value = team_callback_settings_obj.callback_vars.get(var_name)
                if old_value != var_value:
                    team_callback_settings_obj.callback_vars[var_name] = var_value
                    changes["callback_vars_updated"].append(var_name)
        
        # If no changes were made at all, return a message
        if not (changes["callback_type_updated"] or changes["callback_vars_updated"]):
            return {
                "status": "success",
                "message": f"No changes were needed for callback '{callback_name}' for team {team_id}",
                "data": {
                    "team_id": team_id,
                    "callback_name": callback_name,
                    "current_success_callbacks": team_callback_settings_obj.success_callback,
                    "current_failure_callbacks": team_callback_settings_obj.failure_callback,
                    "current_vars": {
                        var_name: team_callback_settings_obj.callback_vars.get(var_name)
                        for var_name in relevant_var_keys
                        if var_name in team_callback_settings_obj.callback_vars
                    }
                }
            }
        
        # Update team in database
        team_metadata["callback_settings"] = team_callback_settings_obj.model_dump()
        team_metadata_json = json.dumps(team_metadata)
        
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": team_id}, data={"metadata": team_metadata_json}  # type: ignore
        )

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team id = {team_id} does not exist. Error updating team callback"},
            )

        return {
            "status": "success",
            "message": f"Callback '{callback_name}' updated for team {team_id}",
            "data": {
                "team_id": updated_team.team_id,
                "callback_name": callback_name,
                "changes": changes,
                "current_success_callbacks": team_callback_settings_obj.success_callback,
                "current_failure_callbacks": team_callback_settings_obj.failure_callback,
                "updated_vars": {
                    var_name: team_callback_settings_obj.callback_vars.get(var_name)
                    for var_name in changes["callback_vars_updated"]
                },
            },
        }

    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.proxy_server.update_team_callback(): Exception occurred - {str(e)}"
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
