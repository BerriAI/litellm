import asyncio
import copy
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Literal, Optional, Union, cast

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    EnterpriseLicenseData,
    Litellm_EntityType,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.health_check import (
    _clean_endpoint_data,
    _update_litellm_params_for_health_check,
    perform_health_check,
    run_with_timeout,
)
from litellm.secret_managers.main import get_secret
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

#### Health ENDPOINTS ####


def _resolve_os_environ_variables(params: dict) -> dict:
    """
    Resolve ``os.environ/`` environment variables in ``litellm_params``.

    This walks the input dict/list structure iteratively (no Python recursion) to
    avoid unbounded recursion / stack overflows on deeply nested inputs.
    """
    if not isinstance(params, dict):
        return params

    # Use an explicit stack to avoid recursion and handle nested dicts/lists.
    # We also keep a `seen` set to guard against accidental cycles.
    resolved_root: dict = {}
    stack: list[tuple[object, object]] = [(params, resolved_root)]
    seen: set[int] = {id(params)}

    while stack:
        src, dst = stack.pop()

        if isinstance(src, dict) and isinstance(dst, dict):
            for key, value in src.items():
                # Direct string replacement for os.environ/ references
                if isinstance(value, str) and value.startswith("os.environ/"):
                    dst[key] = get_secret(value)
                elif isinstance(value, dict):
                    if id(value) in seen:
                        # Cycle detected – keep a shallow copy reference to prevent infinite loops
                        dst[key] = {}
                        continue
                    seen.add(id(value))
                    new_dict: dict = {}
                    dst[key] = new_dict
                    stack.append((value, new_dict))
                elif isinstance(value, list):
                    if id(value) in seen:
                        dst[key] = []
                        continue
                    seen.add(id(value))
                    new_list: list = []
                    dst[key] = new_list
                    stack.append((value, new_list))
                else:
                    dst[key] = value

        elif isinstance(src, list) and isinstance(dst, list):
            for item in src:
                if isinstance(item, str) and item.startswith("os.environ/"):
                    dst.append(get_secret(item))
                elif isinstance(item, dict):
                    if id(item) in seen:
                        dst.append({})
                        continue
                    seen.add(id(item))
                    new_dict = {}
                    dst.append(new_dict)
                    stack.append((item, new_dict))
                elif isinstance(item, list):
                    if id(item) in seen:
                        dst.append([])
                        continue
                    seen.add(id(item))
                    new_list = []
                    dst.append(new_list)
                    stack.append((item, new_list))
                else:
                    dst.append(item)

    return resolved_root


def get_callback_identifier(callback):
    """
    Get the callback identifier string, handling both strings and objects.
    
    This function extracts a string identifier from a callback, which can be:
    - A string (returned as-is)
    - An object with a callback_name attribute
    - An object registered in CustomLoggerRegistry
    - Falls back to callback_name() helper function
    
    Args:
        callback: The callback to identify (can be str or object)
        
    Returns:
        str: The callback identifier string
    """
    if isinstance(callback, str):
        return callback
    if hasattr(callback, 'callback_name') and callback.callback_name:
        return callback.callback_name
    if hasattr(callback, '__class__'):
        callback_strs = CustomLoggerRegistry.get_all_callback_strs_from_class_type(callback.__class__)
        if hasattr(callback, 'callback_name') and callback.callback_name in callback_strs:
            return callback.callback_name
        if callback_strs:
            return callback_strs[0]
    return callback_name(callback)


router = APIRouter()
services = Union[
    Literal[
        "slack_budget_alerts",
        "langfuse",
        "langfuse_otel",
        "slack",
        "openmeter",
        "webhook",
        "email",
        "braintrust",
        "datadog",
        "datadog_llm_observability",
        "generic_api",
        "arize",
        "sqs"
    ],
    str,
]


@router.get(
    "/test",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_endpoint(request: Request):
    """
    [DEPRECATED] use `/health/liveliness` instead.

    A test endpoint that pings the proxy server to check if it's healthy.

    Parameters:
        request (Request): The incoming request.

    Returns:
        dict: A dictionary containing the route of the request URL.
    """
    # ping the proxy server to check if its healthy
    return {"route": request.url.path}


@router.get(
    "/health/services",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_services_endpoint(  # noqa: PLR0915
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    service: services = fastapi.Query(description="Specify the service being hit."),
):
    """
    Use this admin-only endpoint to check if the service is healthy.

    Example:
    ```
    curl -L -X GET 'http://0.0.0.0:4000/health/services?service=datadog' \
    -H 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        from litellm.proxy.proxy_server import (
            general_settings,
            prisma_client,
            proxy_logging_obj,
        )

        if service is None:
            raise HTTPException(
                status_code=400, detail={"error": "Service must be specified."}
            )

        if service not in [
            "slack_budget_alerts",
            "email",
            "langfuse",
            "langfuse_otel",
            "slack",
            "openmeter",
            "webhook",
            "braintrust",
            "otel",
            "custom_callback_api",
            "langsmith",
            "datadog",
            "datadog_llm_observability",
            "generic_api",
            "arize",
            "sqs"
        ]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Service must be in list. Service={service} not in {services}"
                },
            )

        service_in_success_callbacks = False
        if service in litellm.success_callback:
            service_in_success_callbacks = True
        else:
            for cb in litellm.success_callback:
                if hasattr(cb, 'callback_name') and cb.callback_name == service:
                    service_in_success_callbacks = True
                    break
                cb_id = get_callback_identifier(cb)
                if cb_id == service:
                    service_in_success_callbacks = True
                    break
        
        if (
            service == "openmeter"
            or service == "braintrust"
            or service == "generic_api"
            or (service_in_success_callbacks and service != "langfuse")
        ):
            _ = await litellm.acompletion(
                model="openai/litellm-mock-response-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                user="litellm:/health/services",
                mock_response="This is a mock response",
            )
            return {
                "status": "success",
                "message": "Mock LLM request made - check {}.".format(service),
            }
        elif service == "datadog":
            from litellm.integrations.datadog.datadog import DataDogLogger

            datadog_logger = DataDogLogger()
            response = await datadog_logger.async_health_check()
            return {
                "status": response["status"],
                "message": (
                    response["error_message"]
                    if response["status"] == "unhealthy"
                    else "Datadog is healthy"
                ),
            }
        elif service == "arize":
            from litellm.integrations.arize.arize import ArizeLogger

            arize_logger = ArizeLogger()
            response = await arize_logger.async_health_check()
            return {
                "status": response["status"],
                "message": (
                    response["error_message"]
                    if response["status"] == "unhealthy"
                    else "Arize is healthy"
                ),
            }
        elif service == "langfuse":
            from litellm.integrations.langfuse.langfuse import LangFuseLogger

            langfuse_logger = LangFuseLogger()
            langfuse_logger.Langfuse.auth_check()
            _ = litellm.completion(
                model="openai/litellm-mock-response-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                user="litellm:/health/services",
                mock_response="This is a mock response",
            )
            return {
                "status": "success",
                "message": "Mock LLM request made - check langfuse.",
            }

        if service == "webhook":
            user_info = CallInfo(
                token=user_api_key_dict.token or "",
                spend=1,
                max_budget=0,
                user_id=user_api_key_dict.user_id,
                key_alias=user_api_key_dict.key_alias,
                team_id=user_api_key_dict.team_id,
                event_group=Litellm_EntityType.KEY,
            )
            await proxy_logging_obj.budget_alerts(
                type="user_budget",
                user_info=user_info,
            )
        elif service == "sqs":
            from litellm.integrations.sqs import SQSLogger
            sqs_logger = SQSLogger()
            response = await sqs_logger.async_health_check()
            return {
                "status": response["status"],
                "message": response["error_message"],
            }

        if service == "slack" or service == "slack_budget_alerts":
            if "slack" in general_settings.get("alerting", []):
                # test_message = f"""\n🚨 `ProjectedLimitExceededError` 💸\n\n`Key Alias:` litellm-ui-test-alert \n`Expected Day of Error`: 28th March \n`Current Spend`: $100.00 \n`Projected Spend at end of month`: $1000.00 \n`Soft Limit`: $700"""
                # check if user has opted into unique_alert_webhooks
                if (
                    proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url
                    is not None
                ):
                    for (
                        alert_type
                    ) in proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url:
                        # only test alert if it's in active alert types
                        if (
                            proxy_logging_obj.slack_alerting_instance.alert_types
                            is not None
                            and alert_type
                            not in proxy_logging_obj.slack_alerting_instance.alert_types
                        ):
                            continue

                        test_message = "default test message"
                        if alert_type == AlertType.llm_exceptions:
                            test_message = "LLM Exception test alert"
                        elif alert_type == AlertType.llm_too_slow:
                            test_message = "LLM Too Slow test alert"
                        elif alert_type == AlertType.llm_requests_hanging:
                            test_message = "LLM Requests Hanging test alert"
                        elif alert_type == AlertType.budget_alerts:
                            test_message = "Budget Alert test alert"
                        elif alert_type == AlertType.db_exceptions:
                            test_message = "DB Exception test alert"
                        elif alert_type == AlertType.outage_alerts:
                            test_message = "Outage Alert Exception test alert"
                        elif alert_type == AlertType.daily_reports:
                            test_message = "Daily Reports test alert"
                        else:
                            test_message = "Budget Alert test alert"

                        await proxy_logging_obj.alerting_handler(
                            message=test_message, level="Low", alert_type=alert_type
                        )
                else:
                    await proxy_logging_obj.alerting_handler(
                        message="This is a test slack alert message",
                        level="Low",
                        alert_type=AlertType.budget_alerts,
                    )

                if prisma_client is not None:
                    asyncio.create_task(
                        proxy_logging_obj.slack_alerting_instance.send_monthly_spend_report()
                    )
                    asyncio.create_task(
                        proxy_logging_obj.slack_alerting_instance.send_weekly_spend_report()
                    )

                alert_types = (
                    proxy_logging_obj.slack_alerting_instance.alert_types or []
                )
                alert_types = list(alert_types)
                return {
                    "status": "success",
                    "alert_types": alert_types,
                    "message": "Mock Slack Alert sent, verify Slack Alert Received on your channel",
                }
            else:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": '"{}" not in proxy config: general_settings. Unable to test this.'.format(
                            service
                        )
                    },
                )
        if service == "email":
            webhook_event = WebhookEvent(
                event="key_created",
                event_group=Litellm_EntityType.KEY,
                event_message="Test Email Alert",
                token=user_api_key_dict.token or "",
                key_alias="Email Test key (This is only a test alert key. DO NOT USE THIS IN PRODUCTION.)",
                spend=0,
                max_budget=0,
                user_id=user_api_key_dict.user_id,
                user_email=os.getenv("TEST_EMAIL_ADDRESS"),
                team_id=user_api_key_dict.team_id,
            )

            # use create task - this can take 10 seconds. don't keep ui users waiting for notification to check their email
            await proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                webhook_event=webhook_event
            )

            return {
                "status": "success",
                "message": "Mock Email Alert sent, verify Email Alert Received",
            }

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.health_services_endpoint(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _convert_health_check_to_dict(check) -> dict:
    """Convert health check database record to dictionary format"""
    return {
        "health_check_id": check.health_check_id,
        "model_name": check.model_name,
        "model_id": check.model_id,
        "status": check.status,
        "healthy_count": check.healthy_count,
        "unhealthy_count": check.unhealthy_count,
        "error_message": check.error_message,
        "response_time_ms": check.response_time_ms,
        "details": check.details,
        "checked_by": check.checked_by,
        "checked_at": check.checked_at.isoformat() if check.checked_at else None,
        "created_at": check.created_at.isoformat() if check.created_at else None,
    }


def _check_prisma_client():
    """Helper to check if prisma_client is available and raise appropriate error"""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Database not initialized"},
        )
    return prisma_client


async def _save_health_check_to_db(
    prisma_client,
    model_name: str,
    healthy_endpoints: list,
    unhealthy_endpoints: list,
    start_time: float,
    user_id: Optional[str],
    model_id: Optional[str] = None,
):
    """Helper function to save health check results to database"""
    try:
        # Extract error message from first unhealthy endpoint if available
        error_message = (
            str(unhealthy_endpoints[0]["error"])[:500]
            if unhealthy_endpoints and unhealthy_endpoints[0].get("error")
            else None
        )

        await prisma_client.save_health_check_result(
            model_name=model_name,
            model_id=model_id,
            status="healthy" if healthy_endpoints else "unhealthy",
            healthy_count=len(healthy_endpoints),
            unhealthy_count=len(unhealthy_endpoints),
            error_message=error_message,
            response_time_ms=(time.time() - start_time) * 1000,
            details=None,  # Skip details for now to avoid JSON serialization issues
            checked_by=user_id,
        )
    except Exception as db_error:
        verbose_proxy_logger.warning(
            f"Failed to save health check to database for model {model_name}: {db_error}"
        )
        # Continue execution - don't let database save failure break health checks


def _build_model_param_to_info_mapping(model_list: list) -> dict:
    """
    Build a mapping from model parameter to model info (model_name, model_id).
    
    Multiple models might share the same model parameter, so we use a list.
    
    Args:
        model_list: List of model configurations
        
    Returns:
        Dictionary mapping model parameter to list of model info dicts
    """
    model_param_to_info: dict = {}
    for model in model_list:
        model_info = model.get("model_info", {})
        model_name = model.get("model_name")
        model_id = model_info.get("id")
        litellm_params = model.get("litellm_params", {})
        model_param = litellm_params.get("model")
        
        if model_param and model_name:
            if model_param not in model_param_to_info:
                model_param_to_info[model_param] = []
            model_param_to_info[model_param].append({
                "model_name": model_name,
                "model_id": model_id,
            })
    return model_param_to_info


def _aggregate_health_check_results(
    model_param_to_info: dict,
    healthy_endpoints: list,
    unhealthy_endpoints: list,
) -> dict:
    """
    Aggregate health check results per unique model.
    
    Uses (model_id, model_name) as key, or (None, model_name) if model_id is None.
    
    Args:
        model_param_to_info: Mapping from model parameter to model info
        healthy_endpoints: List of healthy endpoint results
        unhealthy_endpoints: List of unhealthy endpoint results
        
    Returns:
        Dictionary mapping (model_id, model_name) to aggregated health check results
    """
    model_results = {}
    
    # Process healthy endpoints
    for endpoint in healthy_endpoints:
        model_param = endpoint.get("model")
        if model_param and model_param in model_param_to_info:
            for model_info in model_param_to_info[model_param]:
                key = (model_info["model_id"], model_info["model_name"])
                if key not in model_results:
                    model_results[key] = {
                        "model_name": model_info["model_name"],
                        "model_id": model_info["model_id"],
                        "healthy_count": 0,
                        "unhealthy_count": 0,
                        "error_message": None,
                    }
                model_results[key]["healthy_count"] += 1
    
    # Process unhealthy endpoints
    for endpoint in unhealthy_endpoints:
        model_param = endpoint.get("model")
        error_message = endpoint.get("error")
        if model_param and model_param in model_param_to_info:
            for model_info in model_param_to_info[model_param]:
                key = (model_info["model_id"], model_info["model_name"])
                if key not in model_results:
                    model_results[key] = {
                        "model_name": model_info["model_name"],
                        "model_id": model_info["model_id"],
                        "healthy_count": 0,
                        "unhealthy_count": 0,
                        "error_message": None,
                    }
                model_results[key]["unhealthy_count"] += 1
                # Use the first error message encountered
                if not model_results[key]["error_message"] and error_message:
                    model_results[key]["error_message"] = str(error_message)[:500]
    
    return model_results


async def _save_health_check_results_if_changed(
    prisma_client,
    model_results: dict,
    latest_checks_map: dict,
    start_time: float,
    checked_by: Optional[str] = None,
):
    """
    Save health check results to database, but only if status changed or >1 hour since last save.
    
    OPTIMIZATION: Only saves to database if the status has changed from the last saved check.
    This dramatically reduces database writes when health status remains stable.
    
    - Stable systems: ~1 write/hour per model (instead of 12 writes/hour with 5-min intervals)
    - Status changes: Immediate write (no delay)
    - Result: ~92% reduction in DB writes for stable systems, while maintaining real-time updates on changes
    
    Args:
        prisma_client: Database client
        model_results: Dictionary of aggregated health check results per model
        latest_checks_map: Dictionary mapping model_id/model_name to latest health check
        start_time: Start time of health check for calculating response time
        checked_by: Identifier for who/what performed the check
    """
    for result in model_results.values():
        new_status = "healthy" if result["healthy_count"] > 0 else "unhealthy"
        
        # Check if we should save this result
        should_save = True
        lookup_key = result["model_id"] if result["model_id"] else result["model_name"]
        if lookup_key in latest_checks_map:
            last_check = latest_checks_map[lookup_key]
            # Only save if status changed or if it's been a while since last check
            if last_check.status == new_status:
                # Check if last check was recent (within 1 hour)
                if last_check.checked_at:
                    from datetime import datetime, timezone
                    time_since_last_check = (
                        datetime.now(timezone.utc) - last_check.checked_at
                    ).total_seconds()
                    # Only skip if status unchanged AND checked recently (within 1 hour)
                    # This ensures we still get periodic updates even if status is stable
                    if time_since_last_check < 3600:  # 1 hour threshold
                        should_save = False
        
        if should_save:
            asyncio.create_task(
                prisma_client.save_health_check_result(
                    model_name=result["model_name"],
                    model_id=result["model_id"],
                    status=new_status,
                    healthy_count=result["healthy_count"],
                    unhealthy_count=result["unhealthy_count"],
                    error_message=result["error_message"],
                    response_time_ms=(time.time() - start_time) * 1000,
                    details=None,
                    checked_by=checked_by,
                )
            )


async def _save_background_health_checks_to_db(
    prisma_client,
    model_list: list,
    healthy_endpoints: list,
    unhealthy_endpoints: list,
    start_time: float,
    checked_by: Optional[str] = None,
):
    """
    Save background health check results to database for each model.
    
    Maps health check endpoints back to their original models to get model_name and model_id.
    Aggregates results per unique model (by model_id if available, otherwise model_name).
    
    OPTIMIZATION: Only saves to database if the status has changed from the last saved check.
    This dramatically reduces database writes when health status remains stable.
    """
    if prisma_client is None:
        return
    
    try:
        # Step 1: Build mapping from model parameter to model info
        model_param_to_info = _build_model_param_to_info_mapping(model_list)
        
        # Step 2: Aggregate health check results per unique model
        model_results = _aggregate_health_check_results(
            model_param_to_info,
            healthy_endpoints,
            unhealthy_endpoints,
        )
        
        # Step 3: Get latest health checks for all models in one query to compare status
        latest_checks = await prisma_client.get_all_latest_health_checks()
        latest_checks_map = {}
        for check in latest_checks:
            # Use model_id as primary key, fallback to model_name
            key = check.model_id if check.model_id else check.model_name
            if key not in latest_checks_map:
                latest_checks_map[key] = check
        
        # Step 4: Save aggregated results, but only if status changed
        await _save_health_check_results_if_changed(
            prisma_client,
            model_results,
            latest_checks_map,
            start_time,
            checked_by,
        )
    except Exception as db_error:
        verbose_proxy_logger.warning(
            f"Failed to save background health checks to database: {db_error}"
        )
        # Continue execution - don't let database save failure break health checks


async def _perform_health_check_and_save(
    model_list,
    target_model,
    cli_model,
    details,
    prisma_client,
    start_time,
    user_id,
    model_id=None,
):
    """Helper function to perform health check and save results to database"""
    healthy_endpoints, unhealthy_endpoints = await perform_health_check(
        model_list=model_list, cli_model=cli_model, model=target_model, details=details
    )

    # Optionally save health check result to database (non-blocking)
    if prisma_client is not None:
        # For CLI model, use cli_model name; for router models, use target_model
        model_name_for_db = cli_model if cli_model is not None else target_model
        if model_name_for_db is not None:
            asyncio.create_task(
                _save_health_check_to_db(
                    prisma_client,
                    model_name_for_db,
                    healthy_endpoints,
                    unhealthy_endpoints,
                    start_time,
                    user_id,
                    model_id=model_id,
                )
            )

    return {
        "healthy_endpoints": healthy_endpoints,
        "unhealthy_endpoints": unhealthy_endpoints,
        "healthy_count": len(healthy_endpoints),
        "unhealthy_count": len(unhealthy_endpoints),
    }


@router.get("/health", tags=["health"], dependencies=[Depends(user_api_key_auth)])
async def health_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
    ),
    model_id: Optional[str] = fastapi.Query(
        None, description="Specify the model ID (optional)"
    ),
):
    """
    🚨 USE `/health/liveliness` to health check the proxy 🚨

    See more 👉 https://docs.litellm.ai/docs/proxy/health


    Check the health of all the endpoints in config.yaml

    To run health checks in the background, add this to config.yaml:
    ```
    general_settings:
        # ... other settings
        background_health_checks: True
    ```
    else, the health checks will be run on models when /health is called.
    """
    import time

    from litellm.proxy.proxy_server import (
        health_check_details,
        health_check_results,
        llm_model_list,
        llm_router,
        prisma_client,
        use_background_health_checks,
        user_model,
    )

    start_time = time.time()

    # Handle model_id parameter - convert to model name for health check
    target_model = model
    if model_id and not model:
        # Use get_deployment from router to find the model name
        if llm_router is not None:
            try:
                deployment = llm_router.get_deployment(model_id=model_id)
                if deployment is not None:
                    target_model = deployment.model_name
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": f"Model with ID {model_id} not found"},
                    )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error getting deployment for model_id {model_id}: {e}"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": f"Model with ID {model_id} not found"},
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Model with ID {model_id} not found"},
            )

    try:
        if llm_model_list is None:
            # if no router set, check if user set a model using litellm --model ollama/llama2
            if user_model is not None:
                return await _perform_health_check_and_save(
                    model_list=[],
                    target_model=None,
                    cli_model=user_model,
                    details=health_check_details,
                    prisma_client=prisma_client,
                    start_time=start_time,
                    user_id=user_api_key_dict.user_id,
                    model_id=None,  # CLI model doesn't have model_id
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Model list not initialized"},
            )
        _llm_model_list = copy.deepcopy(llm_model_list)
        ### FILTER MODELS FOR ONLY THOSE USER HAS ACCESS TO ###
        if len(user_api_key_dict.models) > 0:
            pass
        else:
            pass  #
        if use_background_health_checks:
            return health_check_results
        else:
            return await _perform_health_check_and_save(
                model_list=_llm_model_list,
                target_model=target_model,
                cli_model=None,
                details=health_check_details,
                prisma_client=prisma_client,
                start_time=start_time,
                user_id=user_api_key_dict.user_id,
                model_id=model_id,
            )
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.py::health_endpoint(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e


@router.get(
    "/health/history", tags=["health"], dependencies=[Depends(user_api_key_auth)]
)
async def health_check_history_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Filter by specific model name"
    ),
    status_filter: Optional[str] = fastapi.Query(
        None, description="Filter by status (healthy/unhealthy)"
    ),
    limit: int = fastapi.Query(
        100, description="Number of records to return", ge=1, le=1000
    ),
    offset: int = fastapi.Query(0, description="Number of records to skip", ge=0),
):
    """
    Get health check history for models

    Returns historical health check data with optional filtering.
    """
    prisma_client = _check_prisma_client()

    try:
        history = await prisma_client.get_health_check_history(
            model_name=model,
            limit=limit,
            offset=offset,
            status_filter=status_filter,
        )

        # Convert to dict format for JSON response using helper function
        history_data = [_convert_health_check_to_dict(check) for check in history]

        return {
            "health_checks": history_data,
            "total_records": len(history_data),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting health check history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to retrieve health check history: {str(e)}"},
        )


@router.get(
    "/health/latest", tags=["health"], dependencies=[Depends(user_api_key_auth)]
)
async def latest_health_checks_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the latest health check status for all models

    Returns the most recent health check result for each model.
    """
    prisma_client = _check_prisma_client()

    try:
        latest_checks = await prisma_client.get_all_latest_health_checks()

        # Convert to dict format for JSON response using helper function
        checks_data = {
            (
                check.model_id if check.model_id else check.model_name
            ): _convert_health_check_to_dict(check)
            for check in latest_checks
        }

        return {
            "latest_health_checks": checks_data,
            "total_models": len(checks_data),
        }
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting latest health checks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to retrieve latest health checks: {str(e)}"},
        )


@router.get(
    "/health/shared-status", tags=["health"], dependencies=[Depends(user_api_key_auth)]
)
async def shared_health_check_status_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the status of shared health check coordination across pods.

    Returns information about Redis connectivity, lock status, and cache status.
    """
    from litellm.proxy.proxy_server import redis_usage_cache, use_shared_health_check

    if not use_shared_health_check:
        return {
            "shared_health_check_enabled": False,
            "message": "Shared health check is not enabled",
        }

    if redis_usage_cache is None:
        return {
            "shared_health_check_enabled": True,
            "redis_available": False,
            "message": "Redis is not configured",
        }

    try:
        from litellm.proxy.health_check_utils.shared_health_check_manager import (
            SharedHealthCheckManager,
        )

        shared_health_manager = SharedHealthCheckManager(
            redis_cache=redis_usage_cache,
        )

        health_status = await shared_health_manager.get_health_check_status()
        return {"shared_health_check_enabled": True, "status": health_status}
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting shared health check status: {e}")
        raise HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Failed to retrieve shared health check status: {str(e)}"
            },
        )


def _read_license_data() -> Optional[Dict[str, Any]]:
    from litellm.proxy.proxy_server import (
        _license_check,
        premium_user_data,
    )

    license_data: Optional[EnterpriseLicenseData] = (
        premium_user_data or _license_check.airgapped_license_data
    )

    if (
        license_data is None
        and getattr(_license_check, "license_str", None)
        and getattr(_license_check, "public_key", None)
    ):
        try:
            verification_result = _license_check.verify_license_without_api_request(
                public_key=_license_check.public_key,
                license_key=_license_check.license_str,
            )
            if verification_result is True:
                license_data = _license_check.airgapped_license_data
        except Exception:
            pass

    if license_data is None:
        return None
    return cast(Dict[str, Any], license_data)


def _read_allowed_features(license_data: Dict[str, Any]) -> list:
    raw_allowed_features = license_data.get("allowed_features")
    if isinstance(raw_allowed_features, list):
        return list(raw_allowed_features)
    if raw_allowed_features is None:
        return []
    return [raw_allowed_features]


@router.get(
    "/health/license",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_license_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return metadata about the configured LiteLLM license without exposing the key."""
    from litellm.proxy.proxy_server import (
        _license_check,
        premium_user,
    )

    license_data = _read_license_data()
    has_license = bool(getattr(_license_check, "license_str", None))
    license_type = "enterprise" if premium_user else "community"

    if license_data is None:
        return {
            "has_license": has_license,
            "license_type": license_type,
            "expiration_date": None,
            "allowed_features": [],
            "limits": {
                "max_users": None,
                "max_teams": None,
            },
        }

    expiration_date = license_data.get("expiration_date")
    max_users = license_data.get("max_users")
    max_teams = license_data.get("max_teams")

    return {
        "has_license": has_license,
        "license_type": license_type,
        "expiration_date": expiration_date,
        "allowed_features": _read_allowed_features(license_data),
        "limits": {
            "max_users": max_users,
            "max_teams": max_teams,
        },
    }


db_health_cache = {"status": "unknown", "last_updated": datetime.now()}


async def _db_health_readiness_check():
    from litellm.proxy.proxy_server import prisma_client

    global db_health_cache

    # Note - Intentionally don't try/except this so it raises an exception when it fails
    try:
        # if timedelta is less than 2 minutes return DB Status
        time_diff = datetime.now() - db_health_cache["last_updated"]
        if db_health_cache["status"] != "unknown" and time_diff < timedelta(minutes=2):
            return db_health_cache

        if prisma_client is None:
            db_health_cache = {"status": "disconnected", "last_updated": datetime.now()}
            return db_health_cache

        await prisma_client.health_check()
        db_health_cache = {"status": "connected", "last_updated": datetime.now()}
        return db_health_cache
    except Exception as e:
        PrismaDBExceptionHandler.handle_db_exception(e)
        return db_health_cache


@router.get(
    "/settings",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/active/callbacks",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def active_callbacks():
    """
    Returns a list of litellm level settings

    This is useful for debugging and ensuring the proxy server is configured correctly.

    Response schema:
    ```
    {
        "alerting": _alerting,
        "litellm.callbacks": litellm_callbacks,
        "litellm.input_callback": litellm_input_callbacks,
        "litellm.failure_callback": litellm_failure_callbacks,
        "litellm.success_callback": litellm_success_callbacks,
        "litellm._async_success_callback": litellm_async_success_callbacks,
        "litellm._async_failure_callback": litellm_async_failure_callbacks,
        "litellm._async_input_callback": litellm_async_input_callbacks,
        "all_litellm_callbacks": all_litellm_callbacks,
        "num_callbacks": len(all_litellm_callbacks),
        "num_alerting": _num_alerting,
        "litellm.request_timeout": litellm.request_timeout,
    }
    ```
    """

    from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

    _alerting = str(general_settings.get("alerting"))
    # get success callbacks

    litellm_callbacks = [str(x) for x in litellm.callbacks]
    litellm_input_callbacks = [str(x) for x in litellm.input_callback]
    litellm_failure_callbacks = [str(x) for x in litellm.failure_callback]
    litellm_success_callbacks = [str(x) for x in litellm.success_callback]
    litellm_async_success_callbacks = [str(x) for x in litellm._async_success_callback]
    litellm_async_failure_callbacks = [str(x) for x in litellm._async_failure_callback]
    litellm_async_input_callbacks = [str(x) for x in litellm._async_input_callback]

    all_litellm_callbacks = (
        litellm_callbacks
        + litellm_input_callbacks
        + litellm_failure_callbacks
        + litellm_success_callbacks
        + litellm_async_success_callbacks
        + litellm_async_failure_callbacks
        + litellm_async_input_callbacks
    )

    alerting = proxy_logging_obj.alerting
    _num_alerting = 0
    if alerting and isinstance(alerting, list):
        _num_alerting = len(alerting)

    return {
        "alerting": _alerting,
        "litellm.callbacks": litellm_callbacks,
        "litellm.input_callback": litellm_input_callbacks,
        "litellm.failure_callback": litellm_failure_callbacks,
        "litellm.success_callback": litellm_success_callbacks,
        "litellm._async_success_callback": litellm_async_success_callbacks,
        "litellm._async_failure_callback": litellm_async_failure_callbacks,
        "litellm._async_input_callback": litellm_async_input_callbacks,
        "all_litellm_callbacks": all_litellm_callbacks,
        "num_callbacks": len(all_litellm_callbacks),
        "num_alerting": _num_alerting,
        "litellm.request_timeout": litellm.request_timeout,
    }


def callback_name(callback):
    if isinstance(callback, str):
        return callback

    try:
        return callback.__name__
    except AttributeError:
        try:
            return callback.__class__.__name__
        except AttributeError:
            return str(callback)


@router.get(
    "/health/readiness",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_readiness():
    """
    Unprotected endpoint for checking if worker can receive requests
    """
    from litellm.proxy.proxy_server import prisma_client, version

    try:
        # get success callback
        success_callback_names = []

        try:
            # this was returning a JSON of the values in some of the callbacks
            # all we need is the callback name, hence we do str(callback)
            success_callback_names = [
                callback_name(x) for x in litellm.success_callback
            ]
        except AttributeError:
            # don't let this block the /health/readiness response, if we can't convert to str -> return litellm.success_callback
            success_callback_names = litellm.success_callback

        # check Cache
        cache_type = None
        if litellm.cache is not None:
            from litellm.caching.caching import RedisSemanticCache

            cache_type = litellm.cache.type

            if isinstance(litellm.cache.cache, RedisSemanticCache):
                # ping the cache
                # TODO: @ishaan-jaff - we should probably not ping the cache on every /health/readiness check
                try:
                    index_info = await litellm.cache.cache._index_info()
                except Exception as e:
                    index_info = "index does not exist - error: " + str(e)
                cache_type = {"type": cache_type, "index_info": index_info}

        # check DB
        if prisma_client is not None:  # if db passed in, check if it's connected
            db_health_status = await _db_health_readiness_check()
            return {
                "status": "healthy",
                "db": "connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
                "use_aiohttp_transport": AsyncHTTPHandler._should_use_aiohttp_transport(),
                **db_health_status,
            }
        else:
            return {
                "status": "healthy",
                "db": "Not connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
                "use_aiohttp_transport": AsyncHTTPHandler._should_use_aiohttp_transport(),
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service Unhealthy ({str(e)})")


@router.get(
    "/health/liveliness",  # Historical LiteLLM name; doesn't match k8s terminology but kept for backwards compatibility
    tags=["health"],
)
@router.get(
    "/health/liveness",  # Kubernetes has "liveness" probes (https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-liveness-command)
    tags=["health"],
)
async def health_liveliness():
    """
    Unprotected endpoint for checking if worker is alive
    """
    return "I'm alive!"


@router.options(
    "/health/readiness",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_readiness_options():
    """
    Options endpoint for health/readiness check.
    """
    response_headers = {
        "Allow": "GET, OPTIONS",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    return Response(headers=response_headers, status_code=200)


@router.options(
    "/health/liveliness",
    tags=["health"],
)
@router.options(
    "/health/liveness",  # Kubernetes has "liveness" probes (https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-liveness-command)
    tags=["health"],
)
async def health_liveliness_options():
    """
    Options endpoint for health/liveliness check.
    """
    response_headers = {
        "Allow": "GET, OPTIONS",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    return Response(headers=response_headers, status_code=200)


@router.post(
    "/health/test_connection",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_model_connection(
    request: Request,
    mode: Optional[
        Literal[
            "chat",
            "completion",
            "embedding",
            "audio_speech",
            "audio_transcription",
            "image_generation",
            "video_generation",
            "batch",
            "rerank",
            "realtime",
            "responses",
            "ocr",
        ]
    ] = fastapi.Body("chat", description="The mode to test the model with"),
    litellm_params: Dict = fastapi.Body(
        None,
        description="Parameters for litellm.completion, litellm.embedding for the health check",
    ),
    model_info: Dict = fastapi.Body(
        None,
        description="Model info for the health check",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test a direct connection to a specific model.
    
    This endpoint allows you to verify if your proxy can successfully connect to a specific model.
    It's useful for troubleshooting model connectivity issues without going through the full proxy routing.
    
    Example:
    ```bash
    # If model is configured in proxy_config.yaml, you only need to specify the model name:
    curl -X POST 'http://localhost:4000/health/test_connection' \\
      -H 'Authorization: Bearer sk-1234' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "litellm_params": {
            "model": "gpt-4o"
        },
        "mode": "chat"
      }'
    
    # The endpoint will automatically use api_key, api_base, etc. from proxy_config.yaml
    
    # You can also override specific params or test with custom credentials:
    curl -X POST 'http://localhost:4000/health/test_connection' \\
      -H 'Authorization: Bearer sk-1234' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": "os.environ/AZURE_OPENAI_API_KEY",
            "api_base": "os.environ/AZURE_OPENAI_ENDPOINT",
            "api_version": "2024-10-21"
        },
        "mode": "chat"
      }'
    ```
    
    Note: 
    - If the model is configured in proxy_config.yaml, credentials (api_key, api_base, etc.) 
      will be automatically loaded from the config (with resolved environment variables).
    - You can override specific params by including them in the request.
    - You can use `os.environ/VARIABLE_NAME` syntax to reference environment variables,
      which will be resolved automatically (same as in proxy_config.yaml).
    
    Returns:
        dict: A dictionary containing the health check result with either success information or error details.
    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.proxy.proxy_server import llm_router, premium_user, prisma_client
    from litellm.types.router import Deployment, LiteLLM_Params

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        
        # Get model name from litellm_params
        request_litellm_params = litellm_params or {}
        model_name = request_litellm_params.get("model")
        
        # Look up model configuration from router if model name is provided
        # This gets the litellm_params from proxy config (with resolved env vars)
        config_litellm_params: dict = {}
        if model_name and llm_router is not None:
            try:
                # First try to find by proxy model_name (e.g., "gpt-4o")
                deployments = llm_router.get_model_list(model_name=model_name)
                
                # If not found, try to find by litellm model name (e.g., "azure/gpt-4o")
                if not deployments or len(deployments) == 0:
                    all_deployments = llm_router.get_model_list(model_name=None)
                    if all_deployments:
                        for deployment in all_deployments:
                            if deployment.get("litellm_params", {}).get("model") == model_name:
                                deployments = [deployment]
                                break
                
                if deployments and len(deployments) > 0:
                    # Use the first deployment's litellm_params as base config
                    # These already have resolved environment variables from proxy config
                    config_litellm_params = dict(deployments[0].get("litellm_params", {}))
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not find model {model_name} in router: {e}. "
                    "Proceeding with request params only."
                )
        
        # Merge: config params (from proxy config) as base, request params override
        # This allows users to override specific params while using config for credentials
        merged_litellm_params = {**config_litellm_params, **request_litellm_params}
        
        # Resolve os.environ/ environment variables in any remaining request params
        # This handles cases where user explicitly passes os.environ/ values to override config
        litellm_params = _resolve_os_environ_variables(merged_litellm_params)
        
        ## Auth check
        await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=Deployment(
                model_name="test_model",
                litellm_params=LiteLLM_Params(**litellm_params),
                model_info=model_info,
            ),
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            premium_user=premium_user,
        )
        # Include health_check_params if provided
        litellm_params = _update_litellm_params_for_health_check(
            model_info={},
            litellm_params=litellm_params,
        )
        mode = mode or litellm_params.pop("mode", None)

        result = await run_with_timeout(
            litellm.ahealth_check(
                model_params=litellm_params,
                mode=mode,
                prompt="test from litellm",
                input=["test from litellm"],
            ),
            HEALTH_CHECK_TIMEOUT_SECONDS,
        )

        # Clean the result for display
        cleaned_result = _clean_endpoint_data(
            {**litellm_params, **result}, details=True
        )

        return {
            "status": "error" if "error" in result else "success",
            "result": cleaned_result,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.debug(
            f"litellm.proxy.health_endpoints.test_model_connection(): Exception occurred - {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to test connection: {str(e)}"},
        )
