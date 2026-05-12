import asyncio
import copy
import logging
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Literal, Optional, Union, cast

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    EnterpriseLicenseData,
    Litellm_EntityType,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.health_check import (
    ADMIN_ONLY_HEALTH_DISPLAY_PARAMS,
    _clean_endpoint_data,
    _update_litellm_params_for_health_check,
    perform_health_check,
    run_with_timeout,
)
from litellm.proxy.middleware.in_flight_requests_middleware import (
    get_in_flight_requests,
)

#### Health ENDPOINTS ####


def _reject_os_environ_references(params: dict) -> None:
    """
    Validate that the provided params do not contain any ``os.environ/``
    references. Values with that prefix are expected to come only from
    server-side configuration (already resolved before reaching here). If a
    request-supplied value still carries the prefix, raise ``HTTPException``.
    """
    if not isinstance(params, dict):
        return

    stack: list[object] = [params]
    seen: set[int] = {id(params)}

    while stack:
        src = stack.pop()
        if isinstance(src, dict):
            values: Iterable[object] = src.values()
        elif isinstance(src, list):
            values = src
        else:
            continue

        for value in values:
            if isinstance(value, str) and value.startswith("os.environ/"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Environment variable references are not permitted in request parameters."
                    },
                )
            if isinstance(value, (dict, list)) and id(value) not in seen:
                seen.add(id(value))
                stack.append(value)


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
    if hasattr(callback, "callback_name") and callback.callback_name:
        return callback.callback_name
    if hasattr(callback, "__class__"):
        callback_strs = CustomLoggerRegistry.get_all_callback_strs_from_class_type(
            callback.__class__
        )
        if (
            hasattr(callback, "callback_name")
            and callback.callback_name in callback_strs
        ):
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
        "sqs",
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
            "datadog_metrics",
            "datadog_llm_observability",
            "generic_api",
            "arize",
            "sqs",
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
                if getattr(cb, "callback_name", None) == service:
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
        elif service == "datadog_metrics":
            from litellm.integrations.datadog.datadog_metrics import (
                DatadogMetricsLogger,
            )
            from litellm.litellm_core_utils.litellm_logging import (
                get_custom_logger_compatible_class,
            )

            datadog_metrics_logger = get_custom_logger_compatible_class(
                "datadog_metrics"
            )
            if datadog_metrics_logger is None:
                datadog_metrics_logger = DatadogMetricsLogger(
                    start_periodic_flush=False
                )
            assert isinstance(datadog_metrics_logger, DatadogMetricsLogger)
            response = await datadog_metrics_logger.async_health_check()
            return {
                "status": response["status"],
                "message": (
                    response["error_message"]
                    if response["status"] == "unhealthy"
                    else "Datadog Metrics is healthy"
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
            model_param_to_info[model_param].append(
                {
                    "model_name": model_name,
                    "model_id": model_id,
                }
            )
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


_PROXY_ADMIN_ROLES = frozenset(
    {
        LitellmUserRoles.PROXY_ADMIN.value,
        # View-only admins are operators (oncall, support); they need the
        # routing fields (api_base, api_version) to diagnose health and tell
        # which provider region a check is hitting. They cannot mutate config
        # so granting them the read-only view is safe.
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    }
)


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """
    Return True if the caller has a proxy-admin role (full or view-only).

    user_role on UserAPIKeyAuth can be either a LitellmUserRoles enum or its
    string value depending on how the auth path constructed the object, so we
    compare against the raw value rather than the enum identity.
    """
    role = user_api_key_dict.user_role
    if role is None:
        return False
    role_value = role.value if hasattr(role, "value") else role
    return role_value in _PROXY_ADMIN_ROLES


def _strip_admin_only_fields_from_health_result(result: dict) -> dict:
    """
    Return a copy of the /health response with provider routing fields
    (``api_base``, ``api_version``) removed from each healthy/unhealthy
    endpoint entry. Used to hide those fields from non-admin callers while
    still showing them which deployments they own and whether each one is
    healthy. Proxy admins receive the unmodified result.
    """
    out = dict(result)
    drop = set(ADMIN_ONLY_HEALTH_DISPLAY_PARAMS)
    for key in ("healthy_endpoints", "unhealthy_endpoints"):
        eps = out.get(key)
        if isinstance(eps, list):
            out[key] = [
                (
                    {k: v for k, v in ep.items() if k not in drop}
                    if isinstance(ep, dict)
                    else ep
                )
                for ep in eps
            ]
    return out


def _resolve_targeted_model_ids(
    model_list: list, model: Optional[str], model_id: Optional[str]
) -> Optional[set]:
    """
    Resolve a ``/health`` ``model`` / ``model_id`` query param to the set of
    deployment IDs the response should be scoped to.

    Mirrors the live-path semantics in ``perform_health_check()``: ``model``
    matches either the deployment's ``model_name`` alias or its
    ``litellm_params.model`` provider string. ``model_id`` matches
    ``model_info.id``.

    Both query params are validated against the supplied ``model_list``.
    Callers pass an already-scoped list (filtered to the caller's allowed
    models for non-admins, full list for admins), so a ``model_id`` that
    isn't present resolves to an empty set rather than a single-element
    set — preventing a non-admin from reading another deployment's cached
    health entry by guessing its ID.

    Returns ``None`` when no targeting is requested — callers should treat
    that as "no filter."
    """
    if not model and not model_id:
        return None
    target_ids: set = set()
    for m in model_list:
        deployment_id = (m.get("model_info") or {}).get("id")
        if not deployment_id:
            continue
        if model_id and deployment_id == model_id:
            target_ids.add(deployment_id)
            continue
        if model:
            litellm_model = (m.get("litellm_params") or {}).get("model")
            if m.get("model_name") == model or litellm_model == model:
                target_ids.add(deployment_id)
    return target_ids


def _filter_health_check_results_by_model_ids(
    results: dict, allowed_model_ids: set
) -> dict:
    """
    Restrict a cached background health-check result dict to endpoints whose
    model_id is in ``allowed_model_ids``.

    Endpoints without a model_id (e.g. CLI-model entries that predate the
    model_id wiring) are dropped conservatively — we cannot prove they belong
    to the caller, so they are excluded rather than leaked.

    Each retained endpoint is shallow-copied before being returned, so any
    downstream transform (e.g. _strip_admin_only_fields_from_health_result)
    cannot accidentally mutate the shared ``health_check_results`` cache.
    """
    healthy = [
        dict(ep)
        for ep in (results.get("healthy_endpoints") or [])
        if ep.get("model_id") in allowed_model_ids
    ]
    unhealthy = [
        dict(ep)
        for ep in (results.get("unhealthy_endpoints") or [])
        if ep.get("model_id") in allowed_model_ids
    ]
    return {
        "healthy_endpoints": healthy,
        "unhealthy_endpoints": unhealthy,
        "healthy_count": len(healthy),
        "unhealthy_count": len(unhealthy),
    }


async def _perform_health_check_and_save(
    model_list,
    target_model,
    cli_model,
    details,
    prisma_client,
    start_time,
    user_id,
    model_id=None,
    max_concurrency=None,
):
    """Helper function to perform health check and save results to database"""
    healthy_endpoints, unhealthy_endpoints, _ = await perform_health_check(
        model_list=model_list,
        cli_model=cli_model,
        model=target_model,
        details=details,
        max_concurrency=max_concurrency,
        model_id=model_id,
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
    response: Response,
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
        health_check_concurrency,
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

    is_admin = _is_proxy_admin(user_api_key_dict)
    model_specific_request = bool(model or model_id)

    def _post_process(result: dict) -> dict:
        # api_base / api_version reveal which provider/region/internal host the
        # deployment talks to; only proxy admins receive them. Non-admin keys
        # still see model/model_id and the healthy/unhealthy status. We also
        # set a header so non-admin clients that previously parsed those
        # fields can detect the change programmatically.
        # When a caller asked about a specific model/model_id and zero
        # endpoints came back healthy, surface that as a 503 so monitoring
        # systems can rely on the HTTP status instead of having to parse the
        # body. The body shape is unchanged.
        if model_specific_request and result.get("healthy_count", 0) == 0:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        if is_admin:
            return result
        response.headers["Litellm-Health-Field-Notice"] = (
            "api_base and api_version are admin-only on this endpoint"
        )
        return _strip_admin_only_fields_from_health_result(result)

    try:
        if llm_model_list is None:
            # if no router set, check if user set a model using litellm --model ollama/llama2
            if user_model is not None:
                cli_result = await _perform_health_check_and_save(
                    model_list=[],
                    target_model=None,
                    cli_model=user_model,
                    details=health_check_details,
                    prisma_client=prisma_client,
                    start_time=start_time,
                    user_id=user_api_key_dict.user_id,
                    model_id=None,  # CLI model doesn't have model_id
                    max_concurrency=health_check_concurrency,
                )
                return _post_process(cli_result)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Model list not initialized"},
            )
        _llm_model_list = copy.deepcopy(llm_model_list)
        ### FILTER MODELS FOR ONLY THOSE USER HAS ACCESS TO ###
        # Live path: scope by model_name (every deployment has one).
        # Cache path: scope by model_id (the cache is keyed on model_id).
        # Consequence: a deployment whose model_name the caller can access
        # but which lacks model_info.id will appear in the live /health
        # response but NOT in the background-cache /health response. This is
        # surfaced via the "warnings" field below so operators can fix the
        # missing model_info.id rather than guess at the discrepancy.
        if len(user_api_key_dict.models) > 0:
            allowed_models = set(user_api_key_dict.models)
            _llm_model_list = [
                m for m in _llm_model_list if m.get("model_name") in allowed_models
            ]
        if use_background_health_checks:
            # The cached background result covers every model. When the
            # caller targets a specific model/model_id we have to narrow the
            # cache to that deployment before _post_process evaluates
            # healthy_count, otherwise an unhealthy "foo" combined with any
            # other healthy model would still report healthy_count > 0 and
            # the targeted-503 path would never fire.
            targeted_ids = _resolve_targeted_model_ids(_llm_model_list, model, model_id)
            if len(user_api_key_dict.models) > 0:
                allowed_model_ids = {
                    (m.get("model_info") or {}).get("id")
                    for m in _llm_model_list
                    if (m.get("model_info") or {}).get("id")
                }
                # _llm_model_list is already scoped to the caller's allowed
                # model_names above, so targeted_ids is implicitly the
                # intersection of "targeted" and "allowed."
                filter_ids = (
                    targeted_ids if targeted_ids is not None else allowed_model_ids
                )
                filtered = _filter_health_check_results_by_model_ids(
                    health_check_results, filter_ids
                )
                if targeted_ids is None and not allowed_model_ids:
                    # Caller has accessible model_names but none of the
                    # matching deployments expose a model_info.id, so the
                    # cache filter (which keys on model_id) drops every
                    # entry. Surface this both as a warning log and a
                    # structured "warnings" field on the response so the
                    # caller can distinguish "no deployments found" from
                    # "deployments excluded due to missing model_info.id".
                    verbose_proxy_logger.warning(
                        "health_endpoint: scoped key %s has accessible models %s "
                        "but none of the matching deployments carry a model_info.id; "
                        "background health-check cache will return an empty result.",
                        user_api_key_dict.user_id,
                        list(user_api_key_dict.models),
                    )
                    filtered["warnings"] = [
                        "Some accessible deployments are missing model_info.id "
                        "and were excluded from this response. Ask a proxy admin "
                        "to populate model_info.id for these models."
                    ]
                return _post_process(filtered)
            if targeted_ids is not None:
                # Admin caller targeting a specific model: filter the cache
                # so the response (and the targeted-503 check) reflects only
                # that deployment, not the global aggregate.
                return _post_process(
                    _filter_health_check_results_by_model_ids(
                        health_check_results, targeted_ids
                    )
                )
            return _post_process(health_check_results)
        else:
            router_result = await _perform_health_check_and_save(
                model_list=_llm_model_list,
                target_model=target_model,
                cli_model=None,
                details=health_check_details,
                prisma_client=prisma_client,
                start_time=start_time,
                user_id=user_api_key_dict.user_id,
                model_id=model_id,
                max_concurrency=health_check_concurrency,
            )
            return _post_process(router_result)
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
    from litellm.proxy.proxy_server import _license_check, premium_user_data

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
    from litellm.proxy.proxy_server import _license_check, premium_user

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

    try:
        time_diff = datetime.now() - db_health_cache["last_updated"]
        if db_health_cache["status"] == "connected" and time_diff < timedelta(
            seconds=15
        ):
            return db_health_cache

        if prisma_client is None:
            db_health_cache = {"status": "disconnected", "last_updated": datetime.now()}
            return db_health_cache

        await prisma_client.health_check()
        db_health_cache = {"status": "connected", "last_updated": datetime.now()}
        return db_health_cache
    except Exception as e:
        db_health_cache = {"status": "disconnected", "last_updated": datetime.now()}
        if PrismaDBExceptionHandler.is_database_transport_error(e):
            try:
                verbose_proxy_logger.warning(
                    "_db_health_readiness_check: health_check failed, attempting reconnect"
                )
                await prisma_client.attempt_db_reconnect(
                    reason="health_readiness_check"
                )
                await prisma_client.health_check()
                verbose_proxy_logger.info(
                    "_db_health_readiness_check: reconnect succeeded"
                )
                db_health_cache = {
                    "status": "connected",
                    "last_updated": datetime.now(),
                }
                return db_health_cache
            except Exception:
                verbose_proxy_logger.error(
                    "_db_health_readiness_check: reconnect failed"
                )
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


async def _get_health_readiness_details(
    response: Optional[Response] = None,
) -> Dict[str, Any]:
    """
    Detailed health payload for authenticated diagnostics.
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
        cache_type: Any = None
        if litellm.cache is not None:
            from litellm.caching.caching import RedisSemanticCache

            cache_type = litellm.cache.type

            if isinstance(litellm.cache.cache, RedisSemanticCache):
                # ping the cache
                # TODO: @ishaan-jaff - we should probably not ping the cache on every /health/readiness check
                index_info: Any
                try:
                    index_info = await litellm.cache.cache._index_info()
                except Exception as e:
                    index_info = "index does not exist - error: " + str(e)  # type: ignore[assignment]
                cache_type = {"type": cache_type, "index_info": index_info}  # type: ignore[assignment]

        # check log level
        log_level_name = logging.getLevelName(verbose_logger.getEffectiveLevel())
        is_detailed_debug = verbose_logger.isEnabledFor(logging.DEBUG)

        # check DB
        if prisma_client is not None:  # if db passed in, check if it's connected
            db_health_status = await _db_health_readiness_check()
            # A configured DB that is not reachable means the worker cannot
            # serve requests that depend on persisted state (keys, budgets,
            # spend logs). Return 503 so orchestrators take this pod out of
            # rotation; "Not connected" (no DB configured at all) stays 200.
            if response is not None and db_health_status["status"] != "connected":
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "healthy",
                "db": db_health_status["status"],
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
                "use_aiohttp_transport": AsyncHTTPHandler._should_use_aiohttp_transport(),
                "log_level": log_level_name,
                "is_detailed_debug": is_detailed_debug,
            }
        else:
            return {
                "status": "healthy",
                "db": "Not connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
                "use_aiohttp_transport": AsyncHTTPHandler._should_use_aiohttp_transport(),
                "log_level": log_level_name,
                "is_detailed_debug": is_detailed_debug,
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service Unhealthy ({str(e)})")


def _allow_public_health_readiness_details() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return general_settings.get("allow_public_health_readiness_details") is True


async def _set_public_readiness_status(response: Response) -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return

    db_health_status = await _db_health_readiness_check()
    if db_health_status["status"] != "connected":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


@router.get(
    "/health/readiness",
    tags=["health"],
)
async def health_readiness(response: Response):
    """
    Public readiness probe. Keep this low-detail for unauthenticated load
    balancers by default. Admins can opt into the legacy detailed public
    payload with general_settings.allow_public_health_readiness_details.
    """
    if _allow_public_health_readiness_details():
        return await _get_health_readiness_details(response=response)

    await _set_public_readiness_status(response=response)
    return {"status": "healthy"}


@router.get(
    "/health/readiness/details",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_readiness_details(response: Response):
    """
    Authenticated readiness diagnostics with DB/cache/callback metadata.
    """
    return await _get_health_readiness_details(response=response)


@router.get(
    "/health/backlog",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_backlog():
    """
    Returns the number of HTTP requests currently in-flight on this uvicorn worker.

    Use this to measure per-pod queue depth. A high value means the worker is
    processing many concurrent requests — requests arriving now will have to wait
    for the event loop to get to them, adding latency before LiteLLM even starts
    its own timer.
    """
    return {"in_flight_requests": get_in_flight_requests()}


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
        # Reject request-supplied os.environ/ references. Config values are
        # already resolved before reaching this endpoint; any remaining
        # reference must have come from the request body.
        _reject_os_environ_references(request_litellm_params)
        model_name = request_litellm_params.get("model")

        # Look up model configuration from router if model name is provided
        # This gets the litellm_params from proxy config (with resolved env vars)
        config_litellm_params: dict = {}
        if llm_router is not None:
            # Prefer disambiguation by deployment id (`model_info.id`) when
            # the caller supplies it. This is required when multiple
            # deployments share a `model_name` (e.g. wildcard `openai/*`
            # with multiple `api_base` values for failover): the UI's
            # "Test Connection" button targets a specific row, and that
            # row's id is the only thing that uniquely identifies which
            # deployment to probe. Without this, all duplicates collapse
            # onto `deployments[0]`.
            request_model_info = model_info or {}
            request_model_id = request_model_info.get("id")
            try:
                deployment_by_id = None
                if request_model_id:
                    deployment_by_id = llm_router.get_deployment(
                        model_id=request_model_id
                    )

                if deployment_by_id is not None:
                    config_litellm_params = deployment_by_id.litellm_params.model_dump(
                        exclude_none=True
                    )
                elif model_name:
                    # Fall back to model_name lookup for callers (e.g. the
                    # "Add Model" wizard, or curl) that don't supply an id.
                    # First try to find by proxy model_name (e.g., "gpt-4o")
                    deployments = llm_router.get_model_list(model_name=model_name)

                    # If not found, try to find by litellm model name
                    # (e.g., "azure/gpt-4o")
                    if not deployments or len(deployments) == 0:
                        all_deployments = llm_router.get_model_list(model_name=None)
                        if all_deployments:
                            for deployment in all_deployments:
                                if (
                                    deployment.get("litellm_params", {}).get("model")
                                    == model_name
                                ):
                                    deployments = [deployment]
                                    break

                    if deployments and len(deployments) > 0:
                        # Use the first deployment's litellm_params as base
                        # config. These already have resolved environment
                        # variables from proxy config.
                        config_litellm_params = dict(
                            deployments[0].get("litellm_params", {})
                        )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not find model {model_name} in router: {e}. "
                    "Proceeding with request params only."
                )

        # Merge: config params (from proxy config) as base, request params override
        # This allows users to override specific params while using config for credentials
        litellm_params = {**config_litellm_params, **request_litellm_params}

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
