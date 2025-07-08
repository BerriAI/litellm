import asyncio
import copy
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Literal, Optional, Union

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
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

#### Health ENDPOINTS ####

router = APIRouter()


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
    service: Union[
        Literal[
            "slack_budget_alerts",
            "langfuse",
            "slack",
            "openmeter",
            "webhook",
            "email",
            "braintrust",
            "datadog",
            "generic_api",
        ],
        str,
    ] = fastapi.Query(description="Specify the service being hit."),
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
            "slack",
            "openmeter",
            "webhook",
            "braintrust",
            "otel",
            "custom_callback_api",
            "langsmith",
            "datadog",
            "generic_api",
        ]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Service must be in list. Service={service}. List={['slack_budget_alerts']}"
                },
            )

        if (
            service == "openmeter"
            or service == "braintrust"
            or service == "generic_api"
            or (service in litellm.success_callback and service != "langfuse")
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
    model_id: Optional[str] = None
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
        verbose_proxy_logger.warning(f"Failed to save health check to database for model {model_name}: {db_error}")
        # Continue execution - don't let database save failure break health checks


async def _perform_health_check_and_save(
    model_list,
    target_model,
    cli_model,
    details,
    prisma_client,
    start_time,
    user_id,
    model_id=None
):
    """Helper function to perform health check and save results to database"""
    healthy_endpoints, unhealthy_endpoints = await perform_health_check(
        model_list=model_list, 
        cli_model=cli_model, 
        model=target_model,
        details=details
    )
    
    # Optionally save health check result to database (non-blocking)
    if prisma_client is not None:
        # For CLI model, use cli_model name; for router models, use target_model
        model_name_for_db = cli_model if cli_model is not None else target_model
        if model_name_for_db is not None:
            asyncio.create_task(_save_health_check_to_db(
                prisma_client,
                model_name_for_db,
                healthy_endpoints,
                unhealthy_endpoints,
                start_time,
                user_id,
                model_id=model_id
            ))
    
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
    from litellm.proxy.proxy_server import (
        health_check_details,
        health_check_results,
        llm_model_list,
        llm_router,
        use_background_health_checks,
        user_model,
        prisma_client,
    )
    import time

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
                verbose_proxy_logger.error(f"Error getting deployment for model_id {model_id}: {e}")
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
                    model_id=None  # CLI model doesn't have model_id
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
                model_id=model_id
            )
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.py::health_endpoint(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e


@router.get("/health/history", tags=["health"], dependencies=[Depends(user_api_key_auth)])
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
    offset: int = fastapi.Query(
        0, description="Number of records to skip", ge=0
    ),
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


@router.get("/health/latest", tags=["health"], dependencies=[Depends(user_api_key_auth)])
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
            (check.model_id if check.model_id else check.model_name): _convert_health_check_to_dict(check)
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
            "batch",
            "rerank",
            "realtime",
        ]
    ] = fastapi.Body("chat", description="The mode to test the model with"),
    litellm_params: Dict = fastapi.Body(
        None,
        description="Parameters for litellm.completion, litellm.embedding for the health check",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test a direct connection to a specific model.
    
    This endpoint allows you to verify if your proxy can successfully connect to a specific model.
    It's useful for troubleshooting model connectivity issues without going through the full proxy routing.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:4000/health/test_connection' \\
      -H 'Authorization: Bearer sk-1234' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "litellm_params": {
            "model": "gpt-4",
            "custom_llm_provider": "azure_ai",
            "litellm_credential_name": null,
            "api_key": "6xxxxxxx",
            "api_base": "https://litellm8397336933.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
        },
        "mode": "chat"
      }'
    ```
    
    Returns:
        dict: A dictionary containing the health check result with either success information or error details.
    """
    try:
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

    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.health_endpoints.test_model_connection(): Exception occurred - {str(e)}"
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to test connection: {str(e)}"},
        )
