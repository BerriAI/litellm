import asyncio
import copy
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, Literal, Optional, Union

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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
                event_group="key",
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


@router.get("/health", tags=["health"], dependencies=[Depends(user_api_key_auth)])
async def health_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
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
        use_background_health_checks,
        user_model,
    )

    try:
        if llm_model_list is None:
            # if no router set, check if user set a model using litellm --model ollama/llama2
            if user_model is not None:
                healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                    model_list=[], cli_model=user_model, details=health_check_details
                )
                return {
                    "healthy_endpoints": healthy_endpoints,
                    "unhealthy_endpoints": unhealthy_endpoints,
                    "healthy_count": len(healthy_endpoints),
                    "unhealthy_count": len(unhealthy_endpoints),
                }
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
            healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                _llm_model_list, model, details=health_check_details
            )

            return {
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": unhealthy_endpoints,
                "healthy_count": len(healthy_endpoints),
                "unhealthy_count": len(unhealthy_endpoints),
            }
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.py::health_endpoint(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e


db_health_cache = {"status": "unknown", "last_updated": datetime.now()}


async def _db_health_readiness_check():
    from litellm.proxy.proxy_server import prisma_client

    global db_health_cache

    # Note - Intentionally don't try/except this so it raises an exception when it fails

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
                **db_health_status,
            }
        else:
            return {
                "status": "healthy",
                "db": "Not connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service Unhealthy ({str(e)})")


@router.get(
    "/health/liveliness",  # Historical LiteLLM name; doesn't match k8s terminology but kept for backwards compatibility
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/health/liveness",  # Kubernetes has "liveness" probes (https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-liveness-command)
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
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
    dependencies=[Depends(user_api_key_auth)],
)
@router.options(
    "/health/liveness",  # Kubernetes has "liveness" probes (https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-liveness-command)
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
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
