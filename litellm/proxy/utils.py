import asyncio
import copy
import hashlib
import json
import os
import smtplib
import threading
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

from litellm import _custom_logger_compatible_callbacks_literal
from litellm.constants import DEFAULT_MODEL_CREATED_AT_TIME, MAX_TEAM_LIST_LIMIT
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    CommonProxyErrors,
    ProxyErrorTypes,
    ProxyException,
    SpendLogsMetadata,
    SpendLogsPayload,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral

try:
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
        ResendEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
        SendGridEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
        SMTPEmailLogger,
    )
except ImportError:
    BaseEmailLogger = None  # type: ignore
    SendGridEmailLogger = None  # type: ignore
    SMTPEmailLogger = None  # type: ignore
    ResendEmailLogger = None  # type: ignore

try:
    import backoff
except ImportError:
    raise ImportError(
        "backoff is not installed. Please install it via 'pip install backoff'"
    )

from fastapi import HTTPException, status

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    Router,
)
from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    LiteLLM_VerificationTokenView,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.db.create_views import (
    create_missing_views,
    should_create_missing_views,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.log_db_metrics import log_db_metrics
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.proxy.hooks import PROXY_HOOKS, get_proxy_hook
from litellm.proxy.hooks.cache_control_check import _PROXY_CacheControlCheck
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.slack_alerting import DEFAULT_ALERT_TYPES
from litellm.types.mcp import (
    MCPDuringCallResponseObject,
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
)
from litellm.types.proxy.policy_engine.pipeline_types import PipelineExecutionResult
from litellm.types.utils import LLMResponseTypes, LoggedLiteLLMParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    Span = Union[_Span, Any]
else:
    Span = Any


unified_guardrail = UnifiedLLMGuardrails()


def print_verbose(print_statement):
    """
    Prints the given `print_statement` to the console if `litellm.set_verbose` is True.
    Also logs the `print_statement` at the debug level using `verbose_proxy_logger`.

    :param print_statement: The statement to be printed and logged.
    :type print_statement: Any
    """
    import traceback

    verbose_proxy_logger.debug("{}\n{}".format(print_statement, traceback.format_exc()))
    if litellm.set_verbose:
        print(f"LiteLLM Proxy: {print_statement}")  # noqa


def _get_email_logger_class():
    """
    Determine which email logger class to use based on environment variables.
    Priority: SendGrid > Resend > SMTP > BaseEmailLogger (fallback)

    Returns:
        The email logger class to use, or None if BaseEmailLogger is not available
    """
    if BaseEmailLogger is None:
        return None

    # Check for SendGrid API key
    if SendGridEmailLogger is not None and os.getenv("SENDGRID_API_KEY"):
        return SendGridEmailLogger

    # Check for Resend API key
    if ResendEmailLogger is not None and os.getenv("RESEND_API_KEY"):
        return ResendEmailLogger

    # Check for SMTP configuration
    if SMTPEmailLogger is not None and os.getenv("SMTP_HOST"):
        return SMTPEmailLogger

    # Fallback to BaseEmailLogger (though it won't actually send emails)
    return BaseEmailLogger


class InternalUsageCache:
    def __init__(self, dual_cache: DualCache):
        self.dual_cache: DualCache = dual_cache

    async def async_get_cache(
        self,
        key,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ) -> Any:
        return await self.dual_cache.async_get_cache(
            key=key,
            local_only=local_only,
            parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_set_cache(
        self,
        key,
        value,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ) -> None:
        return await self.dual_cache.async_set_cache(
            key=key,
            value=value,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_batch_set_cache(
        self,
        cache_list: List,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ) -> None:
        return await self.dual_cache.async_set_cache_pipeline(
            cache_list=cache_list,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_batch_get_cache(
        self,
        keys: list,
        parent_otel_span: Optional[Span] = None,
        local_only: bool = False,
    ):
        return await self.dual_cache.async_batch_get_cache(
            keys=keys,
            parent_otel_span=parent_otel_span,
            local_only=local_only,
        )

    async def async_increment_cache(
        self,
        key,
        value: float,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ):
        return await self.dual_cache.async_increment_cache(
            key=key,
            value=value,
            local_only=local_only,
            parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    def set_cache(
        self,
        key,
        value,
        local_only: bool = False,
        **kwargs,
    ) -> None:
        return self.dual_cache.set_cache(
            key=key,
            value=value,
            local_only=local_only,
            **kwargs,
        )

    def get_cache(
        self,
        key,
        local_only: bool = False,
        **kwargs,
    ) -> Any:
        return self.dual_cache.get_cache(
            key=key,
            local_only=local_only,
            **kwargs,
        )


### LOGGING ###
class ProxyLogging:
    """
    Logging/Custom Handlers for proxy.

    Implemented mainly to:
    - log successful/failed db read/writes
    - support the max parallel request integration
    """

    def __init__(
        self,
        user_api_key_cache: DualCache,
        premium_user: bool = False,
    ):
        ## INITIALIZE  LITELLM CALLBACKS ##
        self.call_details: dict = {}
        self.call_details["user_api_key_cache"] = user_api_key_cache
        self.internal_usage_cache: InternalUsageCache = InternalUsageCache(
            dual_cache=DualCache(default_in_memory_ttl=1)  # ping redis cache every 1s
        )
        self.max_parallel_request_limiter = _PROXY_MaxParallelRequestsHandler(
            self.internal_usage_cache
        )
        self.max_budget_limiter = _PROXY_MaxBudgetLimiter()
        self.cache_control_check = _PROXY_CacheControlCheck()
        self.alerting: Optional[List] = None
        self.alerting_threshold: float = 300  # default to 5 min. threshold
        self.alert_types: List[AlertType] = DEFAULT_ALERT_TYPES
        self.alert_to_webhook_url: Optional[dict] = None
        self.slack_alerting_instance: SlackAlerting = SlackAlerting(
            alerting_threshold=self.alerting_threshold,
            alerting=self.alerting,
            internal_usage_cache=self.internal_usage_cache.dual_cache,
        )
        self.email_logging_instance: Optional[Any] = None
        if BaseEmailLogger is not None:
            email_logger_class = _get_email_logger_class()
            if email_logger_class is not None:
                # All email logger classes now accept internal_usage_cache
                self.email_logging_instance = email_logger_class(
                    internal_usage_cache=self.internal_usage_cache.dual_cache,
                )
        self.premium_user = premium_user
        self.service_logging_obj = ServiceLogging()
        self.db_spend_update_writer = DBSpendUpdateWriter()
        self.proxy_hook_mapping: Dict[str, CustomLogger] = {}

        # Guard flags to prevent duplicate background tasks
        self.daily_report_started: bool = False
        self.hanging_requests_check_started: bool = False

    def startup_event(
        self,
        llm_router: Optional[Router],
        redis_usage_cache: Optional[RedisCache],
    ):
        """Initialize logging and alerting on proxy startup"""
        ## UPDATE SLACK ALERTING ##
        self.slack_alerting_instance.update_values(llm_router=llm_router)

        ## UPDATE INTERNAL USAGE CACHE ##
        self.update_values(
            redis_cache=redis_usage_cache
        )  # used by parallel request limiter for rate limiting keys across instances

        self._init_litellm_callbacks(
            llm_router=llm_router
        )  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

        if (
            self.slack_alerting_instance is not None
            and "daily_reports" in self.slack_alerting_instance.alert_types
            and not self.daily_report_started
        ):
            asyncio.create_task(
                self.slack_alerting_instance._run_scheduled_daily_report(
                    llm_router=llm_router
                )
            )  # RUN DAILY REPORT (if scheduled)
            self.daily_report_started = True

        if (
            self.slack_alerting_instance is not None
            and AlertType.llm_requests_hanging
            in self.slack_alerting_instance.alert_types
            and not self.hanging_requests_check_started
        ):
            asyncio.create_task(
                self.slack_alerting_instance.hanging_request_check.check_for_hanging_requests()
            )  # RUN HANGING REQUEST CHECK (if user wants to alert on hanging requests)
            self.hanging_requests_check_started = True

    def update_values(
        self,
        alerting: Optional[List] = None,
        alerting_threshold: Optional[float] = None,
        redis_cache: Optional[RedisCache] = None,
        alert_types: Optional[List[AlertType]] = None,
        alerting_args: Optional[dict] = None,
        alert_to_webhook_url: Optional[dict] = None,
    ):
        updated_slack_alerting: bool = False
        if alerting is not None:
            self.alerting = alerting
            updated_slack_alerting = True
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
            updated_slack_alerting = True
        if alert_types is not None:
            self.alert_types = alert_types
            updated_slack_alerting = True
        if alert_to_webhook_url is not None:
            self.alert_to_webhook_url = alert_to_webhook_url
            updated_slack_alerting = True

        if updated_slack_alerting is True:
            self.slack_alerting_instance.update_values(
                alerting=self.alerting,
                alerting_threshold=self.alerting_threshold,
                alert_types=self.alert_types,
                alerting_args=alerting_args,
                alert_to_webhook_url=self.alert_to_webhook_url,
            )

            if self.alerting is not None and "slack" in self.alerting:
                # NOTE: ENSURE we only add callbacks when alerting is on
                # We should NOT add callbacks when alerting is off
                if (
                    "daily_reports" in self.alert_types
                    or "outage_alerts" in self.alert_types
                    or "region_outage_alerts" in self.alert_types
                ):
                    litellm.logging_callback_manager.add_litellm_callback(self.slack_alerting_instance)  # type: ignore
                litellm.logging_callback_manager.add_litellm_success_callback(
                    self.slack_alerting_instance.response_taking_too_long_callback
                )

        if redis_cache is not None:
            self.internal_usage_cache.dual_cache.redis_cache = redis_cache
            self.db_spend_update_writer.redis_update_buffer.redis_cache = redis_cache
            self.db_spend_update_writer.pod_lock_manager.redis_cache = redis_cache

    def _add_proxy_hooks(self, llm_router: Optional[Router] = None):
        """
        Add proxy hooks to litellm.callbacks
        """
        from litellm.proxy.proxy_server import prisma_client

        for hook in PROXY_HOOKS:
            proxy_hook = get_proxy_hook(hook)
            import inspect

            expected_args = inspect.getfullargspec(proxy_hook).args
            passed_in_args: Dict[str, Any] = {}
            if "internal_usage_cache" in expected_args:
                passed_in_args["internal_usage_cache"] = self.internal_usage_cache
            if "prisma_client" in expected_args:
                passed_in_args["prisma_client"] = prisma_client
            proxy_hook_obj = cast(CustomLogger, proxy_hook(**passed_in_args))
            litellm.logging_callback_manager.add_litellm_callback(proxy_hook_obj)

            self.proxy_hook_mapping[hook] = proxy_hook_obj

    def get_proxy_hook(self, hook: str) -> Optional[CustomLogger]:
        """
        Get a proxy hook from the proxy_hook_mapping
        """
        return self.proxy_hook_mapping.get(hook)

    def _init_litellm_callbacks(self, llm_router: Optional[Router] = None):
        self._add_proxy_hooks(llm_router)
        litellm.logging_callback_manager.add_litellm_callback(self.service_logging_obj)  # type: ignore
        for callback in litellm.callbacks:
            if isinstance(callback, str):
                callback = litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class(  # type: ignore
                    cast(_custom_logger_compatible_callbacks_literal, callback),
                    internal_usage_cache=self.internal_usage_cache.dual_cache,
                    llm_router=llm_router,
                )

                if callback is None:
                    continue

            litellm.logging_callback_manager.add_litellm_callback(callback)

    async def update_request_status(
        self, litellm_call_id: str, status: Literal["success", "fail"]
    ):
        # only use this if slack alerting is being used
        if self.alerting is None:
            return

        # current alerting threshold
        alerting_threshold: float = self.alerting_threshold

        # add a 100 second buffer to the alerting threshold
        # ensures we don't send errant hanging request slack alerts
        alerting_threshold += 100

        await self.internal_usage_cache.async_set_cache(
            key="request_status:{}".format(litellm_call_id),
            value=status,
            local_only=True,
            ttl=alerting_threshold,
            litellm_parent_otel_span=None,
        )

    def _convert_user_api_key_auth_to_dict(self, user_api_key_auth_obj):
        """
        Helper function to convert UserAPIKeyAuth object to dictionary.
        Handles both Pydantic models and regular objects.
        """
        if user_api_key_auth_obj is not None:
            if hasattr(user_api_key_auth_obj, "model_dump"):
                # If it's a Pydantic model, convert to dict
                return user_api_key_auth_obj.model_dump()
            elif hasattr(user_api_key_auth_obj, "__dict__"):
                # If it's a regular object, convert to dict
                return user_api_key_auth_obj.__dict__
        return {}

    def _convert_mcp_to_llm_format(self, request_obj, kwargs: dict) -> dict:
        """
        Convert MCP tool call to LLM message format for existing guardrail validation.
        """
        from litellm.types.llms.openai import ChatCompletionUserMessage

        # Create a synthetic message that represents the tool call
        tool_call_content = (
            f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
        )

        synthetic_message = ChatCompletionUserMessage(
            role="user", content=tool_call_content
        )

        # Create synthetic LLM data that guardrails can process
        synthetic_data = {
            "messages": [synthetic_message],
            "model": kwargs.get("model", "mcp-tool-call"),
            "user_api_key_user_id": kwargs.get("user_api_key_user_id"),
            "user_api_key_team_id": kwargs.get("user_api_key_team_id"),
            "user_api_key_end_user_id": kwargs.get("user_api_key_end_user_id"),
            "user_api_key_hash": kwargs.get("user_api_key_hash"),
            "user_api_key_request_route": kwargs.get("user_api_key_request_route"),
            "mcp_tool_name": request_obj.tool_name,  # Keep original for reference
            "mcp_arguments": request_obj.arguments,  # Keep original for reference
        }

        return synthetic_data

    def _convert_llm_result_to_mcp_response(
        self, llm_result, request_obj
    ) -> Optional[Any]:
        """
        Convert LLM guardrail result back to MCP response format.
        """
        from litellm.types.mcp import MCPPreCallResponseObject

        # If result is an exception, it means the guardrail blocked the request
        if isinstance(llm_result, Exception):
            return MCPPreCallResponseObject(
                should_proceed=False,
                error_message=str(llm_result),
                modified_arguments=None,
            )

        # If result is a dict with modified messages, check for content filtering
        if isinstance(llm_result, dict):
            modified_messages = llm_result.get("messages")
            if modified_messages:
                # Check if content was blocked/modified
                original_content = (
                    f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
                )
                new_content = (
                    modified_messages[0].get("content", "") if modified_messages else ""
                )

                if new_content != original_content:
                    # Content was modified - could be masking, redaction, or blocking
                    if (
                        not new_content
                        or "blocked" in new_content.lower()
                        or "violation" in new_content.lower()
                    ):
                        # Content was blocked completely
                        return MCPPreCallResponseObject(
                            should_proceed=False,
                            error_message="Content blocked by guardrail",
                            modified_arguments=None,
                        )
                    else:
                        # Content was masked/redacted - extract the modified arguments
                        try:
                            # Try to parse the modified arguments from the masked content
                            modified_args = (
                                self._extract_modified_arguments_from_content(
                                    new_content, request_obj
                                )
                            )
                            if modified_args is not None:
                                # Return the masked/redacted arguments for the MCP call to use
                                return MCPPreCallResponseObject(
                                    should_proceed=True,
                                    error_message=None,
                                    modified_arguments=modified_args,
                                )
                            else:
                                # Could not parse modified arguments, allow original call but warn
                                verbose_proxy_logger.warning(
                                    f"Could not parse modified arguments from guardrail response: {new_content}"
                                )
                                return None
                        except Exception as e:
                            verbose_proxy_logger.error(
                                f"Error parsing modified arguments: {e}"
                            )
                            # Fallback: allow original call
                            return None

        # If result is a string, it's likely an error message
        if isinstance(llm_result, str):
            return MCPPreCallResponseObject(
                should_proceed=False, error_message=llm_result, modified_arguments=None
            )

        return None

    def _extract_modified_arguments_from_content(
        self, masked_content: str, request_obj
    ) -> Optional[dict]:
        """
        Extract modified/masked arguments from the guardrail response content.
        """
        import json

        verbose_proxy_logger.debug(
            f"Extracting modified args from content: {masked_content}"
        )

        try:
            # The format should be: "Tool: <tool_name>\nArguments: <json_arguments>"
            # Parse the arguments section
            lines = masked_content.strip().split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Arguments:"):
                    # Get the arguments part - everything after "Arguments: "
                    args_text = line[len("Arguments:") :].strip()

                    verbose_proxy_logger.debug(f"Found arguments text: {args_text}")

                    # Try to parse as JSON first
                    try:
                        modified_args = json.loads(args_text)
                        verbose_proxy_logger.debug(
                            f"Successfully parsed JSON args: {modified_args}"
                        )
                        return modified_args
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, try to extract key-value pairs manually
                        verbose_proxy_logger.debug(
                            f"Failed to parse JSON arguments: {args_text}, error: {e}"
                        )
                        return self._parse_arguments_manually(
                            args_text, request_obj.arguments
                        )

            # If we can't find the Arguments: line, return None
            verbose_proxy_logger.warning(
                "Could not find 'Arguments:' line in masked content"
            )
            return None

        except Exception as e:
            verbose_proxy_logger.error(f"Error extracting modified arguments: {e}")
            return None

    def _parse_arguments_manually(
        self, args_text: str, original_args: dict
    ) -> Optional[dict]:
        """
        Try to manually parse arguments when JSON parsing fails.
        This is a fallback for cases where the guardrail modifies the format.
        """
        import re

        try:
            # Start with original arguments and try to apply modifications
            modified_args = original_args.copy()

            # Look for simple key-value patterns
            # This is a basic implementation - can be enhanced based on specific guardrail formats
            for key, original_value in original_args.items():
                if isinstance(original_value, str):
                    # Look for the key in the masked content and try to extract its value
                    pattern = (
                        rf"['\"]?{re.escape(key)}['\"]?\s*:\s*['\"]?([^,'\"]*)['\"]?"
                    )
                    match = re.search(pattern, args_text, re.IGNORECASE)
                    if match:
                        new_value = match.group(1).strip()
                        if new_value:
                            modified_args[key] = new_value

            return modified_args

        except Exception as e:
            verbose_proxy_logger.error(f"Error in manual argument parsing: {e}")
            return None

    def _convert_llm_result_to_mcp_during_response(
        self, llm_result, request_obj
    ) -> Optional[Any]:
        """
        Convert LLM guardrail result back to MCP during call response format.
        """
        # If result is an exception, it means the guardrail wants to stop execution
        if isinstance(llm_result, Exception):
            return MCPDuringCallResponseObject(
                should_continue=False, error_message=str(llm_result)
            )

        # If result is a dict with modified messages, check for content filtering
        if isinstance(llm_result, dict):
            modified_messages = llm_result.get("messages")
            if modified_messages:
                # Check if content was blocked/modified
                original_content = (
                    f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
                )
                new_content = (
                    modified_messages[0].get("content", "") if modified_messages else ""
                )

                if new_content != original_content:
                    # Content was modified, could be masking or blocking
                    if not new_content or "blocked" in new_content.lower():
                        # Content was blocked
                        return MCPDuringCallResponseObject(
                            should_continue=False,
                            error_message="Content blocked by guardrail during execution",
                        )
                    else:
                        # Content was masked/modified - for now, stop execution
                        return MCPDuringCallResponseObject(
                            should_continue=False,
                            error_message="Content modified by guardrail during execution",
                        )

        # If result is a string, it's likely an error message
        if isinstance(llm_result, str):
            return MCPDuringCallResponseObject(
                should_continue=False, error_message=llm_result
            )

        return None

    def get_combined_callback_list(
        self, dynamic_success_callbacks: Optional[List], global_callbacks: List
    ) -> List:
        if dynamic_success_callbacks is None:
            return list(global_callbacks)
        return list(set(dynamic_success_callbacks + global_callbacks))

    def _parse_pre_mcp_call_hook_response(
        self,
        response: MCPPreCallResponseObject,
        original_request: MCPPreCallRequestObject,
    ) -> Dict[str, Any]:
        """
        Parse the response from the pre_mcp_tool_call_hook

        1. Check if the call should proceed
        2. Apply any argument modifications
        3. Handle validation errors
        """
        result = {
            "should_proceed": response.should_proceed,
            "modified_arguments": response.modified_arguments
            or original_request.arguments,
            "error_message": response.error_message,
            "hidden_params": response.hidden_params,
        }
        return result

    def _create_mcp_request_object_from_kwargs(
        self, kwargs: dict
    ) -> "MCPPreCallRequestObject":
        """
        Helper function to create MCPPreCallRequestObject from kwargs for standard pre_call_hook.
        """
        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPreCallRequestObject

        user_api_key_auth_dict = self._convert_user_api_key_auth_to_dict(
            kwargs.get("user_api_key_auth")
        )

        return MCPPreCallRequestObject(
            tool_name=kwargs.get("name", ""),
            arguments=kwargs.get("arguments", {}),
            server_name=kwargs.get("server_name"),
            user_api_key_auth=user_api_key_auth_dict,
            hidden_params=HiddenParams(),
        )

    def _convert_mcp_hook_response_to_kwargs(
        self, response_data: Optional[dict], original_kwargs: dict
    ) -> dict:
        """
        Helper function to convert pre_call_hook response back to kwargs for MCP usage.
        """
        if not response_data:
            return original_kwargs

        # Apply any argument modifications from the hook response
        modified_kwargs = original_kwargs.copy()

        # If the response contains modified arguments, apply them
        if response_data.get("modified_arguments"):
            modified_kwargs["arguments"] = response_data["modified_arguments"]

        return modified_kwargs

    async def process_pre_call_hook_response(self, response, data, call_type):
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            if call_type in ["completion", "text_completion"]:
                raise RejectedRequestError(
                    message=response,
                    model=data.get("model", ""),
                    llm_provider="",
                    request_data=data,
                )
            else:
                raise HTTPException(status_code=400, detail={"error": response})
        return data

    def _should_use_guardrail_load_balancing(
        self,
        guardrail_name: str,
    ) -> bool:
        """
        Check if load balancing should be used for this guardrail.

        Returns True if the router has multiple deployments for this guardrail name.
        """
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None or not hasattr(llm_router, "guardrail_list"):
            return False

        matching = [
            g
            for g in llm_router.guardrail_list
            if g.get("guardrail_name") == guardrail_name
        ]
        return len(matching) > 1

    async def _execute_guardrail_hook(
        self,
        callback: "CustomGuardrail",
        hook_type: str,
        data: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth],
        call_type: CallTypesLiteral,
        response: Optional[Any] = None,
    ) -> Any:
        """
        Execute a single guardrail's hook.

        Args:
            callback: The guardrail callback to execute
            hook_type: One of "pre_call", "during_call", "post_call"
            data: Request data
            user_api_key_dict: User API key auth
            call_type: Type of call
            response: Response object (for post_call hooks)

        Returns:
            Result from the guardrail execution
        """
        # Use unified_guardrail if callback has apply_guardrail method
        use_unified = "apply_guardrail" in type(callback).__dict__
        if use_unified:
            data["guardrail_to_apply"] = callback

        target = unified_guardrail if use_unified else callback

        if hook_type == "pre_call":
            return await target.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,  # type: ignore
                cache=self.call_details["user_api_key_cache"],
                data=data,
                call_type=call_type,
            )
        elif hook_type == "during_call":
            return await target.async_moderation_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,  # type: ignore
                call_type=call_type,
            )
        elif hook_type == "post_call":
            return await target.async_post_call_success_hook(
                user_api_key_dict=user_api_key_dict,  # type: ignore
                data=data,
                response=response,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown hook_type: {hook_type}")

    async def _execute_guardrail_with_load_balancing(
        self,
        guardrail_name: str,
        hook_type: str,
        data: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth],
        call_type: CallTypesLiteral,
        response: Optional[Any] = None,
    ) -> Any:
        """
        Execute a guardrail using the router's load balancing.

        Args:
            guardrail_name: Name of the guardrail
            hook_type: One of "pre_call", "during_call", "post_call"
            data: Request data
            user_api_key_dict: User API key auth
            call_type: Type of call
            response: Response object (for post_call hooks)

        Returns:
            Result from the guardrail execution
        """
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            raise ValueError("Router not initialized")

        # Select guardrail using router's load balancing
        selected_guardrail = llm_router.get_available_guardrail(
            guardrail_name=guardrail_name
        )

        callback = selected_guardrail.get("callback")
        if callback is None:
            raise ValueError(f"No callback found for guardrail: {guardrail_name}")

        return await self._execute_guardrail_hook(
            callback=callback,
            hook_type=hook_type,
            data=data,
            user_api_key_dict=user_api_key_dict,
            call_type=call_type,
            response=response,
        )

    async def _process_guardrail_callback(
        self,
        callback: CustomGuardrail,
        data: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth],
        call_type: CallTypesLiteral,
        event_type: GuardrailEventHooks,
    ) -> Optional[dict]:
        """
        Process a guardrail callback during pre-call hook.

        Supports load balancing when multiple guardrail deployments exist.

        Args:
            callback: The CustomGuardrail callback to process
            data: The request data dictionary
            user_api_key_dict: User API key authentication details
            call_type: The type of API call being made

        Returns:
            Updated data dictionary if guardrail passes, None if guardrail should be skipped
        """
        from litellm.integrations.prometheus import PrometheusLogger
        from litellm.types.guardrails import GuardrailEventHooks

        # Determine the event type based on call type
        if (
            event_type is GuardrailEventHooks.pre_call
            and call_type == CallTypes.call_mcp_tool.value
        ):
            event_type = GuardrailEventHooks.pre_mcp_call

        # Check if the guardrail should run for this request
        if callback.should_run_guardrail(data=data, event_type=event_type) is not True:
            return None

        guardrail_name = callback.guardrail_name

        # Track timing and errors for prometheus metrics
        # Use time.perf_counter() for more accurate duration measurements
        guardrail_start_time = time.perf_counter()
        status = "success"
        error_type = None

        try:
            # Check if load balancing should be used
            if guardrail_name and self._should_use_guardrail_load_balancing(
                guardrail_name
            ):
                response = await self._execute_guardrail_with_load_balancing(
                    guardrail_name=guardrail_name,
                    hook_type="pre_call",
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                )
            else:
                # Single guardrail - execute directly
                response = await self._execute_guardrail_hook(
                    callback=callback,
                    hook_type="pre_call",
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                )

            # Process the response if one was returned
            if response is not None:
                data = await self.process_pre_call_hook_response(
                    response=response, data=data, call_type=call_type
                )

        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            # Re-raise the exception to maintain existing behavior
            raise
        finally:
            # Record prometheus metrics
            guardrail_end_time = time.perf_counter()
            latency_seconds = guardrail_end_time - guardrail_start_time

            # Get guardrail name for metrics (fallback if not set)
            metrics_guardrail_name = (
                guardrail_name
                or getattr(callback, "guardrail_name", callback.__class__.__name__)
                or "unknown"
            )

            # Find PrometheusLogger in callbacks and record metrics
            for prom_callback in litellm.callbacks:
                if isinstance(prom_callback, PrometheusLogger):
                    prom_callback._record_guardrail_metrics(
                        guardrail_name=metrics_guardrail_name,
                        latency_seconds=latency_seconds,
                        status=status,
                        error_type=error_type,
                        hook_type="pre_call",
                    )
                    break

        return data

    async def _process_prompt_template(
        self,
        data: dict,
        litellm_logging_obj: Any,
        prompt_id: Any,
        prompt_version: Any,
        call_type: CallTypesLiteral,
    ) -> None:
        """Process prompt template if applicable."""

        from litellm.proxy.prompts.prompt_endpoints import (
            construct_versioned_prompt_id,
            get_latest_version_prompt_id,
        )
        from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
        from litellm.utils import get_non_default_completion_params

        if prompt_version is None:
            lookup_prompt_id = get_latest_version_prompt_id(
                prompt_id=prompt_id,
                all_prompt_ids=IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS,
            )
        else:
            lookup_prompt_id = construct_versioned_prompt_id(
                prompt_id=prompt_id, version=prompt_version
            )

        custom_logger = IN_MEMORY_PROMPT_REGISTRY.get_prompt_callback_by_id(
            lookup_prompt_id
        )
        prompt_spec = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(lookup_prompt_id)
        litellm_prompt_id: Optional[str] = None
        if prompt_spec is not None:
            litellm_prompt_id = prompt_spec.litellm_params.prompt_id
            data.pop("prompt_id", None)

        if custom_logger and prompt_spec is not None:
            (
                model,
                messages,
                optional_params,
            ) = await litellm_logging_obj.async_get_chat_completion_prompt(
                model=data.get("model", ""),
                messages=data.get("messages", []),
                non_default_params=get_non_default_completion_params(kwargs=data) or {},
                prompt_id=litellm_prompt_id,
                prompt_spec=prompt_spec,
                prompt_management_logger=custom_logger,
                prompt_variables=data.pop("prompt_variables", None) or {},
                prompt_label=data.pop("prompt_label", None) or {},
                prompt_version=data.pop("prompt_version", None) or {},
            )

            data.update(optional_params)
            data["model"] = model
            data["messages"] = messages
            # prevent re-processing the prompt template
            data.pop("prompt_id", None)
            data.pop("prompt_variables", None)
            data.pop("prompt_label", None)
            data.pop("prompt_version", None)

    def _process_guardrail_metadata(self, data: dict) -> None:
        """Process guardrails from metadata and add to applied_guardrails."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        metadata_standard = data.get("metadata") or {}
        metadata_litellm = data.get("litellm_metadata") or {}

        guardrails_in_metadata = []
        if isinstance(metadata_standard, dict) and "guardrails" in metadata_standard:
            guardrails_in_metadata = metadata_standard.get("guardrails", [])
        elif isinstance(metadata_litellm, dict) and "guardrails" in metadata_litellm:
            guardrails_in_metadata = metadata_litellm.get("guardrails", [])

        if guardrails_in_metadata and isinstance(guardrails_in_metadata, list):
            applied_guardrails = []
            if (
                isinstance(metadata_standard, dict)
                and "applied_guardrails" in metadata_standard
            ):
                applied_guardrails = metadata_standard.get("applied_guardrails", [])
            elif (
                isinstance(metadata_litellm, dict)
                and "applied_guardrails" in metadata_litellm
            ):
                applied_guardrails = metadata_litellm.get("applied_guardrails", [])

            if not isinstance(applied_guardrails, list):
                applied_guardrails = []

            for guardrail_name in guardrails_in_metadata:
                if (
                    isinstance(guardrail_name, str)
                    and guardrail_name not in applied_guardrails
                ):
                    add_guardrail_to_applied_guardrails_header(
                        request_data=data, guardrail_name=guardrail_name
                    )

    async def _maybe_execute_pipelines(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
        event_hook: str,
    ) -> dict:
        """
        Execute guardrail pipelines if any are configured for this request.

        Checks metadata for pipelines resolved by the policy engine
        and executes them. Handles the result (allow/block/modify_response).

        Returns the (possibly modified) data dict.
        """
        metadata = data.get("metadata", data.get("litellm_metadata", {})) or {}
        pipelines = metadata.get("_guardrail_pipelines")
        if not pipelines:
            return data

        for policy_name, pipeline in pipelines:
            if pipeline.mode != event_hook:
                continue

            result: PipelineExecutionResult = await PipelineExecutor.execute_steps(
                steps=pipeline.steps,
                mode=pipeline.mode,
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
                policy_name=policy_name,
            )

            data = self._handle_pipeline_result(
                result=result,
                data=data,
                policy_name=policy_name,
            )

        return data

    @staticmethod
    def _handle_pipeline_result(
        result: Any,
        data: dict,
        policy_name: str,
    ) -> dict:
        """
        Handle a PipelineExecutionResult — allow, block, or modify_response.

        Returns data dict if allowed, raises on block/modify_response.
        """
        if result.terminal_action == "allow":
            if result.modified_data is not None:
                data.update(result.modified_data)
            return data

        if result.terminal_action == "block":
            step_results_serializable = [
                {
                    "guardrail": sr.guardrail_name,
                    "outcome": sr.outcome,
                    "action": sr.action_taken,
                }
                for sr in result.step_results
            ]
            error_detail = {
                "error": {
                    "message": f"Content blocked by guardrail pipeline '{policy_name}'",
                    "type": "guardrail_pipeline_error",
                    "pipeline_context": {
                        "policy": policy_name,
                        "step_results": step_results_serializable,
                    },
                }
            }
            raise HTTPException(status_code=400, detail=error_detail)

        if result.terminal_action == "modify_response":
            raise ModifyResponseException(
                message=result.modify_response_message or "Response modified by pipeline",
                model=data.get("model", "unknown"),
                request_data=data,
                guardrail_name=f"pipeline:{policy_name}",
                detection_info=None,
            )

        return data

    # The actual implementation of the function
    @overload
    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: None,
        call_type: CallTypesLiteral,
    ) -> None:
        pass

    @overload
    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> dict:
        pass

    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Optional[dict],
        call_type: CallTypesLiteral,
    ) -> Optional[dict]:
        """
        Allows users to modify/reject the incoming request to the proxy, without having to deal with parsing Request body.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        """
        verbose_proxy_logger.debug("Inside Proxy Logging Pre-call hook!")

        self._init_response_taking_too_long_task(data=data)

        if data is None:
            return None

        litellm_logging_obj = cast(
            Optional["LiteLLMLoggingObj"], data.get("litellm_logging_obj", None)
        )
        prompt_id = data.get("prompt_id", None)
        prompt_version = data.get("prompt_version", None)

        ## PROMPT TEMPLATE CHECK ##

        if (
            litellm_logging_obj is not None
            and prompt_id is not None
            and (call_type == "completion" or call_type == "acompletion")
        ):
            await self._process_prompt_template(
                data=data,
                litellm_logging_obj=litellm_logging_obj,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                call_type=call_type,
            )

        try:
            # Execute guardrail pipelines before the normal callback loop
            data = await self._maybe_execute_pipelines(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
                event_hook="pre_call",
            )

            # Get pipeline-managed guardrails to skip in normal loop
            metadata = data.get("metadata", data.get("litellm_metadata", {})) or {}
            pipeline_managed: set = metadata.get("_pipeline_managed_guardrails", set())

            for callback in litellm.callbacks:
                start_time = time.time()
                _callback = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback  # type: ignore
                if (
                    _callback is not None
                    and isinstance(_callback, CustomGuardrail)
                    and data is not None
                ):
                    # Skip guardrails managed by a pipeline
                    if _callback.guardrail_name and _callback.guardrail_name in pipeline_managed:
                        continue

                    result = await self._process_guardrail_callback(
                        callback=_callback,
                        data=data,  # type: ignore
                        user_api_key_dict=user_api_key_dict,
                        call_type=call_type,
                        event_type=GuardrailEventHooks.pre_call,
                    )
                    if result is None:
                        continue
                    data = result

                elif (
                    _callback is not None
                    and isinstance(_callback, CustomLogger)
                    and "async_pre_call_hook" in vars(_callback.__class__)
                    and _callback.__class__.async_pre_call_hook
                    != CustomLogger.async_pre_call_hook
                ):
                    if call_type == "call_mcp_tool" and user_api_key_dict is None:
                        continue

                    response = await _callback.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=self.call_details["user_api_key_cache"],
                        data=data,  # type: ignore
                        call_type=call_type,  # type: ignore
                    )
                    if response is not None:
                        data = await self.process_pre_call_hook_response(
                            response=response, data=data, call_type=call_type
                        )

                end_time = time.time()
                duration = end_time - start_time
                if (
                    hasattr(self, "service_logging_obj") and duration > 0.01
                ):  # only if duration is non-negligible - don't spam the logs
                    await self.service_logging_obj.async_service_success_hook(
                        service=ServiceTypes.PROXY_PRE_CALL,
                        duration=duration,
                        call_type=f"{_callback.__class__.__name__}",
                        parent_otel_span=user_api_key_dict.parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                    )

            if data is not None:
                self._process_guardrail_metadata(data)

            return data
        except Exception as e:
            raise e

    async def during_call_hook(
        self,
        data: dict,
        user_api_key_dict: Optional[UserAPIKeyAuth],
        call_type: CallTypesLiteral,
    ):
        """
        Runs the CustomGuardrail's async_moderation_hook() in parallel
        """
        # Step 1: Collect all guardrail tasks to run in parallel
        guardrail_tasks = []

        for callback in litellm.callbacks:
            if isinstance(callback, CustomGuardrail):
                ################################################################
                # Check if guardrail should be run for GuardrailEventHooks.during_call hook
                ################################################################

                # V1 implementation - backwards compatibility
                if callback.event_hook is None and hasattr(
                    callback, "moderation_check"
                ):
                    if callback.moderation_check == "pre_call":  # type: ignore
                        return
                else:
                    # Main - V2 Guardrails implementation
                    from litellm.types.guardrails import GuardrailEventHooks

                    event_type = GuardrailEventHooks.during_call
                    if call_type == CallTypes.call_mcp_tool.value:
                        event_type = GuardrailEventHooks.during_mcp_call

                    if (
                        callback.should_run_guardrail(data=data, event_type=event_type)
                        is not True
                    ):
                        continue
                # Convert user_api_key_dict to proper format for async_moderation_hook
                if call_type == CallTypes.call_mcp_tool.value:
                    user_api_key_auth_dict = self._convert_user_api_key_auth_to_dict(
                        user_api_key_dict
                    )
                else:
                    user_api_key_auth_dict = user_api_key_dict
                # Add task to list for parallel execution
                if (
                    "apply_guardrail" in type(callback).__dict__
                    and user_api_key_dict is not None
                ):
                    data["guardrail_to_apply"] = callback
                    guardrail_task = unified_guardrail.async_moderation_hook(
                        user_api_key_dict=user_api_key_dict,
                        data=data,
                        call_type=call_type,
                    )
                else:
                    guardrail_task = callback.async_moderation_hook(
                        data=data,
                        user_api_key_dict=user_api_key_auth_dict,  # type: ignore
                        call_type=call_type,  # type: ignore
                    )
                guardrail_tasks.append(guardrail_task)

        # Step 2: Run all guardrail tasks in parallel
        if guardrail_tasks:
            try:
                await asyncio.gather(*guardrail_tasks)
            except Exception as e:
                # If any guardrail raises an exception, it will propagate here
                raise e

        return data

    async def failed_tracking_alert(
        self,
        error_message: str,
        failing_model: str,
    ):
        if self.alerting is None:
            return

        if self.slack_alerting_instance:
            await self.slack_alerting_instance.failed_tracking_alert(
                error_message=error_message,
                failing_model=failing_model,
            )

    async def budget_alerts(
        self,
        type: Literal[
            "token_budget",
            "user_budget",
            "soft_budget",
            "max_budget_alert",
            "team_budget",
            "organization_budget",
            "proxy_budget",
            "projected_limit_exceeded",
        ],
        user_info: CallInfo,
    ):
        # For soft_budget alerts with alert_emails set, allow email sending even if alerting is None
        # This enables team-specific soft budget email alerts via metadata.soft_budget_alerting_emails
        # Note: user_info is a CallInfo that can represent user/team/org level info. For team budgets,
        # alert_emails is populated from team_object.metadata.soft_budget_alerting_emails (see auth_checks.py)
        is_soft_budget_with_alert_emails = (
            type == "soft_budget"
            and user_info.alert_emails is not None
            and len(user_info.alert_emails) > 0
        )

        if self.alerting is None and not is_soft_budget_with_alert_emails:
            # do nothing if alerting is not switched on (unless it's a soft_budget alert with team-specific emails)
            return

        if self.alerting is not None and "slack" in self.alerting:
            if self.slack_alerting_instance is not None:
                await self.slack_alerting_instance.budget_alerts(
                    type=type,
                    user_info=user_info,
                )

        # Call email_logging_instance if:
        # 1. "email" is in alerting config, OR
        # 2. It's a soft_budget alert with team-specific alert_emails (bypasses global alerting config)
        should_send_email = (
            self.alerting is not None and "email" in self.alerting
        ) or is_soft_budget_with_alert_emails

        if should_send_email and self.email_logging_instance is not None:
            await self.email_logging_instance.budget_alerts(
                type=type,
                user_info=user_info,
            )

    async def alerting_handler(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: AlertType,
        request_data: Optional[dict] = None,
    ):
        """
        Alerting based on thresholds: - https://github.com/BerriAI/litellm/issues/1298

        - Responses taking too long
        - Requests are hanging
        - Calls are failing
        - DB Read/Writes are failing
        - Proxy Close to max budget
        - Key Close to max budget

        Parameters:
            level: str - Low|Medium|High - if calls might fail (Medium) or are failing (High); Currently, no alerts would be 'Low'.
            message: str - what is the alert about
        """
        if self.alerting is None:
            return

        from datetime import datetime

        # Get the current timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
        formatted_message = (
            f"Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
        )
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        extra_kwargs = {}
        alerting_metadata = {}
        if request_data is not None:
            _url = await _add_langfuse_trace_id_to_alert(request_data=request_data)

            if _url is not None:
                extra_kwargs["🪢 Langfuse Trace"] = _url
                formatted_message += "\n\n🪢 Langfuse Trace: {}".format(_url)
            if (
                "metadata" in request_data
                and request_data["metadata"].get("alerting_metadata", None) is not None
                and isinstance(request_data["metadata"]["alerting_metadata"], dict)
            ):
                alerting_metadata = request_data["metadata"]["alerting_metadata"]
        for client in self.alerting:
            if client == "slack":
                await self.slack_alerting_instance.send_alert(
                    message=message,
                    level=level,
                    alert_type=alert_type,
                    user_info=None,
                    alerting_metadata=alerting_metadata,
                    **extra_kwargs,
                )
            elif client == "sentry":
                if litellm.utils.sentry_sdk_instance is not None:
                    litellm.utils.sentry_sdk_instance.capture_message(formatted_message)
                else:
                    raise Exception("Missing SENTRY_DSN from environment")

    async def failure_handler(
        self, original_exception, duration: float, call_type: str, traceback_str=""
    ):
        """
        Log failed db read/writes

        Currently only logs exceptions to sentry
        """
        ### ALERTING ###
        if AlertType.db_exceptions not in self.alert_types:
            return
        if isinstance(original_exception, HTTPException):
            if isinstance(original_exception.detail, str):
                error_message = original_exception.detail
            elif isinstance(original_exception.detail, dict):
                error_message = json.dumps(original_exception.detail)
            else:
                error_message = str(original_exception)
        else:
            error_message = str(original_exception)
        if isinstance(traceback_str, str):
            error_message += traceback_str[:1000]
        asyncio.create_task(
            self.alerting_handler(
                message=f"DB read/write call failed: {error_message}",
                level="High",
                alert_type=AlertType.db_exceptions,
                request_data={},
            )
        )

        if hasattr(self, "service_logging_obj"):
            await self.service_logging_obj.async_service_failure_hook(
                service=ServiceTypes.DB,
                duration=duration,
                error=error_message,
                call_type=call_type,
            )

        if litellm.utils.capture_exception:
            litellm.utils.capture_exception(error=original_exception)

    async def post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        error_type: Optional[ProxyErrorTypes] = None,
        route: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ) -> Optional[HTTPException]:
        """
        Allows users to raise custom exceptions/log when a call fails, without having to deal with parsing Request body.
        Callbacks can return or raise HTTPException to transform error responses sent to clients.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation

        Args:
            - request_data: dict - The request data.
            - original_exception: Exception - The original exception.
            - user_api_key_dict: UserAPIKeyAuth - The user api key dict.
            - error_type: Optional[ProxyErrorTypes] - The error type.
            - route: Optional[str] - The route.
            - traceback_str: Optional[str] - The traceback string, sometimes upstream endpoints might need to send the upstream traceback. In which case we use this

        Returns:
            - Optional[HTTPException]: If any callback returns or raises an HTTPException, the first one found is returned.
                                      Otherwise, returns None and the original exception is used.
        """

        ### ALERTING ###
        await self.update_request_status(
            litellm_call_id=request_data.get("litellm_call_id", ""), status="fail"
        )
        if AlertType.llm_exceptions in self.alert_types and not isinstance(
            original_exception, HTTPException
        ):
            """
            Just alert on LLM API exceptions. Do not alert on user errors

            Related issue - https://github.com/BerriAI/litellm/issues/3395
            """
            litellm_debug_info = getattr(original_exception, "litellm_debug_info", None)
            exception_str = str(original_exception)
            if litellm_debug_info is not None:
                exception_str += litellm_debug_info

            asyncio.create_task(
                self.alerting_handler(
                    message=f"LLM API call failed: `{exception_str}`",
                    level="High",
                    alert_type=AlertType.llm_exceptions,
                    request_data=request_data,
                )
            )

        ### LOGGING ###
        if self._is_proxy_only_llm_api_error(
            original_exception=original_exception,
            error_type=error_type,
            route=user_api_key_dict.request_route,
        ):
            await self._handle_logging_proxy_only_error(
                request_data=request_data,
                user_api_key_dict=user_api_key_dict,
                route=route,
                original_exception=original_exception,
            )

        # Track the first HTTPException returned or raised by any callback
        transformed_exception: Optional[HTTPException] = None

        for callback in litellm.callbacks:
            try:
                _callback: Optional[CustomLogger] = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback  # type: ignore
                if _callback is not None and isinstance(_callback, CustomLogger):
                    try:
                        hook_result = await _callback.async_post_call_failure_hook(
                            request_data=request_data,
                            user_api_key_dict=user_api_key_dict,
                            original_exception=original_exception,
                            traceback_str=traceback_str,
                        )
                        # If callback returned an HTTPException, use it (first one wins)
                        if (
                            isinstance(hook_result, HTTPException)
                            and transformed_exception is None
                        ):
                            transformed_exception = hook_result
                    except HTTPException as e:
                        # If callback raised an HTTPException, use it (first one wins)
                        if transformed_exception is None:
                            transformed_exception = e
                    except Exception as e:
                        # Log non-HTTPException errors from callbacks but don't break the flow
                        verbose_proxy_logger.exception(
                            f"[Non-Blocking] Error in async_post_call_failure_hook callback: {e}"
                        )
            except Exception as e:
                verbose_proxy_logger.exception(
                    f"[Non-Blocking] Error setting up post_call_failure_hook callback: {e}"
                )

        return transformed_exception

    def _is_proxy_only_llm_api_error(
        self,
        original_exception: Exception,
        error_type: Optional[ProxyErrorTypes] = None,
        route: Optional[str] = None,
    ) -> bool:
        """
        Return True if the error is a Proxy Only LLM API Error

        Prevents double logging of LLM API exceptions

        e.g should only return True for:
            - Authentication Errors from user_api_key_auth
            - HTTP HTTPException (rate limit errors)
        """

        #########################################################
        # Only log LLM API errors for proxy level hooks
        # eg. Authentication errors, rate limit errors, etc.
        # Note: This fixes a security issue where we
        #       would log temporary keys/auth info
        #       from management endpoints
        #########################################################
        if route is None:
            return False
        if RouteChecks.is_llm_api_route(route) is not True:
            return False

        return isinstance(original_exception, HTTPException) or (
            error_type == ProxyErrorTypes.auth_error
        )

    async def _handle_logging_proxy_only_error(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        route: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Handle logging for proxy only errors by calling `litellm_logging_obj.async_failure_handler`

        Is triggered when self._is_proxy_only_error() returns True
        """
        litellm_logging_obj: Optional[Logging] = request_data.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj is None:
            from litellm._uuid import uuid

            request_data["litellm_call_id"] = str(uuid.uuid4())
            user_api_key_logged_metadata = (
                LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
                    user_api_key_dict=user_api_key_dict
                )
            )

            litellm_logging_obj, data = litellm.utils.function_setup(
                original_function=route or "IGNORE_THIS",
                rules_obj=litellm.utils.Rules(),
                start_time=datetime.now(),
                **request_data,
            )
            if "metadata" not in request_data:
                request_data["metadata"] = {}
            request_data["metadata"].update(user_api_key_logged_metadata)

        if litellm_logging_obj is not None:
            ## UPDATE LOGGING INPUT
            _optional_params = {}
            _litellm_params = {}

            litellm_param_keys = LoggedLiteLLMParams.__annotations__.keys()
            for k, v in request_data.items():
                if k in litellm_param_keys:
                    _litellm_params[k] = v
                elif k != "model" and k != "user":
                    _optional_params[k] = v

            litellm_logging_obj.update_environment_variables(
                model=request_data.get("model", ""),
                user=request_data.get("user", ""),
                optional_params=_optional_params,
                litellm_params=_litellm_params,
            )

            input: Union[list, str, dict] = ""
            if "messages" in request_data and isinstance(
                request_data["messages"], list
            ):
                input = request_data["messages"]
                litellm_logging_obj.model_call_details["messages"] = input
                litellm_logging_obj.call_type = CallTypes.acompletion.value
            elif "prompt" in request_data and isinstance(request_data["prompt"], str):
                input = request_data["prompt"]
                litellm_logging_obj.model_call_details["prompt"] = input
                litellm_logging_obj.call_type = CallTypes.atext_completion.value
            elif "input" in request_data and isinstance(request_data["input"], list):
                input = request_data["input"]
                litellm_logging_obj.model_call_details["input"] = input
                litellm_logging_obj.call_type = CallTypes.aembedding.value
            litellm_logging_obj.pre_call(
                input=input,
                api_key="",
            )

            # log the custom exception
            await litellm_logging_obj.async_failure_handler(
                exception=original_exception,
                traceback_exception=traceback.format_exc(),
            )

            threading.Thread(
                target=litellm_logging_obj.failure_handler,
                args=(
                    original_exception,
                    traceback.format_exc(),
                ),
            ).start()

    async def post_call_success_hook(
        self,
        data: dict,
        response: LLMResponseTypes,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Allow user to modify outgoing data

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        4. /files
        """

        from litellm.types.guardrails import GuardrailEventHooks


        guardrail_callbacks: List[CustomGuardrail] = []
        other_callbacks: List[CustomLogger] = []
        try:
            for callback in litellm.callbacks:
                _callback: Optional[CustomLogger] = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback  # type: ignore

                if _callback is not None:
                    if isinstance(_callback, CustomGuardrail):
                        guardrail_callbacks.append(_callback)
                    else:
                        other_callbacks.append(_callback)
                    ############## Handle Guardrails ########################################
                    #############################################################################

            for callback in guardrail_callbacks:
                # Main - V2 Guardrails implementation

                if (
                    callback.should_run_guardrail(
                        data=data, event_type=GuardrailEventHooks.post_call
                    )
                    is not True
                ):
                    continue

                guardrail_response: Optional[Any] = None

                if "apply_guardrail" in type(callback).__dict__:
                    data["guardrail_to_apply"] = callback
                    guardrail_response = (
                        await unified_guardrail.async_post_call_success_hook(
                            user_api_key_dict=user_api_key_dict,
                            data=data,
                            response=response,
                        )
                    )
                else:
                    guardrail_response = await callback.async_post_call_success_hook(
                        user_api_key_dict=user_api_key_dict,
                        data=data,
                        response=response,
                    )

                if guardrail_response is not None:
                    response = guardrail_response

            ############ Handle CustomLogger ###############################
            #################################################################

            for callback in other_callbacks:
                callback_response = await callback.async_post_call_success_hook(
                    user_api_key_dict=user_api_key_dict, data=data, response=response
                )
                if callback_response is not None:
                    response = callback_response
        except Exception as e:
            raise e
        return response

    async def post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Calls async_post_call_response_headers_hook on all CustomLogger callbacks.
        Merges all returned header dicts (later callbacks override earlier ones).

        Returns:
            Dict[str, str]: Merged headers from all callbacks.
        """
        merged_headers: Dict[str, str] = {}
        try:
            for callback in litellm.callbacks:
                _callback: Optional[CustomLogger] = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback  # type: ignore

                if _callback is not None and isinstance(_callback, CustomLogger):
                    result = await _callback.async_post_call_response_headers_hook(
                        data=data,
                        user_api_key_dict=user_api_key_dict,
                        response=response,
                        request_headers=request_headers,
                    )
                    if result is not None:
                        merged_headers.update(result)
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error in post_call_response_headers_hook: %s", str(e)
            )
        return merged_headers

    def is_a2a_streaming_response(self, response: dict) -> bool:
        expected_keys = ["jsonrpc", "id", "result"]
        return all(key in response for key in expected_keys)

    async def async_post_call_streaming_hook(
        self,
        data: dict,
        response: Union[
            ModelResponse, EmbeddingResponse, ImageResponse, ModelResponseStream
        ],
        user_api_key_dict: UserAPIKeyAuth,
        str_so_far: Optional[str] = None,
    ):
        """
        Allow user to modify outgoing streaming data -> per chunk

        Covers:
        1. /chat/completions
        """
        from litellm.proxy.proxy_server import llm_router

        response_str: Optional[str] = None
        if isinstance(response, (ModelResponse, ModelResponseStream)):
            response_str = litellm.get_response_string(response_obj=response)
        elif isinstance(response, dict) and self.is_a2a_streaming_response(response):
            from litellm.llms.a2a.common_utils import extract_text_from_a2a_response

            response_str = extract_text_from_a2a_response(response)
        if response_str is not None:
            for callback in litellm.callbacks:
                try:
                    _callback: Optional[CustomLogger] = None
                    if isinstance(callback, CustomGuardrail):
                        # Main - V2 Guardrails implementation
                        from litellm.types.guardrails import GuardrailEventHooks

                        ## CHECK FOR MODEL-LEVEL GUARDRAILS
                        modified_data = _check_and_merge_model_level_guardrails(
                            data=data, llm_router=llm_router
                        )

                        if (
                            callback.should_run_guardrail(
                                data=modified_data,
                                event_type=GuardrailEventHooks.post_call,
                            )
                            is not True
                        ):
                            continue
                    if isinstance(callback, str):
                        _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                            cast(_custom_logger_compatible_callbacks_literal, callback)
                        )
                    else:
                        _callback = callback  # type: ignore
                    if _callback is not None and isinstance(_callback, CustomLogger):
                        if str_so_far is not None:
                            complete_response = str_so_far + response_str
                        else:
                            complete_response = response_str
                        callback_response = (
                            await _callback.async_post_call_streaming_hook(
                                user_api_key_dict=user_api_key_dict,
                                response=complete_response,
                            )
                        )
                        if callback_response is not None:
                            response = callback_response
                except Exception as e:
                    raise e
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        response,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
    ):
        """
        Allow user to modify outgoing streaming data -> Given a whole response iterator.
        This hook is best used when you need to modify multiple chunks of the response at once.

        Covers:
        1. /chat/completions
        """
        current_response = response

        for callback in litellm.callbacks:
            _callback: Optional[CustomLogger] = None
            if isinstance(callback, str):
                _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                    cast(_custom_logger_compatible_callbacks_literal, callback)
                )
            else:
                _callback = callback  # type: ignore
            if _callback is not None and isinstance(_callback, CustomLogger):
                if not isinstance(
                    _callback, CustomGuardrail
                ) or _callback.should_run_guardrail(
                    data=request_data, event_type=GuardrailEventHooks.post_call
                ):
                    if (
                        "async_post_call_streaming_iterator_hook"
                        in type(callback).__dict__
                    ):
                        current_response = (
                            _callback.async_post_call_streaming_iterator_hook(
                                user_api_key_dict=user_api_key_dict,
                                response=current_response,
                                request_data=request_data,
                            )
                        )
                    elif "apply_guardrail" in type(callback).__dict__:
                        request_data["guardrail_to_apply"] = callback
                        current_response = (
                            unified_guardrail.async_post_call_streaming_iterator_hook(
                                user_api_key_dict=user_api_key_dict,
                                request_data=request_data,
                                response=current_response,
                            )
                        )
                    else:
                        current_response = (
                            _callback.async_post_call_streaming_iterator_hook(
                                user_api_key_dict=user_api_key_dict,
                                response=current_response,
                                request_data=request_data,
                            )
                        )

        # Actually iterate through the chained async generator and yield chunks
        async for chunk in current_response:
            yield chunk

    def _init_response_taking_too_long_task(self, data: Optional[dict] = None):
        """
        Initialize the response taking too long task if user is using slack alerting

        Only run task if user is using slack alerting

        This handles checking for if a request is hanging for too long
        """
        ## ALERTING ###
        if (
            self.slack_alerting_instance
            and self.slack_alerting_instance.alerting is not None
        ):
            asyncio.create_task(
                self.slack_alerting_instance.response_taking_too_long(request_data=data)
            )


### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print_verbose(f"Backing off... this was attempt #{details['tries']}")


def jsonify_object(data: dict) -> dict:
    db_data = copy.deepcopy(data)

    for k, v in db_data.items():
        if isinstance(v, dict):
            try:
                db_data[k] = json.dumps(v)
            except Exception:
                # This avoids Prisma retrying this 5 times, and making 5 clients
                db_data[k] = "failed-to-serialize-json"
    return db_data


# In-memory cache for deprecated key lookups: maps old_token_hash -> (active_token_id, expires_at_ts)
# Avoids a DB query on every auth request for non-deprecated keys.
# Bounded to prevent memory leaks from accumulated rotations.
_deprecated_key_cache: LimitedSizeOrderedDict = LimitedSizeOrderedDict(max_size=1000)
_DEPRECATED_KEY_CACHE_TTL_SECONDS = 60


async def _lookup_deprecated_key(
    db: Any,
    hashed_token: str,
) -> Optional[str]:
    """
    Check if a token exists in the deprecated keys table and is still within its grace period.

    Returns the active_token_id if found and valid, otherwise None.
    Uses an in-memory cache to avoid DB queries on every auth request.
    """
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    # Check cache first
    cached = _deprecated_key_cache.get(hashed_token)
    cached = _deprecated_key_cache.get(hashed_token)
    if cached is not None:
        active_token_id, cache_expires_at_ts, revoke_at_ts = cached
        if now_ts < cache_expires_at_ts and now_ts < revoke_at_ts:
            return active_token_id
        else:
            _deprecated_key_cache.pop(hashed_token, None)

    try:
        deprecated_row = await db.litellm_deprecatedverificationtoken.find_first(
            where={
                "token": hashed_token,
                "revoke_at": {"gt": now},
            },
            select={"active_token_id": True},
        )
        if deprecated_row and deprecated_row.active_token_id:
            _deprecated_key_cache[hashed_token] = (
                deprecated_row.active_token_id,
                now_ts + _DEPRECATED_KEY_CACHE_TTL_SECONDS,
            )
            return deprecated_row.active_token_id
        # Only cache positive results; negative lookups are fast on indexed columns
        # and caching them risks evicting real deprecated key entries.
    except Exception as e:
        verbose_proxy_logger.debug("Deprecated key lookup skipped: %s", e)

    return None


class PrismaClient:
    spend_log_transactions: List = []
    _spend_log_transactions_lock = asyncio.Lock()

    def __init__(
        self,
        database_url: str,
        proxy_logging_obj: ProxyLogging,
        http_client: Optional[Any] = None,
    ):
        ## init logging object
        self.proxy_logging_obj = proxy_logging_obj
        self.iam_token_db_auth: Optional[bool] = str_to_bool(
            os.getenv("IAM_TOKEN_DB_AUTH")
        )
        verbose_proxy_logger.debug("Creating Prisma Client..")
        try:
            from prisma import Prisma  # type: ignore
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to import Prisma client: {e}")
            verbose_proxy_logger.error(
                "This usually means 'prisma generate' hasn't been run yet."
            )
            verbose_proxy_logger.error(
                "Please run 'prisma generate' to generate the Prisma client."
            )
            raise Exception(
                "Unable to find Prisma binaries. Please run 'prisma generate' first."
            )
        if http_client is not None:
            self.db = PrismaWrapper(
                original_prisma=Prisma(http=http_client),
                iam_token_db_auth=(
                    self.iam_token_db_auth
                    if self.iam_token_db_auth is not None
                    else False
                ),
            )
        else:
            self.db = PrismaWrapper(
                original_prisma=Prisma(),
                iam_token_db_auth=(
                    self.iam_token_db_auth
                    if self.iam_token_db_auth is not None
                    else False
                ),
            )  # Client to connect to Prisma db
        verbose_proxy_logger.debug("Success - Created Prisma Client")

    def get_request_status(
        self, payload: Union[dict, SpendLogsPayload]
    ) -> Literal["success", "failure"]:
        """
        Determine if a request was successful or failed based on payload metadata.

        Args:
            payload (Union[dict, SpendLogsPayload]): Request payload containing metadata

        Returns:
            Literal["success", "failure"]: Request status
        """
        try:
            # Get metadata and convert to dict if it's a JSON string
            payload_metadata: Union[Dict, SpendLogsMetadata, str] = payload.get(
                "metadata", {}
            )
            if isinstance(payload_metadata, str):
                payload_metadata_json: Union[Dict, SpendLogsMetadata] = cast(
                    Dict, json.loads(payload_metadata)
                )
            else:
                payload_metadata_json = payload_metadata

            # Check status in metadata dict
            return (
                "failure"
                if payload_metadata_json.get("status") == "failure"
                else "success"
            )

        except (json.JSONDecodeError, AttributeError):
            # Default to success if metadata parsing fails
            return "success"

    def hash_token(self, token: str):
        # Hash the string using SHA-256
        hashed_token = hashlib.sha256(token.encode()).hexdigest()

        return hashed_token

    def jsonify_object(self, data: dict) -> dict:
        db_data = copy.deepcopy(data)

        for k, v in db_data.items():
            if isinstance(v, dict):
                try:
                    db_data[k] = json.dumps(v)
                except Exception:
                    # This avoids Prisma retrying this 5 times, and making 5 clients
                    db_data[k] = "failed-to-serialize-json"
        return db_data

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def check_view_exists(self):
        """
        Checks if the LiteLLM_VerificationTokenView and MonthlyGlobalSpend exists in the user's db.

        LiteLLM_VerificationTokenView: This view is used for getting the token + team data in user_api_key_auth

        MonthlyGlobalSpend: This view is used for the admin view to see global spend for this month

        If the view doesn't exist, one will be created.
        """

        # Check to see if all of the necessary views exist and if they do, simply return
        # This is more efficient because it lets us check for all views in one
        # query instead of multiple queries.
        try:
            expected_views = [
                "LiteLLM_VerificationTokenView",
                "MonthlyGlobalSpend",
                "Last30dKeysBySpend",
                "Last30dModelsBySpend",
                "MonthlyGlobalSpendPerKey",
                "MonthlyGlobalSpendPerUserPerKey",
                "Last30dTopEndUsersSpend",
                "DailyTagSpend",
            ]
            required_view = "LiteLLM_VerificationTokenView"
            expected_views_str = ", ".join(f"'{view}'" for view in expected_views)
            pg_schema = os.getenv("DATABASE_SCHEMA", "public")
            ret = await self.db.query_raw(
                f"""
                WITH existing_views AS (
                    SELECT viewname
                    FROM pg_views
                    WHERE schemaname = '{pg_schema}' AND viewname IN (
                        {expected_views_str}
                    )
                )
                SELECT
                    (SELECT COUNT(*) FROM existing_views) AS view_count,
                    ARRAY_AGG(viewname) AS view_names
                FROM existing_views
                """
            )
            expected_total_views = len(expected_views)
            if ret[0]["view_count"] == expected_total_views:
                verbose_proxy_logger.info("All necessary views exist!")
                return
            else:
                ## check if required view exists ##
                if ret[0]["view_names"] and required_view not in ret[0]["view_names"]:
                    await self.health_check()  # make sure we can connect to db
                    await self.db.execute_raw(
                        """
                            CREATE VIEW "LiteLLM_VerificationTokenView" AS
                            SELECT
                            v.*,
                            t.spend AS team_spend,
                            t.max_budget AS team_max_budget,
                            t.tpm_limit AS team_tpm_limit,
                            t.rpm_limit AS team_rpm_limit
                            FROM "LiteLLM_VerificationToken" v
                            LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id;
                        """
                    )

                    verbose_proxy_logger.info(
                        "LiteLLM_VerificationTokenView Created in DB!"
                    )
                else:
                    should_create_views = await should_create_missing_views(db=self.db)
                    if should_create_views:
                        await create_missing_views(db=self.db)
                    else:
                        # don't block execution if these views are missing
                        # Convert lists to sets for efficient difference calculation
                        ret_view_names_set = (
                            set(ret[0]["view_names"]) if ret[0]["view_names"] else set()
                        )
                        expected_views_set = set(expected_views)
                        # Find missing views
                        missing_views = expected_views_set - ret_view_names_set

                        verbose_proxy_logger.warning(
                            "\n\n\033[93mNot all views exist in db, needed for UI 'Usage' tab. Missing={}.\nRun 'create_views.py' from https://github.com/BerriAI/litellm/tree/main/db_scripts to create missing views.\033[0m\n".format(
                                missing_views
                            )
                        )

        except Exception:
            raise
        return

    @log_db_metrics
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=1,  # maximum number of retries
        max_time=2,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def get_generic_data(
        self,
        key: str,
        value: Any,
        table_name: Literal["users", "keys", "config", "spend"],
    ):
        """
        Generic implementation of get data
        """
        start_time = time.time()
        try:
            if table_name == "users":
                response = await self.db.litellm_usertable.find_first(
                    where={key: value}  # type: ignore
                )
            elif table_name == "keys":
                response = await self.db.litellm_verificationtoken.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            elif table_name == "config":
                response = await self.db.litellm_config.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            elif table_name == "spend":
                response = await self.db.l.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            return response
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception get_generic_data: {str(e)}"
            verbose_proxy_logger.error(error_msg)
            error_msg = error_msg + "\nException Type: {}".format(type(e))
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    traceback_str=error_traceback,
                    call_type="get_generic_data",
                )
            )

            raise e

    async def _query_first_with_cached_plan_fallback(
        self, sql_query: str
    ) -> Optional[dict]:
        """
        Execute a query with automatic fallback for PostgreSQL cached plan errors.

        This handles the "cached plan must not change result type" error that occurs
        during rolling deployments when schema changes are applied while old pods
        still have cached query plans expecting the old schema.

        Args:
            sql_query: SQL query string to execute

        Returns:
            Query result or None

        Raises:
            Original exception if not a cached plan error
        """
        try:
            return await self.db.query_first(query=sql_query)
        except Exception as e:
            error_str = str(e)
            if "cached plan must not change result type" in error_str:
                # Force PostgreSQL to re-plan by invalidating the cache
                # Add a unique comment to make the query different
                sql_query_retry = sql_query.replace(
                    "SELECT",
                    f"SELECT /* cache_invalidated_{int(time.time() * 1000)} */",
                )
                verbose_proxy_logger.warning(
                    "PostgreSQL cached plan error detected for token lookup, "
                    "retrying with fresh plan. This may occur during rolling deployments "
                    "when schema changes are applied."
                )
                return await self.db.query_first(query=sql_query_retry)
            else:
                raise

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    @log_db_metrics
    async def get_data(  # noqa: PLR0915
        self,
        token: Optional[Union[str, list]] = None,
        user_id: Optional[str] = None,
        user_id_list: Optional[list] = None,
        team_id: Optional[str] = None,
        team_id_list: Optional[list] = None,
        key_val: Optional[dict] = None,
        table_name: Optional[
            Literal[
                "user",
                "key",
                "config",
                "spend",
                "enduser",
                "budget",
                "team",
                "user_notification",
                "combined_view",
            ]
        ] = None,
        query_type: Literal["find_unique", "find_all"] = "find_unique",
        expires: Optional[datetime] = None,
        reset_at: Optional[datetime] = None,
        offset: Optional[int] = None,  # pagination, what row number to start from
        limit: Optional[
            int
        ] = None,  # pagination, number of rows to getch when find_all==True
        parent_otel_span: Optional[Span] = None,
        proxy_logging_obj: Optional[ProxyLogging] = None,
        budget_id_list: Optional[List[str]] = None,
        check_deprecated: bool = True,
    ):
        args_passed_in = locals()
        start_time = time.time()
        hashed_token: Optional[str] = None
        try:
            response: Any = None
            if (token is not None and table_name is None) or (
                table_name is not None and table_name == "key"
            ):
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = _hash_token_if_needed(token=token)
                        verbose_proxy_logger.debug(
                            f"PrismaClient: find_unique for token: {hashed_token}"
                        )
                if query_type == "find_unique" and hashed_token is not None:
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )
                    response = await self.db.litellm_verificationtoken.find_unique(
                        where={"token": hashed_token},  # type: ignore
                        include={"litellm_budget_table": True},
                    )
                    if response is not None:
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(
                            response.expires, datetime
                        ):
                            response.expires = response.expires.isoformat()
                    else:
                        # Token does not exist.
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Authentication Error: invalid user key - user key does not exist in db. User Key={token}",
                        )
                elif query_type == "find_all" and user_id is not None:
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={"user_id": user_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all" and team_id is not None:
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={"team_id": team_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif (
                    query_type == "find_all"
                    and expires is not None
                    and reset_at is not None
                ):
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={  # type:ignore
                            "OR": [
                                {"expires": None},
                                {"expires": {"gt": expires}},
                            ],
                            "budget_reset_at": {"lt": reset_at},
                        }
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all":
                    where_filter: dict = {}
                    if token is not None:
                        where_filter["token"] = {}
                        if isinstance(token, str):
                            token = _hash_token_if_needed(token=token)
                            where_filter["token"]["in"] = [token]
                        elif isinstance(token, list):
                            hashed_tokens = []
                            for t in token:
                                assert isinstance(t, str)
                                if t.startswith("sk-"):
                                    new_token = self.hash_token(token=t)
                                    hashed_tokens.append(new_token)
                                else:
                                    hashed_tokens.append(t)
                            where_filter["token"]["in"] = hashed_tokens
                    response = await self.db.litellm_verificationtoken.find_many(
                        order={"spend": "desc"},
                        where=where_filter,  # type: ignore
                        include={"litellm_budget_table": True},
                    )
                if response is not None:
                    return response
                else:
                    # Token does not exist.
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication Error: invalid user key - token does not exist",
                    )
            elif (user_id is not None and table_name is None) or (
                table_name is not None and table_name == "user"
            ):
                if query_type == "find_unique":
                    if key_val is None:
                        key_val = {"user_id": user_id}

                    response = await self.db.litellm_usertable.find_unique(  # type: ignore
                        where=key_val,  # type: ignore
                        include={"organization_memberships": True},
                    )

                elif query_type == "find_all" and key_val is not None:
                    response = await self.db.litellm_usertable.find_many(
                        where=key_val  # type: ignore
                    )  # type: ignore
                elif query_type == "find_all" and reset_at is not None:
                    response = await self.db.litellm_usertable.find_many(
                        where={  # type:ignore
                            "budget_reset_at": {"lt": reset_at},
                        }
                    )
                elif query_type == "find_all" and user_id_list is not None:
                    response = await self.db.litellm_usertable.find_many(
                        where={"user_id": {"in": user_id_list}}
                    )
                elif query_type == "find_all":
                    if expires is not None:
                        response = await self.db.litellm_usertable.find_many(  # type: ignore
                            order={"spend": "desc"},
                            where={  # type:ignore
                                "OR": [
                                    {"expires": None},  # type:ignore
                                    {"expires": {"gt": expires}},  # type:ignore
                                ],
                            },
                        )
                    else:
                        # return all users in the table, get their key aliases ordered by spend
                        sql_query = """
                        SELECT
                            u.*,
                            json_agg(v.key_alias) AS key_aliases
                        FROM
                            "LiteLLM_UserTable" u
                        LEFT JOIN "LiteLLM_VerificationToken" v ON u.user_id = v.user_id
                        GROUP BY
                            u.user_id
                        ORDER BY u.spend DESC
                        LIMIT $1
                        OFFSET $2
                        """
                        response = await self.db.query_raw(sql_query, limit, offset)
                return response
            elif table_name == "spend":
                verbose_proxy_logger.debug(
                    "PrismaClient: get_data: table_name == 'spend'"
                )
                if key_val is not None:
                    if query_type == "find_unique":
                        response = await self.db.litellm_spendlogs.find_unique(  # type: ignore
                            where={  # type: ignore
                                key_val["key"]: key_val["value"],  # type: ignore
                            }
                        )
                    elif query_type == "find_all":
                        response = await self.db.litellm_spendlogs.find_many(  # type: ignore
                            where={
                                key_val["key"]: key_val["value"],  # type: ignore
                            }
                        )
                    return response
                else:
                    response = await self.db.litellm_spendlogs.find_many(  # type: ignore
                        order={"startTime": "desc"},
                    )
                    return response
            elif table_name == "budget" and reset_at is not None:
                if query_type == "find_all":
                    response = await self.db.litellm_budgettable.find_many(
                        where={  # type:ignore
                            "OR": [
                                {
                                    "AND": [
                                        {"budget_reset_at": None},
                                        {"NOT": {"budget_duration": None}},
                                    ]
                                },
                                {"budget_reset_at": {"lt": reset_at}},
                            ]
                        }
                    )
                    return response

            elif table_name == "enduser" and budget_id_list is not None:
                if query_type == "find_all":
                    response = await self.db.litellm_endusertable.find_many(
                        where={"budget_id": {"in": budget_id_list}}
                    )
                    return response
            elif table_name == "team":
                if query_type == "find_unique":
                    response = await self.db.litellm_teamtable.find_unique(
                        where={"team_id": team_id},  # type: ignore
                        include={"litellm_model_table": True},  # type: ignore
                    )
                elif query_type == "find_all" and reset_at is not None:
                    response = await self.db.litellm_teamtable.find_many(
                        where={  # type:ignore
                            "budget_reset_at": {"lt": reset_at},
                        }
                    )
                elif query_type == "find_all" and user_id is not None:
                    response = await self.db.litellm_teamtable.find_many(
                        where={
                            "members": {"has": user_id},
                        },
                        include={"litellm_budget_table": True},
                    )
                elif query_type == "find_all" and team_id_list is not None:
                    response = await self.db.litellm_teamtable.find_many(
                        where={"team_id": {"in": team_id_list}}
                    )
                elif query_type == "find_all" and team_id_list is None:
                    response = await self.db.litellm_teamtable.find_many(
                        take=MAX_TEAM_LIST_LIMIT
                    )
                return response
            elif table_name == "user_notification":
                if query_type == "find_unique":
                    response = await self.db.litellm_usernotifications.find_unique(  # type: ignore
                        where={"user_id": user_id}  # type: ignore
                    )
                elif query_type == "find_all":
                    response = await self.db.litellm_usernotifications.find_many()  # type: ignore
                return response
            elif table_name == "combined_view":
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = _hash_token_if_needed(token=token)
                        verbose_proxy_logger.debug(
                            f"PrismaClient: find_unique for token: {hashed_token}"
                        )
                if query_type == "find_unique":
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )

                    sql_query = f"""
                        SELECT 
                            v.*,
                            t.spend AS team_spend, 
                            t.max_budget AS team_max_budget,
                            t.soft_budget AS team_soft_budget,
                            t.tpm_limit AS team_tpm_limit,
                            t.rpm_limit AS team_rpm_limit,
                            t.models AS team_models,
                            t.metadata AS team_metadata,
                            t.blocked AS team_blocked,
                            t.team_alias AS team_alias,
                            t.metadata AS team_metadata,
                            t.members_with_roles AS team_members_with_roles,
                            t.object_permission_id AS team_object_permission_id,
                            t.organization_id as org_id,
                            tm.spend AS team_member_spend,
                            m.aliases AS team_model_aliases,
                            -- Added comma to separate b.* columns
                            b.max_budget AS litellm_budget_table_max_budget,
                            b.tpm_limit AS litellm_budget_table_tpm_limit,
                            b.rpm_limit AS litellm_budget_table_rpm_limit,
                            b.model_max_budget as litellm_budget_table_model_max_budget,
                            b.soft_budget as litellm_budget_table_soft_budget,
                            o.metadata as organization_metadata,
                            b2.max_budget as organization_max_budget,
                            b2.tpm_limit as organization_tpm_limit,
                            b2.rpm_limit as organization_rpm_limit
                        FROM "LiteLLM_VerificationToken" AS v
                        LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
                        LEFT JOIN "LiteLLM_TeamMembership" AS tm ON v.team_id = tm.team_id AND tm.user_id = v.user_id
                        LEFT JOIN "LiteLLM_ModelTable" m ON t.model_id = m.id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b ON v.budget_id = b.budget_id
                        LEFT JOIN "LiteLLM_OrganizationTable" AS o ON v.organization_id = o.organization_id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b2 ON o.budget_id = b2.budget_id
                        WHERE v.token = '{token}'
                    """

                    response = await self._query_first_with_cached_plan_fallback(
                        sql_query
                    )

                    # If not found in main table, check deprecated keys (grace period)
                    # check_deprecated=False on the recursive call prevents unbounded chaining
                    if (
                        response is None
                        and hashed_token is not None
                        and check_deprecated
                    ):
                        active_token_id = await _lookup_deprecated_key(
                            db=self.db, hashed_token=hashed_token
                        )
                        if active_token_id:
                            response = await self.get_data(
                                token=active_token_id,
                                table_name="combined_view",
                                query_type="find_unique",
                                parent_otel_span=parent_otel_span,
                                proxy_logging_obj=proxy_logging_obj,
                                check_deprecated=False,
                            )
                            if response is not None:
                                verbose_proxy_logger.debug(
                                    "Deprecated key used during grace period"
                                )

                    if response is not None:
                        if response["team_models"] is None:
                            response["team_models"] = []
                        if response["team_blocked"] is None:
                            response["team_blocked"] = False

                        team_member: Optional[Member] = None
                        if (
                            response["team_members_with_roles"] is not None
                            and response["user_id"] is not None
                        ):
                            ## find the team member corresponding to user id
                            """
                            [
                                {
                                    "role": "admin",
                                    "user_id": "default_user_id",
                                    "user_email": null
                                },
                                {
                                    "role": "user",
                                    "user_id": null,
                                    "user_email": "test@email.com"
                                }
                            ]
                            """
                            for tm in response["team_members_with_roles"]:
                                if tm.get("user_id") is not None and response[
                                    "user_id"
                                ] == tm.get("user_id"):
                                    team_member = Member(**tm)
                        response["team_member"] = team_member
                        response = LiteLLM_VerificationTokenView(
                            **response, last_refreshed_at=time.time()
                        )
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(
                            response.expires, datetime
                        ):
                            response.expires = response.expires.isoformat()
                    return response
        except Exception as e:
            import traceback

            prisma_query_info = f"LiteLLM Prisma Client Exception: Error with `get_data`. Args passed in: {args_passed_in}"
            error_msg = prisma_query_info + str(e)
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            verbose_proxy_logger.debug(error_traceback)
            end_time = time.time()
            _duration = end_time - start_time

            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="get_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    def jsonify_team_object(self, db_data: dict):
        db_data = self.jsonify_object(data=db_data)
        if db_data.get("members_with_roles", None) is not None and isinstance(
            db_data["members_with_roles"], list
        ):
            db_data["members_with_roles"] = json.dumps(db_data["members_with_roles"])
        return db_data

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def insert_data(  # noqa: PLR0915
        self,
        data: dict,
        table_name: Literal[
            "user", "key", "config", "spend", "team", "user_notification"
        ],
    ):
        """
        Add a key to the database. If it already exists, do nothing.
        """
        start_time = time.time()
        try:
            verbose_proxy_logger.debug("PrismaClient: insert_data: %s", data)
            if table_name == "key":
                token = data["token"]
                hashed_token = self.hash_token(token=token)
                db_data = self.jsonify_object(data=data)
                db_data["token"] = hashed_token
                print_verbose(
                    "PrismaClient: Before upsert into litellm_verificationtoken"
                )
                new_verification_token = await self.db.litellm_verificationtoken.upsert(  # type: ignore
                    where={
                        "token": hashed_token,
                    },
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                    include={"litellm_budget_table": True},
                )
                verbose_proxy_logger.info("Data Inserted into Keys Table")
                return new_verification_token
            elif table_name == "user":
                db_data = self.jsonify_object(data=data)
                try:
                    new_user_row = await self.db.litellm_usertable.upsert(
                        where={"user_id": data["user_id"]},
                        data={
                            "create": {**db_data},  # type: ignore
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                except Exception as e:
                    if (
                        "Foreign key constraint failed on the field: `LiteLLM_UserTable_organization_id_fkey (index)`"
                        in str(e)
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"Foreign Key Constraint failed. Organization ID={db_data['organization_id']} does not exist in LiteLLM_OrganizationTable. Create via `/organization/new`."
                            },
                        )
                    raise e
                verbose_proxy_logger.info("Data Inserted into User Table")
                return new_user_row
            elif table_name == "team":
                db_data = self.jsonify_team_object(db_data=data)
                new_team_row = await self.db.litellm_teamtable.upsert(
                    where={"team_id": data["team_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Team Table")
                return new_team_row
            elif table_name == "config":
                """
                For each param,
                get the existing table values

                Add the new values

                Update DB
                """
                tasks = []
                for k, v in data.items():
                    updated_data = v
                    updated_data = json.dumps(updated_data)
                    updated_table_row = self.db.litellm_config.upsert(
                        where={"param_name": k},  # type: ignore
                        data={
                            "create": {"param_name": k, "param_value": updated_data},  # type: ignore
                            "update": {"param_value": updated_data},
                        },
                    )

                    tasks.append(updated_table_row)
                await asyncio.gather(*tasks)
                verbose_proxy_logger.info("Data Inserted into Config Table")
            elif table_name == "spend":
                db_data = self.jsonify_object(data=data)
                new_spend_row = await self.db.litellm_spendlogs.upsert(
                    where={"request_id": data["request_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Spend Table")
                return new_spend_row
            elif table_name == "user_notification":
                db_data = self.jsonify_object(data=data)
                new_user_notification_row = (
                    await self.db.litellm_usernotifications.upsert(  # type: ignore
                        where={"request_id": data["request_id"]},
                        data={
                            "create": {**db_data},  # type: ignore
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                )
                verbose_proxy_logger.info("Data Inserted into Model Request Table")
                return new_user_notification_row

        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception in insert_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="insert_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def update_data(  # noqa: PLR0915
        self,
        token: Optional[str] = None,
        data: dict = {},
        data_list: Optional[List] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        query_type: Literal["update", "update_many"] = "update",
        table_name: Optional[
            Literal["user", "key", "config", "spend", "team", "enduser", "budget"]
        ] = None,
        update_key_values: Optional[dict] = None,
        update_key_values_custom_query: Optional[dict] = None,
    ):
        """
        Update existing data
        """
        verbose_proxy_logger.debug(
            f"PrismaClient: update_data, table_name: {table_name}"
        )
        start_time = time.time()
        try:
            db_data = self.jsonify_object(data=data)
            if update_key_values is not None:
                update_key_values = self.jsonify_object(data=update_key_values)
            if token is not None:
                print_verbose(f"token: {token}")
                # check if plain text or hash
                token = _hash_token_if_needed(token=token)
                db_data["token"] = token
                response = await self.db.litellm_verificationtoken.update(
                    where={"token": token},  # type: ignore
                    data={**db_data},  # type: ignore
                )
                verbose_proxy_logger.debug(
                    "\033[91m"
                    + f"DB Token Table update succeeded {response}"
                    + "\033[0m"
                )
                _data: dict = {}
                if response is not None:
                    try:
                        _data = response.model_dump()  # type: ignore
                    except Exception:
                        _data = response.dict()
                return {"token": token, "data": _data}
            elif (
                user_id is not None
                or (table_name is not None and table_name == "user")
                and query_type == "update"
            ):
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if user_id is None:
                    user_id = db_data["user_id"]
                if update_key_values is None:
                    if update_key_values_custom_query is not None:
                        update_key_values = update_key_values_custom_query
                    else:
                        update_key_values = db_data
                update_user_row = await self.db.litellm_usertable.upsert(
                    where={"user_id": user_id},  # type: ignore
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {
                            **update_key_values  # type: ignore
                        },  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m"
                    + f"DB User Table - update succeeded {update_user_row}"
                    + "\033[0m"
                )
                return {"user_id": user_id, "data": update_user_row}
            elif (
                team_id is not None
                or (table_name is not None and table_name == "team")
                and query_type == "update"
            ):
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if team_id is None:
                    team_id = db_data["team_id"]
                if update_key_values is None:
                    update_key_values = db_data
                if "team_id" not in db_data and team_id is not None:
                    db_data["team_id"] = team_id
                if "members_with_roles" in db_data and isinstance(
                    db_data["members_with_roles"], list
                ):
                    db_data["members_with_roles"] = json.dumps(
                        db_data["members_with_roles"]
                    )
                if "members_with_roles" in update_key_values and isinstance(
                    update_key_values["members_with_roles"], list
                ):
                    update_key_values["members_with_roles"] = json.dumps(
                        update_key_values["members_with_roles"]
                    )
                update_team_row = await self.db.litellm_teamtable.upsert(
                    where={"team_id": team_id},  # type: ignore
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {
                            **update_key_values  # type: ignore
                        },  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m"
                    + f"DB Team Table - update succeeded {update_team_row}"
                    + "\033[0m"
                )
                return {"team_id": team_id, "data": update_team_row}
            elif (
                table_name is not None
                and table_name == "key"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, t in enumerate(data_list):
                    # check if plain text or hash
                    if t.token.startswith("sk-"):  # type: ignore
                        t.token = self.hash_token(token=t.token)  # type: ignore
                    try:
                        data_json = self.jsonify_object(
                            data=t.model_dump(exclude_none=True)
                        )
                    except Exception:
                        data_json = self.jsonify_object(data=t.dict(exclude_none=True))
                    batcher.litellm_verificationtoken.update(
                        where={"token": t.token},  # type: ignore
                        data={**data_json},  # type: ignore
                    )
                await batcher.commit()
                print_verbose(
                    "\033[91m" + "DB Token Table update succeeded" + "\033[0m"
                )
            elif (
                table_name is not None
                and table_name == "user"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, user in enumerate(data_list):
                    try:
                        data_json = self.jsonify_object(
                            data=user.model_dump(exclude_none=True)
                        )
                    except Exception:
                        data_json = self.jsonify_object(data=user.dict())
                    batcher.litellm_usertable.upsert(
                        where={"user_id": user.user_id},  # type: ignore
                        data={
                            "create": {**data_json},  # type: ignore
                            "update": {
                                **data_json  # type: ignore
                            },  # just update user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info(
                    "\033[91m" + "DB User Table Batch update succeeded" + "\033[0m"
                )
            elif (
                table_name is not None
                and table_name == "enduser"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for enduser in data_list:
                    try:
                        data_json = self.jsonify_object(
                            data=enduser.model_dump(exclude_none=True)
                        )
                    except Exception:
                        data_json = self.jsonify_object(data=enduser.dict())
                    batcher.litellm_endusertable.upsert(
                        where={"user_id": enduser.user_id},  # type: ignore
                        data={
                            "create": {**data_json},  # type: ignore
                            "update": {
                                **data_json  # type: ignore
                            },  # just update end-user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info(
                    "\033[91m" + "DB End User Table Batch update succeeded" + "\033[0m"
                )
            elif (
                table_name is not None
                and table_name == "budget"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for budget in data_list:
                    try:
                        data_json = self.jsonify_object(
                            data=budget.model_dump(exclude_none=True)
                        )
                    except Exception:
                        data_json = self.jsonify_object(data=budget.dict())
                    batcher.litellm_budgettable.upsert(
                        where={"budget_id": budget.budget_id},  # type: ignore
                        data={
                            "create": {**data_json},  # type: ignore
                            "update": {
                                **data_json  # type: ignore
                            },  # just update end-user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info(
                    "\033[91m" + "DB Budget Table Batch update succeeded" + "\033[0m"
                )
            elif (
                table_name is not None
                and table_name == "team"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                # Batch write update queries
                batcher = self.db.batch_()
                for idx, team in enumerate(data_list):
                    try:
                        data_json = self.jsonify_team_object(
                            db_data=team.model_dump(exclude_none=True)
                        )
                    except Exception:
                        data_json = self.jsonify_object(
                            data=team.dict(exclude_none=True)
                        )
                    batcher.litellm_teamtable.upsert(
                        where={"team_id": team.team_id},  # type: ignore
                        data={
                            "create": {**data_json},  # type: ignore
                            "update": {
                                **data_json  # type: ignore
                            },  # just update user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info(
                    "\033[91m" + "DB Team Table Batch update succeeded" + "\033[0m"
                )

        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception - update_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="update_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def delete_data(
        self,
        tokens: Optional[List] = None,
        team_id_list: Optional[List] = None,
        table_name: Optional[Literal["user", "key", "config", "spend", "team"]] = None,
        user_id: Optional[str] = None,
    ):
        """
        Allow user to delete a key(s)

        Ensure user owns that key, unless admin.
        """
        start_time = time.time()
        try:
            if tokens is not None and isinstance(tokens, List):
                hashed_tokens = []
                for token in tokens:
                    if isinstance(token, str) and token.startswith("sk-"):
                        hashed_token = self.hash_token(token=token)
                    else:
                        hashed_token = token
                    hashed_tokens.append(hashed_token)
                filter_query: dict = {}
                if user_id is not None:
                    filter_query = {
                        "AND": [{"token": {"in": hashed_tokens}}, {"user_id": user_id}]
                    }
                else:
                    filter_query = {"token": {"in": hashed_tokens}}

                deleted_tokens = await self.db.litellm_verificationtoken.delete_many(
                    where=filter_query  # type: ignore
                )
                verbose_proxy_logger.debug("deleted_tokens: %s", deleted_tokens)
                return {"deleted_keys": deleted_tokens}
            elif (
                table_name == "team"
                and team_id_list is not None
                and isinstance(team_id_list, List)
            ):
                # admin only endpoint -> `/team/delete`
                await self.db.litellm_teamtable.delete_many(
                    where={"team_id": {"in": team_id_list}}
                )
                return {"deleted_teams": team_id_list}
            elif (
                table_name == "key"
                and team_id_list is not None
                and isinstance(team_id_list, List)
            ):
                # admin only endpoint -> `/team/delete`
                await self.db.litellm_verificationtoken.delete_many(
                    where={"team_id": {"in": team_id_list}}
                )
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception - delete_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="delete_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def connect(self):
        start_time = time.time()
        try:
            verbose_proxy_logger.debug(
                "PrismaClient: connect() called Attempting to Connect to DB"
            )
            if self.db.is_connected() is False:
                verbose_proxy_logger.debug(
                    "PrismaClient: DB not connected, Attempting to Connect to DB"
                )
                await self.db.connect()
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception connect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="connect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def disconnect(self):
        start_time = time.time()
        try:
            await self.db.disconnect()
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception disconnect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="disconnect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=10,
        on_backoff=on_backoff,
    )
    async def health_check(self):
        """
        Health check endpoint for the prisma client
        """
        start_time = time.time()
        try:
            sql_query = "SELECT 1"

            # Execute the raw query
            # The asterisk before `user_id_list` unpacks the list into separate arguments
            response = await self.db.query_raw(sql_query)
            return response
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception disconnect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="health_check",
                    traceback_str=error_traceback,
                )
            )
            raise e

    async def _get_spend_logs_row_count(self) -> int:
        """
        Get the row count from LiteLLM_SpendLogs table using PostgreSQL system statistics.
        """

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=3,
            max_time=10,
            on_backoff=on_backoff,
        )
        async def _fetch_row_count() -> int:
            sql_query = """
            SELECT reltuples::BIGINT
            FROM pg_class
            WHERE oid = '"LiteLLM_SpendLogs"'::regclass;
            """
            result = await self.db.query_raw(query=sql_query)
            return result[0]["reltuples"]

        try:
            return await _fetch_row_count()
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error getting LiteLLM_SpendLogs row count: {e}"
            )
            return 0

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=10,
        on_backoff=on_backoff,
    )
    async def _set_spend_logs_row_count_in_proxy_state(self) -> None:
        """
        Set the `LiteLLM_SpendLogs`row count in proxy state.

        This is used later to determine if we should run expensive UI Usage queries.
        """
        from litellm.proxy.proxy_server import proxy_state

        _num_spend_logs_rows = await self._get_spend_logs_row_count()
        proxy_state.set_proxy_state_variable(
            variable_name="spend_logs_row_count",
            value=_num_spend_logs_rows,
        )

    # Health Check Database Methods
    def _validate_response_time(
        self, response_time_ms: Optional[float]
    ) -> Optional[float]:
        """Validate and clean response time value"""
        if response_time_ms is None:
            return None
        try:
            value = float(response_time_ms)
            return (
                value
                if value == value and value not in (float("inf"), float("-inf"))
                else None
            )
        except (ValueError, TypeError):
            verbose_proxy_logger.warning(
                f"Invalid response_time_ms value: {response_time_ms}"
            )
            return None

    def _clean_details(self, details: Optional[dict]) -> Optional[dict]:
        """Clean and validate details JSON"""
        if not isinstance(details, dict):
            return None
        try:
            return safe_json_loads(safe_dumps(details))
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to clean details JSON: {e}")
            return None

    async def save_health_check_result(
        self,
        model_name: str,
        status: str,
        healthy_count: int = 0,
        unhealthy_count: int = 0,
        error_message: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        details: Optional[dict] = None,
        checked_by: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """Save health check result to database"""
        try:
            # Build base data with required fields
            health_check_data = {
                "model_name": str(model_name),
                "status": str(status),
                "healthy_count": int(healthy_count),
                "unhealthy_count": int(unhealthy_count),
            }

            # Add optional fields using dict comprehension and helper methods
            optional_fields = {
                "error_message": str(error_message)[:500] if error_message else None,
                "response_time_ms": self._validate_response_time(response_time_ms),
                "details": self._clean_details(details),
                "checked_by": str(checked_by) if checked_by else None,
                "model_id": str(model_id) if model_id else None,
            }

            # Add only non-None optional fields
            health_check_data.update(
                {k: v for k, v in optional_fields.items() if v is not None}
            )

            verbose_proxy_logger.debug(f"Saving health check data: {health_check_data}")
            return await self.db.litellm_healthchecktable.create(data=health_check_data)

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error saving health check result for model {model_name}: {e}"
            )
            return None

    async def get_health_check_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ):
        """
        Get health check history with optional filtering
        """
        try:
            where_clause = {}
            if model_name:
                where_clause["model_name"] = model_name
            if status_filter:
                where_clause["status"] = status_filter

            results = await self.db.litellm_healthchecktable.find_many(
                where=where_clause,
                order={"checked_at": "desc"},
                take=limit,
                skip=offset,
            )
            return results
        except Exception as e:
            verbose_proxy_logger.error(f"Error getting health check history: {e}")
            return []

    async def get_all_latest_health_checks(self):
        """
        Get the latest health check for each model
        """
        try:
            # Get all unique model names first
            all_checks = await self.db.litellm_healthchecktable.find_many(
                order={"checked_at": "desc"}
            )

            # Group by model_name and get the latest for each
            latest_checks = {}
            for check in all_checks:
                # Create a unique key: prefer model_id if available, otherwise use model_name
                # This ensures we get the latest check for each unique model
                if check.model_id:
                    key = (check.model_id, check.model_name)
                else:
                    key = (None, check.model_name)

                # Only add if we haven't seen this key yet (since checks are ordered by checked_at desc)
                if key not in latest_checks:
                    latest_checks[key] = check

            return list(latest_checks.values())
        except Exception as e:
            verbose_proxy_logger.error(f"Error getting all latest health checks: {e}")
            return []


### HELPER FUNCTIONS ###


async def _cache_user_row(user_id: str, cache: DualCache, db: PrismaClient):
    """
    Check if a user_id exists in cache,
    if not retrieve it.
    """
    cache_key = f"{user_id}_user_api_key_user_id"
    response = cache.get_cache(key=cache_key)
    if response is None:  # Cache miss
        user_row = await db.get_data(user_id=user_id)
        if user_row is not None:
            print_verbose(f"User Row: {user_row}, type = {type(user_row)}")
            if hasattr(user_row, "model_dump_json") and callable(
                getattr(user_row, "model_dump_json")
            ):
                cache_value = user_row.model_dump_json()
                cache.set_cache(
                    key=cache_key, value=cache_value, ttl=600
                )  # store for 10 minutes
    return


async def send_email(
    receiver_email: Optional[str] = None,
    subject: Optional[str] = None,
    html: Optional[str] = None,
):
    """
    smtp_host,
    smtp_port,
    smtp_username,
    smtp_password,
    sender_name,
    sender_email,
    """
    ## SERVER SETUP ##

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))  # default to port 587
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SMTP_SENDER_EMAIL", None)
    if sender_email is None:
        raise ValueError("Trying to use SMTP, but SMTP_SENDER_EMAIL is not set")
    if receiver_email is None:
        raise ValueError(f"No receiver email provided for SMTP email. {receiver_email}")
    if subject is None:
        raise ValueError(f"No subject provided for SMTP email. {subject}")
    if html is None:
        raise ValueError(f"No HTML body provided for SMTP email. {html}")

    ## EMAIL SETUP ##
    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject
    verbose_proxy_logger.debug(
        "sending email from %s to %s", sender_email, receiver_email
    )

    if smtp_host is None:
        raise ValueError("Trying to use SMTP, but SMTP_HOST is not set")

    # Attach the body to the email
    email_message.attach(MIMEText(html, "html"))

    try:
        # Establish a secure connection with the SMTP server
        with smtplib.SMTP(
            host=smtp_host,
            port=smtp_port,
        ) as server:
            if os.getenv("SMTP_TLS", "True") != "False":
                server.starttls()

            # Login to your email account only if smtp_username and smtp_password are provided
            if smtp_username and smtp_password:
                server.login(
                    user=smtp_username,
                    password=smtp_password,
                )

            # Send the email
            server.send_message(
                msg=email_message,
                from_addr=sender_email,
                to_addrs=receiver_email,
            )

    except Exception as e:
        verbose_proxy_logger.exception(
            "An error occurred while sending the email:" + str(e)
        )


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


def _hash_token_if_needed(token: str) -> str:
    """
    Hash the token if it's a string and starts with "sk-"

    Else return the token as is
    """
    if token.startswith("sk-"):
        return hash_token(token=token)
    else:
        return token


class ProxyUpdateSpend:
    @staticmethod
    async def update_end_user_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        end_user_list_transactions: Dict[str, float],
    ):
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            end_user_id,
                            response_cost,
                        ) in end_user_list_transactions.items():
                            if litellm.max_end_user_budget is not None:
                                pass
                            batcher.litellm_endusertable.upsert(
                                where={"user_id": end_user_id},
                                data={
                                    "create": {
                                        "user_id": end_user_id,
                                        "spend": response_cost,
                                        "blocked": False,
                                    },
                                    "update": {"spend": {"increment": response_cost}},
                                },
                            )

                break
            except DB_CONNECTION_ERROR_TYPES as e:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                _raise_failed_update_spend_exception(
                    e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                )

    @staticmethod
    async def update_spend_logs(
        n_retry_times: int,
        prisma_client: PrismaClient,
        db_writer_client: Optional[AsyncHTTPHandler],
        proxy_logging_obj: ProxyLogging,
    ):
        BATCH_SIZE = 1000  # Preferred size of each batch to write to the database
        MAX_LOGS_PER_INTERVAL = (
            10000  # Maximum number of logs to flush in a single interval
        )
        # Atomically read and remove logs to process (protected by lock)
        async with prisma_client._spend_log_transactions_lock:
            logs_to_process = prisma_client.spend_log_transactions[
                :MAX_LOGS_PER_INTERVAL
            ]
            # Remove the logs we're about to process
            prisma_client.spend_log_transactions = prisma_client.spend_log_transactions[
                len(logs_to_process) :
            ]
        start_time = time.time()
        try:
            for i in range(n_retry_times + 1):
                try:
                    base_url = os.getenv("SPEND_LOGS_URL", None)
                    if (
                        len(logs_to_process) > 0
                        and base_url is not None
                        and db_writer_client is not None
                    ):
                        if not base_url.endswith("/"):
                            base_url += "/"
                        verbose_proxy_logger.debug("base_url: {}".format(base_url))
                        json_data = json.dumps(logs_to_process)
                        response = await db_writer_client.post(
                            url=base_url + "spend/update",
                            data=json_data,
                            headers={"Content-Type": "application/json"},
                        )
                        del json_data
                        if response.status_code == 200:
                            # Items already removed from queue at start of function
                            pass
                    else:
                        for j in range(0, len(logs_to_process), BATCH_SIZE):
                            batch = logs_to_process[j : j + BATCH_SIZE]
                            batch_with_dates = [
                                prisma_client.jsonify_object({**entry})
                                for entry in batch
                            ]
                            await prisma_client.db.litellm_spendlogs.create_many(
                                data=batch_with_dates, skip_duplicates=True
                            )
                            verbose_proxy_logger.debug(
                                f"Flushed {len(batch)} logs to the DB."
                            )
                            # Explicitly clear batch memory
                            del batch, batch_with_dates

                        # Items already removed from queue at start of function
                        async with prisma_client._spend_log_transactions_lock:
                            remaining_count = len(prisma_client.spend_log_transactions)
                        verbose_proxy_logger.debug(
                            f"{len(logs_to_process)} logs processed. Remaining in queue: {remaining_count}"
                        )
                    break
                except DB_CONNECTION_ERROR_TYPES:
                    if i is None:
                        i = 0
                    if i >= n_retry_times:
                        raise
                    await asyncio.sleep(2**i)
        except Exception as e:
            # Logs already removed from queue at start - don't put them back
            # This matches the original behavior where logs are removed even on error
            _raise_failed_update_spend_exception(
                e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
            )
        finally:
            # Clean up logs_to_process after all processing is complete
            del logs_to_process

    @staticmethod
    def disable_spend_updates() -> bool:
        """
        returns True if should not update spend in db
        Skips writing spend logs and updates to key, team, user spend to DB
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_spend_updates") is True:
            return True
        return False


async def update_spend(  # noqa: PLR0915
    prisma_client: PrismaClient,
    db_writer_client: Optional[AsyncHTTPHandler],
    proxy_logging_obj: ProxyLogging,
):
    """
    Batch write updates to db.

    Triggered every minute.

    Requires:
    user_id_list: dict,
    keys_list: list,
    team_list: list,
    spend_logs: list,
    """
    n_retry_times = 3
    await proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler(
        prisma_client=prisma_client,
        n_retry_times=n_retry_times,
        proxy_logging_obj=proxy_logging_obj,
    )

    ### UPDATE SPEND LOGS ###
    # Check queue size with lock protection
    async with prisma_client._spend_log_transactions_lock:
        queue_size = len(prisma_client.spend_log_transactions)
    verbose_proxy_logger.debug("Spend Logs transactions: {}".format(queue_size))

    # Process spend log transactions when called directly.
    # This keeps backwards compatibility with the old behavior.
    # See update_spend_logs_job and _monitor_spend_logs_queue for the new behavior.
    # Safe to keep: under high concurrency this can take up to ~30s to run,
    # so it's unlikely to overlap with monitor_spend_logs_queue.
    if queue_size > 0:
        await update_spend_logs_job(
            prisma_client=prisma_client,
            db_writer_client=db_writer_client,
            proxy_logging_obj=proxy_logging_obj,
        )


async def update_spend_logs_job(
    prisma_client: PrismaClient,
    db_writer_client: Optional[AsyncHTTPHandler],
    proxy_logging_obj: ProxyLogging,
):
    """
    Job to process spend_log_transactions queue.

    This job is triggered based on queue size rather than time.
    Processes spend log transactions when the queue reaches a threshold.
    """
    n_retry_times = 3

    # Check queue size with lock protection
    async with prisma_client._spend_log_transactions_lock:
        queue_size = len(prisma_client.spend_log_transactions)

    if queue_size == 0:
        return

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=n_retry_times,
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        db_writer_client=db_writer_client,
    )


async def _monitor_spend_logs_queue(
    prisma_client: PrismaClient,
    db_writer_client: Optional[AsyncHTTPHandler],
    proxy_logging_obj: ProxyLogging,
):
    """
    Background task that monitors the spend_log_transactions queue size
    and triggers processing when the threshold is reached.

    Args:
        prisma_client: Prisma client instance
        db_writer_client: Optional HTTP handler for external spend logs endpoint
        proxy_logging_obj: Proxy logging object
    """
    from litellm.constants import (
        SPEND_LOG_QUEUE_POLL_INTERVAL,
        SPEND_LOG_QUEUE_SIZE_THRESHOLD,
    )

    threshold = SPEND_LOG_QUEUE_SIZE_THRESHOLD
    base_interval = SPEND_LOG_QUEUE_POLL_INTERVAL
    max_backoff = 30.0  # Maximum backoff interval in seconds
    backoff_multiplier = 1.5  # Exponential backoff multiplier
    current_interval = base_interval

    verbose_proxy_logger.info(
        f"Starting spend logs queue monitor (threshold: {threshold}, poll_interval: {base_interval}s)"
    )

    while True:
        try:
            # Check queue size with lock protection
            async with prisma_client._spend_log_transactions_lock:
                queue_size = len(prisma_client.spend_log_transactions)

            if queue_size > 0:
                if queue_size >= threshold:
                    verbose_proxy_logger.debug(
                        f"Spend logs queue size ({queue_size}) reached threshold ({threshold}), triggering processing"
                    )
                    # Reset to base interval when threshold is reached
                    current_interval = base_interval
                else:
                    verbose_proxy_logger.debug(
                        f"Spend logs queue size ({queue_size}) below threshold ({threshold}), processing with backoff"
                    )
                    # Exponential backoff when below threshold but still processing
                    current_interval = min(
                        current_interval * backoff_multiplier, max_backoff
                    )

                await update_spend_logs_job(
                    prisma_client=prisma_client,
                    db_writer_client=db_writer_client,
                    proxy_logging_obj=proxy_logging_obj,
                )
            else:
                # Exponential backoff when no logs to process
                current_interval = min(
                    current_interval * backoff_multiplier, max_backoff
                )

            await asyncio.sleep(current_interval)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in spend logs queue monitor: {str(e)}\n{traceback.format_exc()}"
            )
            # Continue monitoring even if there's an error, with exponential backoff
            current_interval = min(current_interval * backoff_multiplier, max_backoff)
            await asyncio.sleep(current_interval)


def _raise_failed_update_spend_exception(
    e: Exception, start_time: float, proxy_logging_obj: ProxyLogging
):
    """
    Raise an exception for failed update spend logs

    - Calls proxy_logging_obj.failure_handler to log the error
    - Ensures error messages says "Non-Blocking"
    """
    import traceback

    error_msg = (
        f"[Non-Blocking]LiteLLM Prisma Client Exception - update spend logs: {str(e)}"
    )
    error_traceback = error_msg + "\n" + traceback.format_exc()
    end_time = time.time()
    _duration = end_time - start_time
    asyncio.create_task(
        proxy_logging_obj.failure_handler(
            original_exception=e,
            duration=_duration,
            call_type="update_spend",
            traceback_str=error_traceback,
        )
    )
    raise e


def _get_month_end_date(today: date) -> date:
    if today.month == 12:
        return date(today.year + 1, 1, 1) - timedelta(days=1)
    return date(today.year, today.month + 1, 1) - timedelta(days=1)


def _is_projected_spend_over_limit(
    current_spend: float, soft_budget_limit: Optional[float]
):
    if soft_budget_limit is None:
        # If there's no limit, we can't exceed it.
        return False

    today = date.today()

    # Finding the first day of the next month, then subtracting one day to get the end of the current month.
    end_month = _get_month_end_date(today)

    remaining_days = (end_month - today).days

    # Check for the start of the month to avoid division by zero
    if today.day == 1:
        daily_spend_estimate = current_spend
    else:
        daily_spend_estimate = current_spend / (today.day - 1)

    # Total projected spend for the month
    projected_spend = current_spend + (daily_spend_estimate * remaining_days)

    if projected_spend > soft_budget_limit:
        print_verbose("Projected spend exceeds soft budget limit!")
        return True
    return False


def _get_projected_spend_over_limit(
    current_spend: float, soft_budget_limit: Optional[float]
) -> Optional[tuple]:
    if soft_budget_limit is None:
        return None

    today = date.today()
    end_month = _get_month_end_date(today)
    remaining_days = (end_month - today).days

    # assuming the current spend till today (not including today)
    if today.day == 1:
        daily_spend = current_spend
    else:
        daily_spend = current_spend / (today.day - 1)
    projected_spend = current_spend + (daily_spend * remaining_days)

    if projected_spend > soft_budget_limit:
        if daily_spend <= 0:
            limit_exceed_date = today
        else:
            remaining_budget = soft_budget_limit - current_spend
            if remaining_budget <= 0:
                limit_exceed_date = today
            else:
                approx_days = remaining_budget / daily_spend
                limit_exceed_date = today + timedelta(days=approx_days)

        # return the projected spend and the date it will exceeded
        return projected_spend, limit_exceed_date

    return None


def _is_valid_team_configs(team_id=None, team_config=None, request_data=None):
    if team_id is None or team_config is None or request_data is None:
        return
    # check if valid model called for team
    if "models" in team_config:
        valid_models = team_config.pop("models")
        model_in_request = request_data["model"]
        if model_in_request not in valid_models:
            raise Exception(
                f"Invalid model for team {team_id}: {model_in_request}.  Valid models for team are: {valid_models}\n"
            )
    return


def _to_ns(dt):
    return int(dt.timestamp() * 1e9)


def _check_and_merge_model_level_guardrails(
    data: dict, llm_router: Optional[Router]
) -> dict:
    """
    Check if the model has guardrails defined and merge them with existing guardrails in the request data.

    Args:
        data: The request data dict
        llm_router: The LLM router instance to get deployment info from

    Returns:
        Modified data dict with merged guardrails (if any model-level guardrails exist)
    """
    if llm_router is None:
        return data

    # Get the model ID from the data
    metadata = data.get("metadata") or {}
    model_info = metadata.get("model_info") or {}
    model_id = model_info.get("id", None)

    if model_id is None:
        return data

    # Check if the model has guardrails
    deployment = llm_router.get_deployment(model_id=model_id)
    if deployment is None:
        return data

    model_level_guardrails = deployment.litellm_params.get("guardrails")

    if model_level_guardrails is None:
        return data

    # Merge model-level guardrails with existing ones
    return _merge_guardrails_with_existing(data, model_level_guardrails)


def _merge_guardrails_with_existing(data: dict, model_level_guardrails: Any) -> dict:
    """
    Merge model-level guardrails with any existing guardrails in the request data.

    Args:
        data: The request data dict
        model_level_guardrails: Guardrails defined at the model level

    Returns:
        Modified data dict with merged guardrails in metadata
    """
    modified_data = data.copy()
    metadata = modified_data.setdefault("metadata", {})
    existing_guardrails = metadata.get("guardrails", [])

    # Ensure existing_guardrails is a list
    if not isinstance(existing_guardrails, list):
        existing_guardrails = [existing_guardrails] if existing_guardrails else []

    # Ensure model_level_guardrails is a list
    if not isinstance(model_level_guardrails, list):
        model_level_guardrails = (
            [model_level_guardrails] if model_level_guardrails else []
        )

    # Combine existing and model-level guardrails
    metadata["guardrails"] = list(set(existing_guardrails + model_level_guardrails))
    return modified_data


def get_error_message_str(e: Exception) -> str:
    error_message = ""
    if isinstance(e, HTTPException):
        if isinstance(e.detail, str):
            error_message = e.detail
        elif isinstance(e.detail, dict):
            error_message = json.dumps(e.detail)
        elif hasattr(e, "message"):
            _error = getattr(e, "message", None)
            if isinstance(_error, str):
                error_message = _error
            elif isinstance(_error, dict):
                error_message = json.dumps(_error)
        else:
            error_message = str(e)
    else:
        error_message = str(e)
    return error_message


def _get_redoc_url() -> Optional[str]:
    """
    Get the Redoc URL from the environment variables.

    - If REDOC_URL is set, return it.
    - If NO_REDOC is True, return None.
    - Otherwise, default to "/redoc".
    """
    if redoc_url := os.getenv("REDOC_URL"):
        return redoc_url

    if str_to_bool(os.getenv("NO_REDOC")) is True:
        return None

    return "/redoc"


def _get_docs_url() -> Optional[str]:
    """
    Get the docs (Swagger UI) URL from the environment variables.

    - If DOCS_URL is set, return it.
    - If NO_DOCS is True, return None.
    - Otherwise, default to "/".
    """
    if docs_url := os.getenv("DOCS_URL"):
        return docs_url

    if str_to_bool(os.getenv("NO_DOCS")) is True:
        return None

    return "/"


def handle_exception_on_proxy(e: Exception) -> ProxyException:
    """
    Returns an Exception as ProxyException, this ensures all exceptions are OpenAI API compatible
    """
    from fastapi import status

    verbose_proxy_logger.exception(f"Exception: {e}")

    if isinstance(e, HTTPException):
        return ProxyException(
            message=getattr(e, "detail", f"error({str(e)})"),
            type=ProxyErrorTypes.internal_server_error,
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
        )
    elif isinstance(e, ProxyException):
        return e
    return ProxyException(
        message="Internal Server Error, " + str(e),
        type=ProxyErrorTypes.internal_server_error,
        param=getattr(e, "param", "None"),
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _premium_user_check(feature: Optional[str] = None):
    """
    Raises an HTTPException if the user is not a premium user
    """
    from litellm.proxy.proxy_server import premium_user

    if feature:
        detail_msg = f"This feature is only available for LiteLLM Enterprise users: {feature}. {CommonProxyErrors.not_premium_user.value}"
    else:
        detail_msg = f"This feature is only available for LiteLLM Enterprise users. {CommonProxyErrors.not_premium_user.value}"

    if not premium_user:
        raise HTTPException(
            status_code=403,
            detail={"error": detail_msg},
        )


def is_known_model(model: Optional[str], llm_router: Optional[Router]) -> bool:
    """
    Returns True if the model is in the llm_router model names
    """
    if model is None or llm_router is None:
        return False
    model_names = llm_router.get_model_names()

    model_names_set = set(model_names)

    is_in_list = False
    if model in model_names_set:
        is_in_list = True

    return is_in_list


def is_known_vector_store_index(index_name: str) -> bool:
    """
    Returns True if the vector store index is in the llm_router vector store indexes
    """

    if litellm.vector_store_index_registry is None:
        return False
    return index_name in litellm.vector_store_index_registry.get_vector_store_indexes()


def join_paths(base_path: str, route: str) -> str:
    # Remove trailing slashes from base_path and leading slashes from route
    base_path = base_path.rstrip("/")
    route = route.lstrip("/")

    # If base_path is empty, return route with leading slash
    if not base_path:
        return f"/{route}" if route else "/"

    # If route is empty, return just base_path
    if not route:
        return base_path

    # Check if base_path already ends with the route to avoid duplication
    if base_path.endswith(f"/{route}"):
        final_path = base_path
    else:
        # Join with single slash
        final_path = f"{base_path}/{route}"

    return final_path


def get_custom_url(request_base_url: str, route: Optional[str] = None) -> str:
    # Use environment variable value, otherwise use URL from request
    server_base_url = get_proxy_base_url()
    if server_base_url is not None:
        base_url = server_base_url
    else:
        base_url = request_base_url

    server_root_path = get_server_root_path()
    if route is not None:
        if server_root_path != "":
            # First join base_url with server_root_path, then with route
            intermediate_url = join_paths(base_url, server_root_path)
            return join_paths(intermediate_url, route)
        else:
            return join_paths(base_url, route)
    else:
        return join_paths(base_url, server_root_path)


def get_proxy_base_url() -> Optional[str]:
    """
    Get the proxy base url from the environment variables.
    """
    return os.getenv("PROXY_BASE_URL")


def get_server_root_path() -> str:
    """
    Get the server root path from the environment variables.

    - If SERVER_ROOT_PATH is set, return it.
    - Otherwise, default to "/".
    """
    return os.getenv("SERVER_ROOT_PATH", "")


def get_prisma_client_or_throw(message: str):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": message},
        )
    return prisma_client


def is_valid_api_key(key: str) -> bool:
    """
    Validates API key format:
    - sk- keys: must match ^sk-[A-Za-z0-9_-]+$
    - hashed keys: must match ^[a-fA-F0-9]{64}$
    - Length between 20 and 100 characters
    """
    import re

    if not isinstance(key, str):
        return False
    if 3 <= len(key) <= 100:
        if re.match(r"^sk-[A-Za-z0-9_-]+$", key):
            return True
        if re.match(r"^[a-fA-F0-9]{64}$", key):
            return True
    return False


def construct_database_url_from_env_vars() -> Optional[str]:
    """
    Construct a DATABASE_URL from individual environment variables.
    Returns:
        Optional[str]: The constructed DATABASE_URL or None if required variables are missing
    """
    import urllib.parse

    # Check if all required variables are provided
    database_host = os.getenv("DATABASE_HOST")
    database_username = os.getenv("DATABASE_USERNAME")
    database_password = os.getenv("DATABASE_PASSWORD")
    database_name = os.getenv("DATABASE_NAME")
    database_schema = os.getenv("DATABASE_SCHEMA")

    if database_host and database_username and database_name:
        # Handle the problem of special character escaping in the database URL
        database_username_enc = urllib.parse.quote_plus(database_username)
        database_password_enc = (
            urllib.parse.quote_plus(database_password) if database_password else ""
        )
        database_name_enc = urllib.parse.quote_plus(database_name)

        # Construct DATABASE_URL from the provided variables
        if database_password:
            database_url = f"postgresql://{database_username_enc}:{database_password_enc}@{database_host}/{database_name_enc}"
        else:
            database_url = f"postgresql://{database_username_enc}@{database_host}/{database_name_enc}"

        if database_schema:
            database_url += f"?schema={database_schema}"

        return database_url

    return None


async def get_available_models_for_user(
    user_api_key_dict: "UserAPIKeyAuth",
    llm_router: Optional["Router"],
    general_settings: dict,
    user_model: Optional[str],
    prisma_client: Optional["PrismaClient"] = None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
    team_id: Optional[str] = None,
    include_model_access_groups: bool = False,
    only_model_access_groups: bool = False,
    return_wildcard_routes: bool = False,
    user_api_key_cache: Optional["DualCache"] = None,
) -> List[str]:
    """
    Get the list of models available to a user based on their API key and team permissions.

    Args:
        user_api_key_dict: User API key authentication object
        llm_router: LiteLLM router instance
        general_settings: General settings from config
        user_model: User-specific model
        prisma_client: Prisma client for database operations
        proxy_logging_obj: Proxy logging object
        team_id: Specific team ID to check (optional)
        include_model_access_groups: Whether to include model access groups
        only_model_access_groups: Whether to only return model access groups
        return_wildcard_routes: Whether to return wildcard routes

    Returns:
        List of model names available to the user
    """
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.auth.model_checks import (
        get_complete_model_list,
        get_key_models,
        get_team_models,
    )
    from litellm.proxy.management_endpoints.team_endpoints import validate_membership

    # Get proxy model list and access groups
    if llm_router is None:
        proxy_model_list = []
        model_access_groups = {}
    else:
        proxy_model_list = llm_router.get_model_names()
        model_access_groups = llm_router.get_model_access_groups()

    # Get key models
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
        include_model_access_groups=include_model_access_groups,
    )

    # Get team models
    team_models: List[str] = user_api_key_dict.team_models

    # If specific team_id is provided, validate and get team models
    if team_id and prisma_client and proxy_logging_obj and user_api_key_cache:
        key_models = []
        team_object = await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        validate_membership(user_api_key_dict=user_api_key_dict, team_table=team_object)
        team_models = team_object.models

    team_models = get_team_models(
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
        include_model_access_groups=include_model_access_groups,
    )

    # Get complete model list
    all_models = get_complete_model_list(
        key_models=key_models,
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
        return_wildcard_routes=return_wildcard_routes,
        llm_router=llm_router,
        model_access_groups=model_access_groups,
        include_model_access_groups=include_model_access_groups,
        only_model_access_groups=only_model_access_groups,
    )

    return all_models


def create_model_info_response(
    model_id: str,
    provider: str,
    include_metadata: bool = False,
    fallback_type: Optional[str] = None,
    llm_router: Optional["Router"] = None,
) -> dict:
    """
    Create a standardized model info response.

    Args:
        model_id: The model ID
        provider: The model provider
        include_metadata: Whether to include metadata
        fallback_type: Type of fallbacks to include
        llm_router: LiteLLM router instance

    Returns:
        Dictionary containing model information
    """
    from litellm.proxy.auth.model_checks import get_all_fallbacks

    model_info = {
        "id": model_id,
        "object": "model",
        "created": DEFAULT_MODEL_CREATED_AT_TIME,
        "owned_by": provider,
    }

    # Add metadata if requested
    if include_metadata:
        metadata = {}

        # Default fallback_type to "general" if include_metadata is true
        effective_fallback_type = (
            fallback_type if fallback_type is not None else "general"
        )

        # Validate fallback_type
        valid_fallback_types = ["general", "context_window", "content_policy"]
        if effective_fallback_type not in valid_fallback_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid fallback_type. Must be one of: {valid_fallback_types}",
            )

        fallbacks = get_all_fallbacks(
            model=model_id,
            llm_router=llm_router,
            fallback_type=effective_fallback_type,
        )
        metadata["fallbacks"] = fallbacks

        model_info["metadata"] = metadata

    return model_info


def validate_model_access(
    model_id: str,
    available_models: List[str],
) -> None:
    """
    Validate that a model is accessible to the user.
    Supports batch requests with comma-separated model IDs.

    Args:
        model_id: The model ID to validate (can be comma-separated for batch requests)
        available_models: List of models available to the user

    Raises:
        HTTPException: If the model is not accessible
    """
    # Handle batch requests with comma-separated models
    if "," in model_id:
        models = [m.strip() for m in model_id.split(",")]
        inaccessible_models = [m for m in models if m not in available_models]
        if inaccessible_models:
            raise HTTPException(
                status_code=404,
                detail="The following model(s) do not exist or are not accessible: {}".format(
                    ", ".join(inaccessible_models)
                ),
            )
    else:
        # Single model validation
        if model_id not in available_models:
            raise HTTPException(
                status_code=404,
                detail="The model `{}` does not exist or is not accessible".format(
                    model_id
                ),
            )


def _path_matches_pattern(path: str, pattern: str) -> bool:
    """Check if a path matches a pattern (supporting * wildcard for list indices)."""
    path_parts = path.split(".")
    pattern_parts = pattern.split(".")

    if len(path_parts) != len(pattern_parts):
        return False

    for path_part, pattern_part in zip(path_parts, pattern_parts):
        if pattern_part == "*":
            # Wildcard matches any numeric index
            if not path_part.isdigit():
                return False
        elif path_part != pattern_part:
            return False

    return True


def _build_preserved_paths(
    data: Any, current_path: str, preserve_fields: List[str], preserved_paths: set
) -> None:
    """Iteratively build set of paths that should be preserved."""
    # Use a stack to avoid recursion: (data, path)
    stack = [(data, current_path)]

    while stack:
        current_data, current_path_str = stack.pop()

        if isinstance(current_data, dict):
            for key, value in current_data.items():
                new_path = f"{current_path_str}.{key}" if current_path_str else key

                # Check if this path matches any preserve pattern
                for pattern in preserve_fields:
                    if _path_matches_pattern(new_path, pattern):
                        preserved_paths.add(new_path)

                if isinstance(value, (dict, list)):
                    stack.append((value, new_path))

        elif isinstance(current_data, list):
            for idx, item in enumerate(current_data):
                new_path = f"{current_path_str}.{idx}" if current_path_str else str(idx)
                if isinstance(item, (dict, list)):
                    stack.append((item, new_path))


def _remove_none_except_preserved(
    data: Any, current_path: str, preserved_paths: set
) -> Any:
    """Iteratively remove None values except for preserved paths."""
    if not isinstance(data, (dict, list)):
        return data

    # Use a stack for iterative processing: (data, path, is_first_visit)
    # We'll process in a way that allows us to build the result bottom-up
    stack = [(data, current_path, True)]  # (data, path, is_first_visit)
    results_map: dict[int, Any] = {}  # Maps id(data) -> processed result

    while stack:
        current_data, current_path_str, is_first_visit = stack.pop()

        if is_first_visit:
            # First visit - mark for revisit and add children to stack
            stack.append((current_data, current_path_str, False))

            if isinstance(current_data, dict):
                # Add children in reverse order so they're processed in correct order
                for key in reversed(list(current_data.keys())):
                    value = current_data[key]
                    new_path = f"{current_path_str}.{key}" if current_path_str else key

                    if isinstance(value, (dict, list)):
                        stack.append((value, new_path, True))

            elif isinstance(current_data, list):
                # Add children in reverse order
                for idx in reversed(range(len(current_data))):
                    item = current_data[idx]
                    new_path = (
                        f"{current_path_str}.{idx}" if current_path_str else str(idx)
                    )

                    if isinstance(item, (dict, list)):
                        stack.append((item, new_path, True))
        else:
            # Second visit - children are processed, build result
            result: Union[dict[str, Any], list[Any]]
            if isinstance(current_data, dict):
                result = {}
                for key, value in current_data.items():
                    new_path = f"{current_path_str}.{key}" if current_path_str else key

                    if value is None:
                        if new_path in preserved_paths:
                            result[key] = None
                    elif isinstance(value, (dict, list)):
                        processed = results_map.get(id(value))
                        if (
                            processed is not None
                            and processed != {}
                            and processed != []
                        ):
                            result[key] = processed
                    else:
                        result[key] = value

                results_map[id(current_data)] = result

            elif isinstance(current_data, list):
                result = []
                for idx, item in enumerate(current_data):
                    new_path = (
                        f"{current_path_str}.{idx}" if current_path_str else str(idx)
                    )

                    if item is None:
                        if new_path in preserved_paths:
                            result.append(None)
                    elif isinstance(item, (dict, list)):
                        processed = results_map.get(id(item))
                        if processed is not None:
                            result.append(processed)
                    else:
                        result.append(item)

                results_map[id(current_data)] = result

    return results_map.get(id(data), data)


def model_dump_with_preserved_fields(
    obj: Any,
    preserve_fields: Optional[List[str]] = None,
    exclude_unset: bool = True,
) -> Dict[str, Any]:
    """
    Serialize a Pydantic model to a dictionary while preserving specific fields even if they are None.

    This function is useful when you need to maintain API compatibility where certain fields
    must always be present in the response (e.g., message.content in OpenAI API responses).

    Args:
        obj: The Pydantic BaseModel instance to serialize
        preserve_fields: List of field paths to preserve even if None (e.g., ["choices.*.message.content"])
        exclude_unset: Whether to exclude fields that were not explicitly set

    Returns:
        Dictionary representation with None values excluded except for preserved fields

    Example:
        >>> result = model_dump_with_preserved_fields(
        ...     response,
        ...     preserve_fields=["choices.*.message.content", "choices.*.message.role"]
        ... )
    """
    if preserve_fields is None:
        preserve_fields = [
            "choices.*.message.content",
            "choices.*.message.role",
            "choices.*.delta.content",
        ]

    # First, get the full dump without excluding None values
    full_dump = obj.model_dump(exclude_none=False, exclude_unset=exclude_unset)

    # Build the set of preserved paths
    preserved_paths: set = set()
    _build_preserved_paths(full_dump, "", preserve_fields, preserved_paths)

    # Remove None values except for preserved paths
    return _remove_none_except_preserved(full_dump, "", preserved_paths)
