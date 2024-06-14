import asyncio
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


async def send_llm_exception_alert(
    litellm_router_instance: LitellmRouter,
    request_kwargs: dict,
    error_traceback_str: str,
    original_exception,
):
    """
    Sends a Slack / MS Teams alert for the LLM API call failure.

    Parameters:
        litellm_router_instance (_Router): The LitellmRouter instance.
        original_exception (Any): The original exception that occurred.

    Returns:
        None
    """
    if litellm_router_instance is None:
        return

    if not hasattr(litellm_router_instance, "slack_alerting_logger"):
        return

    if litellm_router_instance.slack_alerting_logger is None:
        return

    if "proxy_server_request" in request_kwargs:
        # Do not send any alert if it's a request from litellm proxy server request
        # the proxy is already instrumented to send LLM API call failures
        return

    litellm_debug_info = getattr(original_exception, "litellm_debug_info", None)
    exception_str = str(original_exception)
    if litellm_debug_info is not None:
        exception_str += litellm_debug_info
    exception_str += f"\n\n{error_traceback_str[:2000]}"

    await litellm_router_instance.slack_alerting_logger.send_alert(
        message=f"LLM API call failed: `{exception_str}`",
        level="High",
        alert_type="llm_exceptions",
    )
