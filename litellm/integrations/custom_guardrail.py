from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import (
    DynamicGuardrailParams,
    GuardrailEventHooks,
    LitellmParams,
    PiiEntityType,
)
from litellm.types.utils import StandardLoggingGuardrailInformation


class CustomGuardrail(CustomLogger):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        supported_event_hooks: Optional[List[GuardrailEventHooks]] = None,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks]]
        ] = None,
        default_on: bool = False,
        mask_request_content: bool = False,
        mask_response_content: bool = False,
        **kwargs,
    ):
        """
        Initialize the CustomGuardrail class

        Args:
            guardrail_name: The name of the guardrail. This is the name used in your requests.
            supported_event_hooks: The event hooks that the guardrail supports
            event_hook: The event hook to run the guardrail on
            default_on: If True, the guardrail will be run by default on all requests
            mask_request_content: If True, the guardrail will mask the request content
            mask_response_content: If True, the guardrail will mask the response content
        """
        self.guardrail_name = guardrail_name
        self.supported_event_hooks = supported_event_hooks
        self.event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks]]
        ] = event_hook
        self.default_on: bool = default_on
        self.mask_request_content: bool = mask_request_content
        self.mask_response_content: bool = mask_response_content

        if supported_event_hooks:

            ## validate event_hook is in supported_event_hooks
            self._validate_event_hook(event_hook, supported_event_hooks)
        super().__init__(**kwargs)

    def _validate_event_hook(
        self,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]],
        supported_event_hooks: List[GuardrailEventHooks],
    ) -> None:

        if event_hook is None:
            return
        if isinstance(event_hook, str):
            event_hook = GuardrailEventHooks(event_hook)
        if isinstance(event_hook, list):
            for hook in event_hook:
                if isinstance(hook, str):
                    hook = GuardrailEventHooks(hook)
                if hook not in supported_event_hooks:
                    raise ValueError(
                        f"Event hook {hook} is not in the supported event hooks {supported_event_hooks}"
                    )
        elif isinstance(event_hook, GuardrailEventHooks):
            if event_hook not in supported_event_hooks:
                raise ValueError(
                    f"Event hook {event_hook} is not in the supported event hooks {supported_event_hooks}"
                )

    def get_guardrail_from_metadata(
        self, data: dict
    ) -> Union[List[str], List[Dict[str, DynamicGuardrailParams]]]:
        """
        Returns the guardrail(s) to be run from the metadata
        """
        metadata = data.get("metadata") or {}
        requested_guardrails = metadata.get("guardrails") or []
        return requested_guardrails

    def _guardrail_is_in_requested_guardrails(
        self,
        requested_guardrails: Union[List[str], List[Dict[str, DynamicGuardrailParams]]],
    ) -> bool:
        for _guardrail in requested_guardrails:
            if isinstance(_guardrail, dict):
                if self.guardrail_name in _guardrail:
                    return True
            elif isinstance(_guardrail, str):
                if self.guardrail_name == _guardrail:
                    return True
        return False

    def should_run_guardrail(self, data, event_type: GuardrailEventHooks) -> bool:
        """
        Returns True if the guardrail should be run on the event_type
        """
        requested_guardrails = self.get_guardrail_from_metadata(data)

        verbose_logger.debug(
            "inside should_run_guardrail for guardrail=%s event_type= %s guardrail_supported_event_hooks= %s requested_guardrails= %s self.default_on= %s",
            self.guardrail_name,
            event_type,
            self.event_hook,
            requested_guardrails,
            self.default_on,
        )

        if self.default_on is True:
            if self._event_hook_is_event_type(event_type):
                return True
            return False

        if (
            self.event_hook
            and not self._guardrail_is_in_requested_guardrails(requested_guardrails)
            and event_type.value != "logging_only"
        ):
            return False

        if not self._event_hook_is_event_type(event_type):
            return False

        return True

    def _event_hook_is_event_type(self, event_type: GuardrailEventHooks) -> bool:
        """
        Returns True if the event_hook is the same as the event_type

        eg. if `self.event_hook == "pre_call" and event_type == "pre_call"` -> then True
        eg. if `self.event_hook == "pre_call" and event_type == "post_call"` -> then False
        """

        if self.event_hook is None:
            return True
        if isinstance(self.event_hook, list):
            return event_type.value in self.event_hook
        return self.event_hook == event_type.value

    def get_guardrail_dynamic_request_body_params(self, request_data: dict) -> dict:
        """
        Returns `extra_body` to be added to the request body for the Guardrail API call

        Use this to pass dynamic params to the guardrail API call - eg. success_threshold, failure_threshold, etc.

        ```
        [{"lakera_guard": {"extra_body": {"foo": "bar"}}}]
        ```

        Will return: for guardrail=`lakera-guard`:
        {
            "foo": "bar"
        }

        Args:
            request_data: The original `request_data` passed to LiteLLM Proxy
        """
        requested_guardrails = self.get_guardrail_from_metadata(request_data)

        # Look for the guardrail configuration matching self.guardrail_name
        for guardrail in requested_guardrails:
            if isinstance(guardrail, dict) and self.guardrail_name in guardrail:
                # Get the configuration for this guardrail
                guardrail_config: DynamicGuardrailParams = DynamicGuardrailParams(
                    **guardrail[self.guardrail_name]
                )
                if self._validate_premium_user() is not True:
                    return {}

                # Return the extra_body if it exists, otherwise empty dict
                return guardrail_config.get("extra_body", {})

        return {}

    def _validate_premium_user(self) -> bool:
        """
        Returns True if the user is a premium user
        """
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            verbose_logger.warning(
                f"Trying to use premium guardrail without premium user {CommonProxyErrors.not_premium_user.value}"
            )
            return False
        return True

    def add_standard_logging_guardrail_information_to_request_data(
        self,
        guardrail_json_response: Union[Exception, str, dict, List[dict]],
        request_data: dict,
        guardrail_status: Literal["success", "failure"],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        masked_entity_count: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Builds `StandardLoggingGuardrailInformation` and adds it to the request metadata so it can be used for logging to DataDog, Langfuse, etc.
        """
        if isinstance(guardrail_json_response, Exception):
            guardrail_json_response = str(guardrail_json_response)
        slg = StandardLoggingGuardrailInformation(
            guardrail_name=self.guardrail_name,
            guardrail_mode=self.event_hook,
            guardrail_response=guardrail_json_response,
            guardrail_status=guardrail_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            masked_entity_count=masked_entity_count,
        )
        if "metadata" in request_data:
            if request_data["metadata"] is None:
                request_data["metadata"] = {}
            request_data["metadata"]["standard_logging_guardrail_information"] = slg
        elif "litellm_metadata" in request_data:
            request_data["litellm_metadata"][
                "standard_logging_guardrail_information"
            ] = slg
        else:
            verbose_logger.warning(
                "unable to log guardrail information. No metadata found in request_data"
            )

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[PiiEntityType]] = None,
    ) -> str:
        """
        Apply your guardrail logic to the given text

        Args:
            text: The text to apply the guardrail to
            language: The language of the text
            entities: The entities to mask, optional

        Any of the custom guardrails can override this method to provide custom guardrail logic

        Returns the text with the guardrail applied

        Raises:
            Exception:
                - If the guardrail raises an exception

        """
        return text

    def _process_response(
        self,
        response: Optional[Dict],
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """
        Add StandardLoggingGuardrailInformation to the request data

        This gets logged on downsteam Langfuse, DataDog, etc.
        """
        # Convert None to empty dict to satisfy type requirements
        guardrail_response = {} if response is None else response
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status="success",
            duration=duration,
            start_time=start_time,
            end_time=end_time,
        )
        return response

    def _process_error(
        self,
        e: Exception,
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """
        Add StandardLoggingGuardrailInformation to the request data

        This gets logged on downsteam Langfuse, DataDog, etc.
        """
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=e,
            request_data=request_data,
            guardrail_status="failure",
            duration=duration,
            start_time=start_time,
            end_time=end_time,
        )
        raise e

    def mask_content_in_string(
        self,
        content_string: str,
        mask_string: str,
        start_index: int,
        end_index: int,
    ) -> str:
        """
        Mask the content in the string between the start and end indices.
        """

        # Do nothing if the start or end are not valid
        if not (0 <= start_index < end_index <= len(content_string)):
            return content_string

        # Mask the content
        return content_string[:start_index] + mask_string + content_string[end_index:]

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        Update the guardrails litellm params in memory
        """
        pass


def log_guardrail_information(func):
    """
    Decorator to add standard logging guardrail information to any function

    Add this decorator to ensure your guardrail response is logged to DataDog, OTEL, s3, GCS etc.

    Logs for:
        - pre_call
        - during_call
        - TODO: log post_call. This is more involved since the logs are sent to DD, s3 before the guardrail is even run
    """
    import asyncio
    import functools

    start_time = datetime.now()

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        self: CustomGuardrail = args[0]
        request_data: Optional[dict] = (
            kwargs.get("data") or kwargs.get("request_data") or {}
        )
        try:
            response = await func(*args, **kwargs)
            return self._process_response(
                response=response,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return self._process_error(
                e=e,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        self: CustomGuardrail = args[0]
        request_data: Optional[dict] = (
            kwargs.get("data") or kwargs.get("request_data") or {}
        )
        try:
            response = func(*args, **kwargs)
            return self._process_response(
                response=response,
                request_data=request_data,
                duration=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return self._process_error(
                e=e,
                request_data=request_data,
                duration=(datetime.now() - start_time).total_seconds(),
            )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return async_wrapper(*args, **kwargs)
        return sync_wrapper(*args, **kwargs)

    return wrapper
