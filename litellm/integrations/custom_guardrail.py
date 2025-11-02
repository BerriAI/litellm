from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, get_args

from litellm._logging import verbose_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import (
    DynamicGuardrailParams,
    GuardrailEventHooks,
    LitellmParams,
    Mode,
    PiiEntityType,
)
from litellm.types.llms.openai import (
    AllMessageValues,
)
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import (
    CallTypes,
    GuardrailStatus,
    LLMResponseTypes,
    StandardLoggingGuardrailInformation,
)

dc = DualCache()


class CustomGuardrail(CustomLogger):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        supported_event_hooks: Optional[List[GuardrailEventHooks]] = None,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
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
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ] = event_hook
        self.default_on: bool = default_on
        self.mask_request_content: bool = mask_request_content
        self.mask_response_content: bool = mask_response_content

        if supported_event_hooks:
            ## validate event_hook is in supported_event_hooks
            self._validate_event_hook(event_hook, supported_event_hooks)
        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Returns the config model for the guardrail

        This is used to render the config model in the UI.
        """
        return None

    def _validate_event_hook(
        self,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ],
        supported_event_hooks: List[GuardrailEventHooks],
    ) -> None:
        def _validate_event_hook_list_is_in_supported_event_hooks(
            event_hook: Union[List[GuardrailEventHooks], List[str]],
            supported_event_hooks: List[GuardrailEventHooks],
        ) -> None:
            for hook in event_hook:
                if isinstance(hook, str):
                    hook = GuardrailEventHooks(hook)
                if hook not in supported_event_hooks:
                    raise ValueError(
                        f"Event hook {hook} is not in the supported event hooks {supported_event_hooks}"
                    )

        if event_hook is None:
            return
        if isinstance(event_hook, str):
            event_hook = GuardrailEventHooks(event_hook)
        if isinstance(event_hook, list):
            _validate_event_hook_list_is_in_supported_event_hooks(
                event_hook, supported_event_hooks
            )
        elif isinstance(event_hook, Mode):
            _validate_event_hook_list_is_in_supported_event_hooks(
                list(event_hook.tags.values()), supported_event_hooks
            )
            if event_hook.default:
                _validate_event_hook_list_is_in_supported_event_hooks(
                    [event_hook.default], supported_event_hooks
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
        Returns the guardrail(s) to be run from the metadata or root
        """
        if "guardrails" in data:
            return data["guardrails"]
        metadata = data.get("litellm_metadata") or data.get("metadata", {})
        return metadata.get("guardrails") or []

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

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        from litellm.proxy._types import UserAPIKeyAuth

        # should run guardrail
        litellm_guardrails = kwargs.get("guardrails")
        if litellm_guardrails is None or not isinstance(litellm_guardrails, list):
            return kwargs

        if (
            self.should_run_guardrail(
                data=kwargs, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return kwargs

        # CHECK IF GUARDRAIL REJECTS THE REQUEST
        if call_type == CallTypes.completion or call_type == CallTypes.acompletion:
            result = await self.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(
                    user_id=kwargs.get("user_api_key_user_id"),
                    team_id=kwargs.get("user_api_key_team_id"),
                    end_user_id=kwargs.get("user_api_key_end_user_id"),
                    api_key=kwargs.get("user_api_key_hash"),
                    request_route=kwargs.get("user_api_key_request_route"),
                ),
                cache=dc,
                data=kwargs,
                call_type=call_type.value or "acompletion",  # type: ignore
            )

            if result is not None and isinstance(result, dict):
                result_messages = result.get("messages")
                if result_messages is not None:  # update for any pii / masking logic
                    kwargs["messages"] = result_messages

        return kwargs

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: LLMResponseTypes,
        call_type: Optional[CallTypes],
    ) -> Optional[LLMResponseTypes]:
        """
        Allow modifying / reviewing the response just after it's received from the deployment.
        """
        from litellm.proxy._types import UserAPIKeyAuth

        # should run guardrail
        litellm_guardrails = request_data.get("guardrails")
        if litellm_guardrails is None or not isinstance(litellm_guardrails, list):
            return response

        if (
            self.should_run_guardrail(
                data=request_data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        # CHECK IF GUARDRAIL REJECTS THE REQUEST
        result = await self.async_post_call_success_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id=request_data.get("user_api_key_user_id"),
                team_id=request_data.get("user_api_key_team_id"),
                end_user_id=request_data.get("user_api_key_end_user_id"),
                api_key=request_data.get("user_api_key_hash"),
                request_route=request_data.get("user_api_key_request_route"),
            ),
            data=request_data,
            response=response,
        )

        if result is None or not isinstance(result, get_args(LLMResponseTypes)):
            return response

        return result

    def should_run_guardrail(
        self,
        data,
        event_type: GuardrailEventHooks,
    ) -> bool:
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
                if isinstance(self.event_hook, Mode):
                    try:
                        from litellm_enterprise.integrations.custom_guardrail import (
                            EnterpriseCustomGuardrailHelper,
                        )
                    except ImportError:
                        raise ImportError(
                            "Setting tag-based guardrails is only available in litellm-enterprise. You must be a premium user to use this feature."
                        )
                    result = EnterpriseCustomGuardrailHelper._should_run_if_mode_by_tag(
                        data, self.event_hook
                    )
                    if result is not None:
                        return result
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

        if isinstance(self.event_hook, Mode):
            try:
                from litellm_enterprise.integrations.custom_guardrail import (
                    EnterpriseCustomGuardrailHelper,
                )
            except ImportError:
                raise ImportError(
                    "Setting tag-based guardrails is only available in litellm-enterprise. You must be a premium user to use this feature."
                )
            result = EnterpriseCustomGuardrailHelper._should_run_if_mode_by_tag(
                data, self.event_hook
            )
            if result is not None:
                return result        
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
        if isinstance(self.event_hook, Mode):
            return event_type.value in self.event_hook.tags.values()
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
        guardrail_status: GuardrailStatus,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        masked_entity_count: Optional[Dict[str, int]] = None,
        guardrail_provider: Optional[str] = None,
    ) -> None:
        """
        Builds `StandardLoggingGuardrailInformation` and adds it to the request metadata so it can be used for logging to DataDog, Langfuse, etc.
        """
        if isinstance(guardrail_json_response, Exception):
            guardrail_json_response = str(guardrail_json_response)
        from litellm.types.utils import GuardrailMode

        slg = StandardLoggingGuardrailInformation(
            guardrail_name=self.guardrail_name,
            guardrail_provider=guardrail_provider,
            guardrail_mode=(
                GuardrailMode(**self.event_hook.model_dump())  # type: ignore
                if isinstance(self.event_hook, Mode)
                else self.event_hook
            ),
            guardrail_response=guardrail_json_response,
            guardrail_status=guardrail_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            masked_entity_count=masked_entity_count,
        )

        def _append_guardrail_info(container: dict) -> None:
            key = "standard_logging_guardrail_information"
            existing = container.get(key)
            if existing is None:
                container[key] = [slg]
            elif isinstance(existing, list):
                existing.append(slg)
            else:
                # should not happen
                container[key] = [existing, slg]

        if "metadata" in request_data:
            if request_data["metadata"] is None:
                request_data["metadata"] = {}
            _append_guardrail_info(request_data["metadata"])
        elif "litellm_metadata" in request_data:
            _append_guardrail_info(request_data["litellm_metadata"])
        else:
            verbose_logger.warning(
                "unable to log guardrail information. No metadata found in request_data"
            )

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[PiiEntityType]] = None,
        request_data: Optional[dict] = None,
    ) -> str:
        """
        Apply your guardrail logic to the given text

        Args:
            text: The text to apply the guardrail to
            language: The language of the text
            entities: The entities to mask, optional
            request_data: The request data dictionary to store guardrail metadata

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
            guardrail_status="guardrail_failed_to_respond",
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
        for key, value in vars(litellm_params).items():
            setattr(self, key, value)

    def get_guardrails_messages_for_call_type(
        self, call_type: CallTypes, data: Optional[dict] = None
    ) -> Optional[List[AllMessageValues]]:
        """
        Returns the messages for the given call type and data
        """
        if call_type is None or data is None:
            return None

        #########################################################
        # /chat/completions
        # /messages
        # Both endpoints store the messages in the "messages" key
        #########################################################
        if (
            call_type == CallTypes.completion.value
            or call_type == CallTypes.acompletion.value
            or call_type == CallTypes.anthropic_messages.value
        ):
            return data.get("messages")

        #########################################################
        # /responses
        # User/System messages are stored in the "input" key, use litellm transformation to get the messages
        #########################################################
        if (
            call_type == CallTypes.responses.value
            or call_type == CallTypes.aresponses.value
        ):
            from typing import cast

            from litellm.responses.litellm_completion_transformation.transformation import (
                LiteLLMCompletionResponsesConfig,
            )

            input_data = data.get("input")
            if input_data is None:
                return None

            messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input_data,
                responses_api_request=data,
            )
            return cast(List[AllMessageValues], messages)
        return None


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

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = datetime.now()  # Move start_time inside the wrapper
        self: CustomGuardrail = args[0]
        request_data: dict = kwargs.get("data") or kwargs.get("request_data") or {}
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
        start_time = datetime.now()  # Move start_time inside the wrapper
        self: CustomGuardrail = args[0]
        request_data: dict = kwargs.get("data") or kwargs.get("request_data") or {}
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
