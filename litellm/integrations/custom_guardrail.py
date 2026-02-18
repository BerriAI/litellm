from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
)

from litellm._logging import verbose_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import (
    DynamicGuardrailParams,
    GuardrailEventHooks,
    LitellmParams,
    Mode,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import (
    CallTypes,
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
    LLMResponseTypes,
    StandardLoggingGuardrailInformation,
)

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None  # type: ignore

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
dc = DualCache()


class ModifyResponseException(Exception):
    """
    Exception raised when a guardrail wants to modify the response.

    This exception carries the synthetic response that should be returned
    to the user instead of calling the LLM or instead of the LLM's response.
    It should be caught by the proxy and returned with a 200 status code.

    This is a base exception that all guardrails can use to replace responses,
    allowing violation messages to be returned as successful responses
    rather than errors.
    """

    def __init__(
        self,
        message: str,
        model: str,
        request_data: Dict[str, Any],
        guardrail_name: Optional[str] = None,
        detection_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the modify response exception.

        Args:
            message: The violation message to return to the user
            model: The model that was being called
            request_data: The original request data
            guardrail_name: Name of the guardrail that raised this exception
            detection_info: Additional detection metadata (scores, rules, etc.)
        """
        self.message = message
        self.model = model
        self.request_data = request_data
        self.guardrail_name = guardrail_name
        self.detection_info = detection_info or {}
        super().__init__(message)


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
        violation_message_template: Optional[str] = None,
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
        self.violation_message_template: Optional[str] = violation_message_template

        if supported_event_hooks:
            ## validate event_hook is in supported_event_hooks
            self._validate_event_hook(event_hook, supported_event_hooks)
        super().__init__(**kwargs)

    def render_violation_message(
        self, default: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Return a custom violation message if template is configured."""

        if not self.violation_message_template:
            return default

        format_context: Dict[str, Any] = {"default_message": default}
        if context:
            format_context.update(context)
        try:
            return self.violation_message_template.format(**format_context)
        except Exception as e:
            verbose_logger.warning(
                "Failed to format violation message template for guardrail %s: %s",
                self.guardrail_name,
                e,
            )
            return default

    def raise_passthrough_exception(
        self,
        violation_message: str,
        request_data: Dict[str, Any],
        detection_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Raise a passthrough exception for guardrail violations.

        This helper method should be used by guardrails when they detect a violation
        in passthrough mode.

        The exception will be caught by the proxy endpoints and converted to a 200 response
        with the violation message, preventing the LLM call from being made (pre_call/during_call)
        or replacing the LLM response (post_call).

        Args:
            violation_message: The formatted violation message to return to the user
            request_data: The original request data dictionary
            detection_info: Optional dictionary with detection metadata (scores, rules, etc.)

        Raises:
            ModifyResponseException: Always raises this exception to short-circuit
                                     the LLM call and return the violation message

        Example:
            if violation_detected and self.on_flagged_action == "passthrough":
                message = self._format_violation_message(detection_info)
                self.raise_passthrough_exception(
                    violation_message=message,
                    request_data=data,
                    detection_info=detection_info
                )
        """
        model = request_data.get("model", "unknown")

        raise ModifyResponseException(
            message=violation_message,
            model=model,
            request_data=request_data,
            guardrail_name=self.guardrail_name,
            detection_info=detection_info,
        )

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

    def get_disable_global_guardrail(self, data: dict) -> Optional[bool]:
        """
        Returns True if the global guardrail should be disabled
        """
        if "disable_global_guardrail" in data:
            return data["disable_global_guardrail"]
        metadata = data.get("litellm_metadata") or data.get("metadata", {})
        if "disable_global_guardrail" in metadata:
            return metadata["disable_global_guardrail"]
        return False

    def _is_valid_response_type(self, result: Any) -> bool:
        """
        Check if result is a valid LLMResponseTypes instance.

        Safely handles TypedDict types which don't support isinstance checks.
        For non-LiteLLM responses (like passthrough httpx.Response), returns True
        to allow them through.
        """
        if result is None:
            return False

        try:
            # Try isinstance check on valid types that support it
            response_types = get_args(LLMResponseTypes)
            return isinstance(result, response_types)
        except TypeError as e:
            # TypedDict types don't support isinstance checks
            # In this case, we can't validate the type, so we allow it through
            if "TypedDict" in str(e):
                return True
            raise

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

        if not self._is_valid_response_type(result):
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
        disable_global_guardrail = self.get_disable_global_guardrail(data)
        verbose_logger.debug(
            "inside should_run_guardrail for guardrail=%s event_type= %s guardrail_supported_event_hooks= %s requested_guardrails= %s self.default_on= %s",
            self.guardrail_name,
            event_type,
            self.event_hook,
            requested_guardrails,
            self.default_on,
        )
        if self.default_on is True and disable_global_guardrail is not True:
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
                extra_body = guardrail_config.get("extra_body", {})
                if self._validate_premium_user() is not True:
                    if isinstance(extra_body, dict) and extra_body:
                        verbose_logger.warning(
                            "Guardrail %s: ignoring dynamic extra_body keys %s because premium_user is False",
                            self.guardrail_name,
                            list(extra_body.keys()),
                        )
                    return {}

                # Return the extra_body if it exists, otherwise empty dict
                return extra_body

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
        event_type: Optional[GuardrailEventHooks] = None,
        tracing_detail: Optional[GuardrailTracingDetail] = None,
    ) -> None:
        """
        Builds `StandardLoggingGuardrailInformation` and adds it to the request metadata so it can be used for logging to DataDog, Langfuse, etc.

        Args:
            tracing_detail: Optional typed dict with provider-specific tracing fields
                (guardrail_id, policy_template, detection_method, confidence_score,
                classification, match_details, patterns_checked, alert_recipients).
        """
        if isinstance(guardrail_json_response, Exception):
            guardrail_json_response = str(guardrail_json_response)
        from litellm.types.utils import GuardrailMode

        # Use event_type if provided, otherwise fall back to self.event_hook
        guardrail_mode: Union[
            GuardrailEventHooks, GuardrailMode, List[GuardrailEventHooks]
        ]
        if event_type is not None:
            guardrail_mode = event_type
        elif isinstance(self.event_hook, Mode):
            guardrail_mode = GuardrailMode(**dict(self.event_hook.model_dump()))  # type: ignore[typeddict-item]
        else:
            guardrail_mode = self.event_hook  # type: ignore[assignment]

        from litellm.litellm_core_utils.core_helpers import (
            filter_exceptions_from_params,
        )

        # Sanitize the response to ensure it's JSON serializable and free of circular refs
        # This prevents RecursionErrors in downstream loggers (Langfuse, Datadog, etc.)
        clean_guardrail_response = filter_exceptions_from_params(
            guardrail_json_response
        )

        slg = StandardLoggingGuardrailInformation(
            guardrail_name=self.guardrail_name,
            guardrail_provider=guardrail_provider,
            guardrail_mode=guardrail_mode,
            guardrail_response=clean_guardrail_response,
            guardrail_status=guardrail_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            masked_entity_count=masked_entity_count,
            **(tracing_detail or {}),
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
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply your guardrail logic to the given inputs

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to apply the guardrail to
                - images: Optional list of images to apply the guardrail to
                - tool_calls: Optional list of tool calls to apply the guardrail to
            request_data: The request data dictionary - containing user api key metadata (e.g. user_id, team_id, etc.)
            input_type: The type of input to apply the guardrail to - "request" or "response"
            logging_obj: Optional logging object for tracking the guardrail execution

        Any of the custom guardrails can override this method to provide custom guardrail logic

        Returns the texts with the guardrail applied and the images with the guardrail applied (if any)

        Raises:
            Exception:
                - If the guardrail raises an exception

        """
        return inputs

    def _process_response(
        self,
        response: Optional[Dict],
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
        original_inputs: Optional[Dict] = None,
    ):
        """
        Add StandardLoggingGuardrailInformation to the request data

        This gets logged on downsteam Langfuse, DataDog, etc.
        """
        # Convert None to empty dict to satisfy type requirements
        guardrail_response: Union[Dict[str, Any], str] = (
            {} if response is None else response
        )

        # For apply_guardrail functions in custom_code_guardrail scenario,
        # simplify the logged response to "allow", "deny", or "mask"
        if original_inputs is not None and isinstance(response, dict):
            # Check if inputs were modified by comparing them
            if self._inputs_were_modified(original_inputs, response):
                guardrail_response = "mask"
            else:
                guardrail_response = "allow"

        verbose_logger.debug(f"Guardrail response: {response}")

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status="success",
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
        )
        return response

    @staticmethod
    def _is_guardrail_intervention(e: Exception) -> bool:
        """
        Returns True if the exception represents an intentional guardrail block
        (this was logged previously as an API failure - guardrail_failed_to_respond).

        Guardrails signal intentional blocks by raising:
        - HTTPException with status 400 (content policy violation)
        - ModifyResponseException (passthrough mode violation)
        """

        if isinstance(e, ModifyResponseException):
            return True
        if (
            HTTPException is not None
            and isinstance(e, HTTPException)
            and e.status_code == 400
        ):
            return True
        return False

    def _process_error(
        self,
        e: Exception,
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
    ):
        """
        Add StandardLoggingGuardrailInformation to the request data

        This gets logged on downsteam Langfuse, DataDog, etc.
        """
        guardrail_status: GuardrailStatus = (
            "guardrail_intervened"
            if self._is_guardrail_intervention(e)
            else "guardrail_failed_to_respond"
        )
        # For custom_code_guardrail scenario, log as "deny" instead of full exception
        # Check if this is from custom_code_guardrail by checking the class name
        guardrail_response: Union[Exception, str] = e
        if "CustomCodeGuardrail" in self.__class__.__name__:
            guardrail_response = "deny"

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
        )
        raise e

    def _inputs_were_modified(self, original_inputs: Dict, response: Dict) -> bool:
        """
        Compare original inputs with response to determine if content was modified.

        Returns True if the inputs were modified (mask scenario), False otherwise (allow scenario).
        """
        # Get all keys from both dictionaries
        all_keys = set(original_inputs.keys()) | set(response.keys())

        # Compare each key's value
        for key in all_keys:
            original_value = original_inputs.get(key)
            response_value = response.get(key)
            if original_value != response_value:
                return True

        # No modifications detected
        return False

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
        - post_call
    """
    import functools
    import inspect

    def _infer_event_type_from_function_name(
        func_name: str,
    ) -> Optional[GuardrailEventHooks]:
        """Infer the actual event type from the function name"""
        if func_name == "async_pre_call_hook":
            return GuardrailEventHooks.pre_call
        elif func_name == "async_moderation_hook":
            return GuardrailEventHooks.during_call
        elif func_name in (
            "async_post_call_success_hook",
            "async_post_call_streaming_hook",
        ):
            return GuardrailEventHooks.post_call
        return None

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = datetime.now()  # Move start_time inside the wrapper
        self: CustomGuardrail = args[0]
        request_data: dict = kwargs.get("data") or kwargs.get("request_data") or {}
        event_type = _infer_event_type_from_function_name(func.__name__)

        # Store original inputs for comparison (for apply_guardrail functions)
        original_inputs = None
        if func.__name__ == "apply_guardrail" and "inputs" in kwargs:
            original_inputs = kwargs.get("inputs")

        try:
            response = await func(*args, **kwargs)
            return self._process_response(
                response=response,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
                original_inputs=original_inputs,
            )
        except Exception as e:
            return self._process_error(
                e=e,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
            )

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = datetime.now()  # Move start_time inside the wrapper
        self: CustomGuardrail = args[0]
        request_data: dict = kwargs.get("data") or kwargs.get("request_data") or {}
        event_type = _infer_event_type_from_function_name(func.__name__)

        # Store original inputs for comparison (for apply_guardrail functions)
        original_inputs = None
        if func.__name__ == "apply_guardrail" and "inputs" in kwargs:
            original_inputs = kwargs.get("inputs")

        try:
            response = func(*args, **kwargs)
            return self._process_response(
                response=response,
                request_data=request_data,
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
                original_inputs=original_inputs,
            )
        except Exception as e:
            return self._process_error(
                e=e,
                request_data=request_data,
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=event_type,
            )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return async_wrapper(*args, **kwargs)
        return sync_wrapper(*args, **kwargs)

    return wrapper
