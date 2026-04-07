import base64
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponseText,
)
from litellm.types.responses.main import DecodedResponseId
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    SpecialEnums,
    Usage,
)


class ResponsesAPIRequestUtils:
    """Helper utils for constructing ResponseAPI requests"""

    @staticmethod
    def _check_valid_arg(
        supported_params: Optional[List[str]],
        non_default_params: Dict,
        drop_params: Optional[bool],
        custom_llm_provider: Optional[str],
        model: str,
    ):
        if supported_params is None:
            return
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                unsupported_params[k] = non_default_params[k]
        if unsupported_params:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                pass
            else:
                raise litellm.UnsupportedParamsError(
                    status_code=500,
                    message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
                )

    @staticmethod
    def get_optional_params_responses_api(
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        allowed_openai_params: Optional[List[str]] = None,
    ) -> Dict:
        """
        Get optional parameters for the responses API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            responses_api_provider_config: The provider configuration for responses API

        Returns:
            A dictionary of supported parameters for the responses API
        """
        from litellm.utils import _apply_openai_param_overrides

        # Remove None values and internal parameters
        # Get supported parameters for the model
        supported_params = responses_api_provider_config.get_supported_openai_params(
            model
        )

        non_default_params = cast(Dict, response_api_optional_params)
        # Check for unsupported parameters
        ResponsesAPIRequestUtils._check_valid_arg(
            supported_params=supported_params + (allowed_openai_params or []),
            non_default_params=non_default_params,
            drop_params=litellm.drop_params,
            custom_llm_provider=responses_api_provider_config.custom_llm_provider,
            model=model,
        )

        # Map parameters to provider-specific format
        mapped_params = responses_api_provider_config.map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        # add any allowed_openai_params to the mapped_params
        mapped_params = _apply_openai_param_overrides(
            optional_params=mapped_params,
            non_default_params=non_default_params,
            allowed_openai_params=allowed_openai_params or [],
        )

        return mapped_params

    @staticmethod
    def get_requested_response_api_optional_param(
        params: Dict[str, Any],
    ) -> ResponsesAPIOptionalRequestParams:
        """
        Filter parameters to only include those defined in ResponsesAPIOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ResponsesAPIOptionalRequestParams instance with only the valid parameters
        """
        from litellm.utils import PreProcessNonDefaultParams

        valid_keys = get_type_hints(ResponsesAPIOptionalRequestParams).keys()
        custom_llm_provider = params.pop("custom_llm_provider", None)
        special_params = params.pop("kwargs", {})

        additional_drop_params = params.pop("additional_drop_params", None)
        non_default_params = (
            PreProcessNonDefaultParams.base_pre_process_non_default_params(
                passed_params=params,
                special_params=special_params,
                custom_llm_provider=custom_llm_provider,
                additional_drop_params=additional_drop_params,
                default_param_values={k: None for k in valid_keys},
                additional_endpoint_specific_params=["input"],
            )
        )

        # decode previous_response_id if it's a litellm encoded id
        if "previous_response_id" in non_default_params:
            decoded_previous_response_id = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
                non_default_params["previous_response_id"]
            )
            non_default_params["previous_response_id"] = decoded_previous_response_id

        if "metadata" in non_default_params:
            from litellm.utils import add_openai_metadata

            converted_metadata = add_openai_metadata(non_default_params["metadata"])
            if converted_metadata is not None:
                non_default_params["metadata"] = converted_metadata
            else:
                non_default_params.pop("metadata", None)

        return cast(ResponsesAPIOptionalRequestParams, non_default_params)

    # fmt: off
    @overload
    @staticmethod
    def _update_responses_api_response_id_with_model_id(
        responses_api_response: ResponsesAPIResponse,
        custom_llm_provider: Optional[str],
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> ResponsesAPIResponse: 
        ...

    @overload
    @staticmethod
    def _update_responses_api_response_id_with_model_id(
        responses_api_response: Dict[str, Any],
        custom_llm_provider: Optional[str],
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: 
        ...

    # fmt: on

    @staticmethod
    def _update_responses_api_response_id_with_model_id(
        responses_api_response: Union[ResponsesAPIResponse, Dict[str, Any]],
        custom_llm_provider: Optional[str],
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[ResponsesAPIResponse, Dict[str, Any]]:
        """Update the responses_api_response_id with model_id and custom_llm_provider.

        Handles both ``ResponsesAPIResponse`` objects and plain dictionaries returned
        by some streaming providers.
        """
        litellm_metadata = litellm_metadata or {}
        model_info: Dict[str, Any] = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id")

        # access the response id based on the object type
        if isinstance(responses_api_response, dict):
            response_id = responses_api_response.get("id")
        else:
            response_id = getattr(responses_api_response, "id", None)

        # If no response_id, return the response as-is (likely an error response)
        if response_id is None:
            return responses_api_response

        updated_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            model_id=model_id,
            custom_llm_provider=custom_llm_provider,
            response_id=response_id,
        )

        if isinstance(responses_api_response, dict):
            responses_api_response["id"] = updated_id
        else:
            responses_api_response.id = updated_id

        if litellm_metadata.get("encrypted_content_affinity_enabled"):
            responses_api_response = (
                ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
                    response=responses_api_response,
                    model_id=model_id,
                )
            )

        # Encode container IDs in the response output
        responses_api_response = (
            ResponsesAPIRequestUtils._update_container_ids_in_response(
                responses_api_response=responses_api_response,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=litellm_metadata,
            )
        )

        return responses_api_response

    @staticmethod
    def _build_encrypted_item_id(model_id: str, item_id: str) -> str:
        """Encode model_id into an output item ID for encrypted-content items.

        Format: ``encitem_{base64("litellm:model_id:{model_id};item_id:{original_id}")}``
        """
        assembled = f"litellm:model_id:{model_id};item_id:{item_id}"
        encoded = base64.b64encode(assembled.encode("utf-8")).decode("utf-8")
        return f"encitem_{encoded}"

    @staticmethod
    def _decode_encrypted_item_id(encoded_id: str) -> Optional[Dict[str, str]]:
        """Decode a litellm-encoded encrypted-content item ID.

        Returns a dict with ``model_id`` and ``item_id`` keys, or ``None`` if
        the string is not a litellm-encoded item ID.
        """
        if not encoded_id.startswith("encitem_"):
            return None
        try:
            cleaned = encoded_id[len("encitem_") :]
            # Restore any padding that may have been stripped in transit
            missing = len(cleaned) % 4
            if missing:
                cleaned += "=" * (4 - missing)
            decoded = base64.b64decode(cleaned.encode("utf-8")).decode("utf-8")
            # Split on first ";" only so that semicolons inside item_id are preserved
            parts = decoded.split(";", 1)
            if len(parts) < 2:
                return None
            model_id = parts[0].replace("litellm:model_id:", "")
            item_id = parts[1].replace("item_id:", "")
            return {"model_id": model_id, "item_id": item_id}
        except Exception:
            return None

    @staticmethod
    def _wrap_encrypted_content_with_model_id(
        encrypted_content: str, model_id: str
    ) -> str:
        """Wrap encrypted_content with model_id metadata for affinity routing.

        When Codex or other clients send items with encrypted_content but no ID,
        we encode the model_id directly into the encrypted_content itself.

        Format: ``litellm_enc:{base64("model_id:{model_id}")};{original_encrypted_content}``
        """
        metadata = f"model_id:{model_id}"
        encoded_metadata = base64.b64encode(metadata.encode("utf-8")).decode("utf-8")
        return f"litellm_enc:{encoded_metadata};{encrypted_content}"

    @staticmethod
    def _unwrap_encrypted_content_with_model_id(
        wrapped_content: str,
    ) -> tuple[Optional[str], str]:
        """Unwrap encrypted_content to extract model_id and original content.

        Returns:
            Tuple of (model_id, original_encrypted_content).
            If not wrapped, returns (None, original_content).
        """
        if not wrapped_content.startswith("litellm_enc:"):
            return None, wrapped_content

        try:
            # Split on first ";" to separate metadata from content
            parts = wrapped_content.split(";", 1)
            if len(parts) < 2:
                return None, wrapped_content

            metadata_b64 = parts[0].replace("litellm_enc:", "")
            original_content = parts[1]

            # Restore padding if needed
            missing = len(metadata_b64) % 4
            if missing:
                metadata_b64 += "=" * (4 - missing)

            decoded_metadata = base64.b64decode(metadata_b64.encode("utf-8")).decode(
                "utf-8"
            )
            model_id = decoded_metadata.replace("model_id:", "")
            return model_id, original_content
        except Exception:
            return None, wrapped_content

    @staticmethod
    def _update_encrypted_content_item_ids_in_response(
        response: Union["ResponsesAPIResponse", Dict[str, Any]],
        model_id: Optional[str],
    ) -> Union["ResponsesAPIResponse", Dict[str, Any]]:
        """Rewrite item IDs for output items that contain ``encrypted_content``.

        Encodes ``model_id`` into the item ID so that follow-up requests can be
        routed back to the originating deployment without any cache lookup.

        For items without an ID (e.g., from Codex), encodes model_id directly
        into the encrypted_content itself.
        """
        if not model_id:
            return response

        output: Optional[list] = None
        if isinstance(response, dict):
            output = response.get("output")
        else:
            output = getattr(response, "output", None)

        if not isinstance(output, list):
            return response

        for item in output:
            if isinstance(item, dict):
                item_id = item.get("id")
                encrypted_content = item.get("encrypted_content")

                if encrypted_content and isinstance(encrypted_content, str):
                    # Always wrap encrypted_content with model_id for redundancy
                    item[
                        "encrypted_content"
                    ] = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                        encrypted_content, model_id
                    )
                    # Also encode the ID if present
                    if item_id and isinstance(item_id, str):
                        item["id"] = ResponsesAPIRequestUtils._build_encrypted_item_id(
                            model_id, item_id
                        )
            else:
                item_id = getattr(item, "id", None)
                encrypted_content = getattr(item, "encrypted_content", None)

                if encrypted_content and isinstance(encrypted_content, str):
                    # Always wrap encrypted_content with model_id for redundancy
                    try:
                        item.encrypted_content = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                            encrypted_content, model_id
                        )
                    except AttributeError:
                        pass
                    # Also encode the ID if present
                    if item_id and isinstance(item_id, str):
                        try:
                            item.id = ResponsesAPIRequestUtils._build_encrypted_item_id(
                                model_id, item_id
                            )
                        except AttributeError:
                            pass

        return response

    @staticmethod
    def _restore_encrypted_content_item_ids_in_input(request_input: Any) -> Any:
        """Decode litellm-encoded item IDs in request input back to original IDs.

        Called before forwarding the request to the upstream provider so the
        provider receives the original item IDs and unwrapped encrypted_content.

        Handles both:
        1. Items with encoded IDs (encitem_...)
        2. Items with wrapped encrypted_content (litellm_enc:...)
        """
        if not isinstance(request_input, list):
            return request_input

        for item in request_input:
            if isinstance(item, dict):
                item_id = item.get("id")
                if item_id and isinstance(item_id, str):
                    decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(
                        item_id
                    )
                    if decoded:
                        item["id"] = decoded["item_id"]

                encrypted_content = item.get("encrypted_content")
                if encrypted_content and isinstance(encrypted_content, str):
                    (
                        _,
                        unwrapped,
                    ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(
                        encrypted_content
                    )
                    if unwrapped != encrypted_content:
                        item["encrypted_content"] = unwrapped

        return request_input

    @staticmethod
    def _build_responses_api_response_id(
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
        response_id: str,
    ) -> str:
        """Build the responses_api_response_id"""
        assembled_id: str = str(
            SpecialEnums.LITELLM_MANAGED_RESPONSE_COMPLETE_STR.value
        ).format(custom_llm_provider, model_id, response_id)
        base64_encoded_id: str = base64.b64encode(assembled_id.encode("utf-8")).decode(
            "utf-8"
        )
        return f"resp_{base64_encoded_id}"

    @staticmethod
    def _decode_responses_api_response_id(
        response_id: str,
    ) -> DecodedResponseId:
        """
        Decode the responses_api_response_id

        Returns:
            DecodedResponseId: Structured tuple with custom_llm_provider, model_id, and response_id
        """
        try:
            # Remove prefix and decode
            cleaned_id = response_id.replace("resp_", "")
            decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

            # Parse components using known prefixes
            if ";" not in decoded_id:
                return DecodedResponseId(
                    custom_llm_provider=None,
                    model_id=None,
                    response_id=response_id,
                )

            parts = decoded_id.split(";")

            # Format: litellm:custom_llm_provider:{};model_id:{};response_id:{}
            custom_llm_provider = None
            model_id = None

            if (
                len(parts) >= 3
            ):  # Full format with custom_llm_provider, model_id, and response_id
                custom_llm_provider_part = parts[0]
                model_id_part = parts[1]
                response_part = parts[2]

                custom_llm_provider = custom_llm_provider_part.replace(
                    "litellm:custom_llm_provider:", ""
                )
                model_id = model_id_part.replace("model_id:", "")
                decoded_response_id = response_part.replace("response_id:", "")
            else:
                decoded_response_id = response_id

            return DecodedResponseId(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                response_id=decoded_response_id,
            )
        except Exception as e:
            verbose_logger.debug(f"Error decoding response_id '{response_id}': {e}")
            return DecodedResponseId(
                custom_llm_provider=None,
                model_id=None,
                response_id=response_id,
            )

    @staticmethod
    def get_model_id_from_response_id(response_id: Optional[str]) -> Optional[str]:
        """Get the model_id from the response_id"""
        if response_id is None:
            return None
        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)
        )
        return decoded_response_id.get("model_id") or None

    @staticmethod
    def decode_previous_response_id_to_original_previous_response_id(
        previous_response_id: str,
    ) -> str:
        """
        Decode the previous_response_id to the original previous_response_id

        Why?
            - LiteLLM encodes the `custom_llm_provider` and `model_id` into the `previous_response_id` this helps with maintaining session consistency when load balancing multiple deployments of the same model.
            - We cannot send the litellm encoded b64 to the upstream llm api, hence we decode it to the original `previous_response_id`

        Args:
            previous_response_id: The previous_response_id to decode

        Returns:
            The original previous_response_id
        """
        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                previous_response_id
            )
        )
        return decoded_response_id.get("response_id", previous_response_id)

    @staticmethod
    def _build_container_id(
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
        container_id: str,
    ) -> str:
        """Build a managed container ID with provider and model info encoded.
        
        Format: cntr_{base64("litellm:custom_llm_provider:{provider};model_id:{model};container_id:{original}")}
        """
        # Avoid serializing Python None as the literal string "None" (breaks router affinity).
        provider_part = "" if custom_llm_provider is None else custom_llm_provider
        model_part = "" if model_id is None else model_id
        assembled_id = f"litellm:custom_llm_provider:{provider_part};model_id:{model_part};container_id:{container_id}"
        base64_encoded_id = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")
        return f"cntr_{base64_encoded_id}"

    @staticmethod
    def _decode_container_id(container_id: str) -> DecodedResponseId:
        """Decode a managed container ID to extract provider, model, and original container ID.
        
        Returns:
            DecodedResponseId with custom_llm_provider, model_id, and response_id (original container_id)
        """
        try:
            # If it doesn't start with cntr_, it's not a managed ID
            if not container_id.startswith("cntr_"):
                return DecodedResponseId(
                    custom_llm_provider=None,
                    model_id=None,
                    response_id=container_id,
                )
            
            # Remove prefix and decode
            cleaned_id = container_id.replace("cntr_", "")
            decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")
            
            # Parse components using regex to handle semicolons in the container_id
            if not decoded_id.startswith("litellm:"):
                return DecodedResponseId(
                    custom_llm_provider=None,
                    model_id=None,
                    response_id=container_id,
                )
            
            # Use regex to extract the three parts, allowing semicolons in container_id
            # Format: litellm:custom_llm_provider:{provider};model_id:{model};container_id:{container}
            # * for provider/model allows empty segments (missing router model_id).
            pattern = r"^litellm:custom_llm_provider:([^;]*);model_id:([^;]*);container_id:(.+)$"
            match = re.match(pattern, decoded_id)
            
            if not match:
                return DecodedResponseId(
                    custom_llm_provider=None,
                    model_id=None,
                    response_id=container_id,
                )
            
            raw_provider = match.group(1)
            raw_model_id = match.group(2)
            custom_llm_provider = (
                None if raw_provider in ("", "None") else raw_provider
            )
            model_id = None if raw_model_id in ("", "None") else raw_model_id
            original_container_id = match.group(3)
            
            return DecodedResponseId(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                response_id=original_container_id,
            )
        except Exception as e:
            verbose_logger.debug(f"Error decoding container_id '{container_id}': {e}")
            return DecodedResponseId(
                custom_llm_provider=None,
                model_id=None,
                response_id=container_id,
            )

    @staticmethod
    def decode_container_id_to_original(container_id: str) -> str:
        """Decode a managed container ID to get the original provider-issued ID.
        
        This is used when making upstream API calls - we need to send the original
        container ID that the provider issued, not our encoded version.
        """
        decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
        return decoded.get("response_id", container_id)

    @staticmethod
    def _encode_container_ids_in_annotations(
        annotations: Any,
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
    ) -> None:
        """Encode ``container_id`` on each annotation (e.g. ``container_file_citation``)."""
        if not annotations or not isinstance(annotations, list):
            return
        for ann in annotations:
            ResponsesAPIRequestUtils._encode_container_id_on_output_item(
                ann,
                custom_llm_provider,
                model_id,
            )

    @staticmethod
    def _encode_container_ids_in_message_content(
        content: Any,
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
    ) -> None:
        """Walk message ``content`` parts and encode citation ``container_id`` values."""
        if not content:
            return
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ResponsesAPIRequestUtils._encode_container_ids_in_annotations(
                        part.get("annotations"),
                        custom_llm_provider,
                        model_id,
                    )
                else:
                    ResponsesAPIRequestUtils._encode_container_ids_in_annotations(
                        getattr(part, "annotations", None),
                        custom_llm_provider,
                        model_id,
                    )

    @staticmethod
    def _encode_container_id_on_output_item(
        item: Any,
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
    ) -> None:
        """Mutate one output item (dict or object): wrap raw ``container_id`` as LiteLLM-managed.

        Handles top-level ``container_id`` and nested ``code_interpreter_call.container_id``
        (some wire payloads nest the tool call). Used by non-streaming responses and by
        streaming ``response.output_item.*`` events so UIs see managed IDs incrementally.

        For ``message`` items, also encodes ``container_id`` inside
        ``content[].annotations`` (``container_file_citation``), which is what clients use
        to fetch generated files.
        """
        if item is None:
            return

        def _maybe_encode(container_id: str) -> Optional[str]:
            decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
            if decoded.get("custom_llm_provider") is not None:
                return None
            return ResponsesAPIRequestUtils._build_container_id(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                container_id=container_id,
            )

        if isinstance(item, dict):
            cid = item.get("container_id")
            if isinstance(cid, str):
                enc = _maybe_encode(cid)
                if enc is not None:
                    item["container_id"] = enc
            nested = item.get("code_interpreter_call")
            if isinstance(nested, dict):
                nc = nested.get("container_id")
                if isinstance(nc, str):
                    enc = _maybe_encode(nc)
                    if enc is not None:
                        nested["container_id"] = enc
            if item.get("type") == "message":
                ResponsesAPIRequestUtils._encode_container_ids_in_message_content(
                    item.get("content"),
                    custom_llm_provider,
                    model_id,
                )
            return

        cid_attr = getattr(item, "container_id", None)
        if isinstance(cid_attr, str):
            enc = _maybe_encode(cid_attr)
            if enc is not None:
                try:
                    setattr(item, "container_id", enc)
                except Exception:
                    verbose_logger.debug(
                        "Could not set container_id on streaming output item",
                        exc_info=True,
                    )

        nested_obj = getattr(item, "code_interpreter_call", None)
        if nested_obj is not None:
            ResponsesAPIRequestUtils._encode_container_id_on_output_item(
                nested_obj,
                custom_llm_provider,
                model_id,
            )

        if getattr(item, "type", None) == "message":
            ResponsesAPIRequestUtils._encode_container_ids_in_message_content(
                getattr(item, "content", None),
                custom_llm_provider,
                model_id,
            )

    @staticmethod
    def _update_container_ids_in_response(
        responses_api_response: Union[ResponsesAPIResponse, Dict[str, Any]],
        custom_llm_provider: Optional[str],
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[ResponsesAPIResponse, Dict[str, Any]]:
        """Encode container IDs in the response output with provider/model info.
        
        This walks through all output items and encodes any container_id fields
        so that follow-up container API calls can auto-route to the correct provider.
        """
        litellm_metadata = litellm_metadata or {}
        model_info: Dict[str, Any] = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id")
        
        # Get the output list
        if isinstance(responses_api_response, dict):
            output = responses_api_response.get("output", [])
        else:
            output = getattr(responses_api_response, "output", [])
        
        if not output:
            return responses_api_response
        
        for item in output:
            ResponsesAPIRequestUtils._encode_container_id_on_output_item(
                item=item,
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
            )
        
        return responses_api_response

    @staticmethod
    def convert_text_format_to_text_param(
        text_format: Optional[Union[Type["BaseModel"], dict]],
        text: Optional["ResponseText"] = None,
    ) -> Optional["ResponseText"]:
        """
        Convert text_format parameter to text parameter for the responses API.

        Args:
            text_format: Pydantic model class or dict to convert to response format
            text: Existing text parameter (if provided, text_format is ignored)

        Returns:
            ResponseText object with the converted format, or None if conversion fails
        """
        if text_format is not None and text is None:
            from litellm.llms.base_llm.base_utils import type_to_response_format_param

            # Convert Pydantic model to response format
            response_format = type_to_response_format_param(text_format)
            if response_format is not None:
                # Create ResponseText object with the format
                # The responses API expects the format to have name at the top level
                text = {
                    "format": {
                        "type": response_format["type"],
                        "name": response_format["json_schema"]["name"],
                        "schema": response_format["json_schema"]["schema"],
                        "strict": response_format["json_schema"]["strict"],
                    }
                }
                return text
        return text

    @staticmethod
    def extract_mcp_headers_from_request(
        secret_fields: Optional[Dict[str, Any]],
        tools: Optional[Iterable[Any]],
    ) -> tuple[
        Optional[str],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
    ]:
        """
        Extract MCP auth headers from the request to pass to MCP server.
        Headers from tools.headers in request body should be passed to MCP server.
        """
        from starlette.datastructures import Headers

        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        # Extract headers from secret_fields which contains the original request headers
        raw_headers_from_request: Optional[Dict[str, str]] = None
        if secret_fields and isinstance(secret_fields, dict):
            raw_headers_from_request = secret_fields.get("raw_headers")

        # Extract MCP-specific headers using MCPRequestHandler methods
        mcp_auth_header: Optional[str] = None
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None
        oauth2_headers: Optional[Dict[str, str]] = None

        if raw_headers_from_request:
            headers_obj = Headers(raw_headers_from_request)
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(
                headers_obj
            )
            mcp_server_auth_headers = (
                MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers_obj)
            )
            oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(
                headers_obj
            )

        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "mcp":
                    tool_headers = tool.get("headers", {})
                    if tool_headers and isinstance(tool_headers, dict):
                        # Merge tool headers into mcp_server_auth_headers
                        # Extract server-specific headers from tool.headers
                        headers_obj_from_tool = Headers(tool_headers)
                        tool_mcp_server_auth_headers = (
                            MCPRequestHandler._get_mcp_server_auth_headers_from_headers(
                                headers_obj_from_tool
                            )
                        )
                        if tool_mcp_server_auth_headers:
                            if mcp_server_auth_headers is None:
                                mcp_server_auth_headers = {}
                            # Merge the headers from tool into existing headers
                            for (
                                server_alias,
                                headers_dict,
                            ) in tool_mcp_server_auth_headers.items():
                                if server_alias not in mcp_server_auth_headers:
                                    mcp_server_auth_headers[server_alias] = {}
                                mcp_server_auth_headers[server_alias].update(
                                    headers_dict
                                )
                        # Also merge raw headers (non-prefixed headers from tool.headers)
                        if raw_headers_from_request is None:
                            raw_headers_from_request = {}
                        raw_headers_from_request.update(tool_headers)

        return (
            mcp_auth_header,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers_from_request,
        )


class ResponseAPILoggingUtils:
    @staticmethod
    def _is_response_api_usage(usage: Union[dict, ResponseAPIUsage]) -> bool:
        """returns True if usage is from OpenAI Response API"""
        if isinstance(usage, ResponseAPIUsage):
            return True
        if "input_tokens" in usage and "output_tokens" in usage:
            return True
        return False

    @staticmethod
    def _transform_response_api_usage_to_chat_usage(
        usage_input: Optional[Union[dict, ResponseAPIUsage]],
    ) -> Usage:
        """
        Transforms ResponseAPIUsage or ImageUsage to a Usage object.

        Both have the same spec with input_tokens, output_tokens, and
        input_tokens_details (text_tokens, image_tokens).
        """
        if usage_input is None:
            return Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        response_api_usage: ResponseAPIUsage
        if isinstance(usage_input, dict):
            total_tokens = usage_input.get("total_tokens")
            if total_tokens is None:
                input_tokens = usage_input.get("input_tokens")
                output_tokens = usage_input.get("output_tokens")
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens
                    usage_input["total_tokens"] = total_tokens
            response_api_usage = ResponseAPIUsage(**usage_input)
        else:
            response_api_usage = usage_input
        prompt_tokens: int = response_api_usage.input_tokens or 0
        completion_tokens: int = response_api_usage.output_tokens or 0
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        if response_api_usage.input_tokens_details:
            if isinstance(response_api_usage.input_tokens_details, dict):
                prompt_tokens_details = PromptTokensDetailsWrapper(
                    **response_api_usage.input_tokens_details
                )
            else:
                prompt_tokens_details = PromptTokensDetailsWrapper(
                    cached_tokens=getattr(
                        response_api_usage.input_tokens_details, "cached_tokens", None
                    ),
                    audio_tokens=getattr(
                        response_api_usage.input_tokens_details, "audio_tokens", None
                    ),
                    text_tokens=getattr(
                        response_api_usage.input_tokens_details, "text_tokens", None
                    ),
                    image_tokens=getattr(
                        response_api_usage.input_tokens_details, "image_tokens", None
                    ),
                )
        completion_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
        output_tokens_details = getattr(
            response_api_usage, "output_tokens_details", None
        )
        if output_tokens_details:
            completion_tokens_details = CompletionTokensDetailsWrapper(
                reasoning_tokens=getattr(
                    output_tokens_details, "reasoning_tokens", None
                ),
                image_tokens=getattr(output_tokens_details, "image_tokens", None),
                text_tokens=getattr(output_tokens_details, "text_tokens", None),
            )

        chat_usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=prompt_tokens_details,
            completion_tokens_details=completion_tokens_details,
        )

        # Preserve cost attribute if it exists on ResponseAPIUsage
        if hasattr(response_api_usage, "cost") and response_api_usage.cost is not None:
            setattr(chat_usage, "cost", response_api_usage.cost)

        return chat_usage
