"""MCP-specific inspection logic for the Cisco AI Defense guardrail.

The public guardrail class imports this private mixin from
``cisco_ai_defense.py``. Keeping MCP logic here avoids circular imports
while preserving the existing public import path.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.types.mcp import MCPPostCallResponseObject

    from .cisco_ai_defense import _ScanContext


def _serialize_mcp_content_item(item: object) -> Dict[str, Any]:
    """Serialize an MCP content item to a JSON-friendly dict.

    Handles raw dicts, MCP SDK Pydantic models, and simple ``.text`` objects.
    """
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump(exclude_none=True))
        except TypeError:
            return dict(model_dump())
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return {"type": getattr(item, "type", "text"), "text": text}
    return {"type": "text", "text": str(item)}


class _CiscoAIDefenseMcpMixin:
    """MCP-specific instance methods for ``CiscoAIDefenseGuardrail``.

    Holds the MCP hooks, JSON-RPC payload builders, and redaction helpers.
    """

    if TYPE_CHECKING:
        api_base: str
        inspect_path: str
        inspection_type: str
        _PROVIDER_NAME: str
        guardrail_name: Optional[str]

        def should_run_guardrail(
            self, data: dict, event_type: GuardrailEventHooks
        ) -> bool: ...

        async def _post_inspection(
            self, url: str, payload: Dict[str, Any], surface: str
        ) -> Dict[str, Any]: ...

        def _handle_api_error(
            self,
            error: Exception,
            *,
            request_data: Optional[dict] = ...,
            start_time: Optional[datetime] = ...,
            surface: str = ...,
            direction: str = ...,
        ) -> Dict[str, Any]: ...

        def _finalize_inspection(
            self,
            inspect_response: Dict[str, Any],
            request_data: dict,
            context: "_ScanContext",
            start_time: datetime,
            response_obj: object = ...,
        ) -> Dict[str, Any]: ...

    # ------------------------------------------------------------------
    # MCP post-tool hook (dispatcher contract)
    # ------------------------------------------------------------------

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict,
        response_obj: "MCPPostCallResponseObject",
        start_time: datetime,
        end_time: datetime,
    ) -> Optional["MCPPostCallResponseObject"]:
        """Scan MCP tool output and return a replacement object on block."""
        del start_time, end_time

        if self.inspection_type != "mcp":
            return None

        request_data: Dict[str, Any] = {}
        for key in (
            "name",
            "litellm_call_id",
            "id",
            "user",
            "mcp_tool_name",
            "tool_name",
            "mcp_arguments",
            "arguments",
            "mcp_server_name",
            "server_name",
            "metadata",
            "litellm_metadata",
            "mcp_tool_call_metadata",
            "guardrails",
        ):
            if key in kwargs and kwargs[key] is not None:
                request_data[key] = kwargs[key]
        self._hydrate_mcp_tool_context(request_data)

        if not (
            self.should_run_guardrail(
                data=request_data,
                event_type=GuardrailEventHooks.during_mcp_call,
            )
            or self.should_run_guardrail(
                data=request_data,
                event_type=GuardrailEventHooks.pre_mcp_call,
            )
        ):
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail (%s): no MCP mode configured "
                "— skipping MCP response scan.",
                self.guardrail_name,
            )
            return None

        mcp_tool_response = self._extract_mcp_tool_call_response(response_obj)
        if mcp_tool_response is None:
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: no MCP tool response payload "
                "to scan, skipping"
            )
            return None

        original_response = kwargs.get("original_response")
        try:
            await self._inspect_mcp_response(
                request_data=request_data,
                response=mcp_tool_response,
                redact_response_obj=(
                    original_response
                    if original_response is not None
                    else mcp_tool_response
                ),
            )
        except HTTPException as exc:
            blocking_response = self._build_blocking_mcp_response(
                detail=exc.detail, original_response_obj=response_obj
            )
            self._replace_mcp_tool_response(response_obj, blocking_response)
            if original_response is not None:
                self._replace_mcp_tool_response(original_response, blocking_response)
            add_guardrail_to_applied_guardrails_header(
                request_data=request_data, guardrail_name=self.guardrail_name
            )
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): MCP response blocked — "
                "tool output replaced with synthesized violation message.",
                self.guardrail_name,
            )
            return blocking_response

        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )
        return None

    def _build_blocking_mcp_response(
        self,
        detail: object,
        original_response_obj: object,
    ) -> "MCPPostCallResponseObject":
        """Build a synthetic MCPPostCallResponseObject for blocked output."""
        import json as _json

        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPostCallResponseObject
        from mcp.types import TextContent

        if isinstance(detail, dict):
            payload = detail
        else:
            payload = {
                "error": "Blocked by Cisco AI Defense Guardrail",
                "message": (
                    str(detail) if detail else "Blocked by Cisco AI Defense Guardrail"
                ),
                "provider": self._PROVIDER_NAME,
                "guardrail": self.guardrail_name,
                "surface": "mcp",
                "direction": "output",
                "action": "block",
            }

        original_hidden = getattr(original_response_obj, "hidden_params", None)
        if isinstance(original_hidden, HiddenParams):
            hidden_params: Any = original_hidden
        else:
            response_cost = getattr(original_hidden, "response_cost", None)
            hidden_params = (
                HiddenParams(response_cost=response_cost)
                if response_cost is not None
                else HiddenParams()
            )

        return MCPPostCallResponseObject(
            mcp_tool_call_response=[
                TextContent(type="text", text=_json.dumps(payload))
            ],
            hidden_params=hidden_params,
        )

    @staticmethod
    def _replace_mcp_tool_response(
        response_obj: object, replacement_obj: object
    ) -> bool:
        replacement = getattr(replacement_obj, "mcp_tool_call_response", None)
        if replacement is None:
            return False

        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is not None:
            if _CiscoAIDefenseMcpMixin._replace_mcp_tool_response(
                inner, replacement_obj
            ):
                return True
            try:
                setattr(response_obj, "mcp_tool_call_response", replacement)
                return True
            except (AttributeError, TypeError, ValueError):
                return False

        content = getattr(response_obj, "content", None)
        if isinstance(content, list):
            content[:] = replacement
            structured_replacement = (
                _CiscoAIDefenseMcpMixin._replacement_structured_content(replacement)
            )
            if hasattr(response_obj, "structuredContent"):
                try:
                    setattr(response_obj, "structuredContent", structured_replacement)
                except (AttributeError, TypeError, ValueError):
                    pass
            if hasattr(response_obj, "isError"):
                try:
                    setattr(response_obj, "isError", True)
                except (AttributeError, TypeError, ValueError):
                    pass
            return True

        if isinstance(response_obj, list):
            response_obj[:] = replacement
            return True

        if isinstance(response_obj, dict):
            result = response_obj.get("result")
            if isinstance(result, dict):
                result["content"] = replacement
                result["structuredContent"] = (
                    _CiscoAIDefenseMcpMixin._replacement_structured_content(replacement)
                )
                result["isError"] = True
                return True
            response_obj["result"] = {
                "content": replacement,
                "structuredContent": _CiscoAIDefenseMcpMixin._replacement_structured_content(
                    replacement
                ),
                "isError": True,
            }
            return True

        return False

    @staticmethod
    def _replacement_structured_content(
        replacement: object,
    ) -> Optional[Dict[str, str]]:
        if not isinstance(replacement, list) or not replacement:
            return None
        first = replacement[0]
        text = (
            first.get("text")
            if isinstance(first, dict)
            else getattr(first, "text", None)
        )
        return {"result": text} if isinstance(text, str) else None

    @staticmethod
    def _extract_mcp_tool_call_response(response_obj: object) -> object:
        """Pull the raw tool-call response off a MCPPostCallResponseObject."""
        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is None and isinstance(response_obj, dict):
            inner = response_obj.get("mcp_tool_call_response")
        return inner if inner is not None else response_obj

    # ------------------------------------------------------------------
    # MCP request / response inspection
    # ------------------------------------------------------------------

    async def _inspect_mcp_request(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Dict[str, Any]:
        del user_api_key_dict  # carried via logging metadata, not the wire payload
        url = f"{self.api_base}{self.inspect_path}"
        payload = self._build_mcp_request_payload(data=data)
        if payload is None:
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: could not build MCP request "
                "payload, skipping"
            )
            return {}
        start_time = datetime.now()
        try:
            inspect_response = await self._post_inspection(
                url=url, payload=payload, surface="mcp"
            )
        except HTTPException:
            raise
        except Exception as exc:
            return self._handle_api_error(
                exc,
                request_data=data,
                start_time=start_time,
                surface="mcp",
                direction="input",
            )

        from .cisco_ai_defense import _ScanContext

        return self._finalize_inspection(
            inspect_response=inspect_response,
            request_data=data,
            context=_ScanContext(surface="mcp", direction="input"),
            start_time=start_time,
        )

    async def _inspect_mcp_response(
        self,
        request_data: dict,
        response: object,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
        redact_response_obj: object = None,
    ) -> Dict[str, Any]:
        del user_api_key_dict  # carried via logging metadata, not the wire payload
        url = f"{self.api_base}{self.inspect_path}"
        payload = self._build_mcp_response_payload(
            request_data=request_data,
            response=response,
        )
        if payload is None:
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: could not build MCP response "
                "payload, skipping"
            )
            return {}
        start_time = datetime.now()
        try:
            inspect_response = await self._post_inspection(
                url=url, payload=payload, surface="mcp"
            )
        except HTTPException:
            raise
        except Exception as exc:
            return self._handle_api_error(
                exc,
                request_data=request_data,
                start_time=start_time,
                surface="mcp",
                direction="output",
            )

        from .cisco_ai_defense import _ScanContext

        return self._finalize_inspection(
            inspect_response=inspect_response,
            request_data=request_data,
            context=_ScanContext(surface="mcp", direction="output"),
            start_time=start_time,
            response_obj=(
                response if redact_response_obj is None else redact_response_obj
            ),
        )

    def _build_mcp_request_payload(
        self,
        data: dict,
    ) -> Optional[Dict[str, Any]]:
        """Build the JSON-RPC ``tools/call`` envelope sent to ``/inspect/mcp``.

        The Cisco AI Defense MCP inspect endpoint expects the JSON-RPC
        envelope itself as the request body — *not* wrapped under a
        ``request`` key with sibling ``metadata`` / ``config`` keys. Policies
        are applied based on the API key linked to the request. Operator
        metadata (user, call id, src/dst app, etc.) is carried out-of-band
        via the standard logging payload so the wire contract stays
        identical to a hand-rolled ``curl`` against ``/inspect/mcp``.
        """
        if data.get("jsonrpc") == "2.0":
            return {
                "jsonrpc": "2.0",
                "id": (data.get("id") or data.get("litellm_call_id") or "litellm-mcp"),
                "method": data.get("method") or "tools/call",
                "params": data.get("params") or {},
            }

        tool_name = (
            data.get("mcp_tool_name") or data.get("tool_name") or data.get("name")
        )
        if not tool_name:
            return None

        arguments = data.get("mcp_arguments")
        if arguments is None:
            arguments = data.get("arguments")

        return {
            "jsonrpc": "2.0",
            "id": data.get("litellm_call_id") or "litellm-mcp",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": (arguments if isinstance(arguments, dict) else {}),
            },
        }

    def _build_mcp_response_payload(
        self,
        request_data: dict,
        response: object,
    ) -> Optional[Dict[str, Any]]:
        """Build the MCP response-inspection body sent to ``/inspect/mcp``."""
        request_payload = self._build_mcp_request_payload(data=request_data)
        if request_payload is None:
            return None
        normalized = self._normalize_mcp_response(response)
        if normalized is None:
            return None

        payload = dict(request_payload)
        response_id = normalized.get("id")
        if response_id not in (None, "litellm-mcp"):
            payload["id"] = response_id
        elif payload.get("id") in (None, "litellm-mcp"):
            request_id = request_data.get("litellm_call_id") or request_data.get("id")
            if request_id:
                payload["id"] = request_id

        if "result" in normalized:
            payload["result"] = normalized["result"]
        if "error" in normalized:
            payload["error"] = normalized["error"]
        return payload

    @staticmethod
    def _hydrate_mcp_tool_context(request_data: Dict[str, Any]) -> None:
        metadata = request_data.get("mcp_tool_call_metadata")
        if metadata is None:
            nested = request_data.get("metadata") or request_data.get(
                "litellm_metadata"
            )
            if isinstance(nested, dict):
                metadata = nested.get("mcp_tool_call_metadata")
        if not isinstance(metadata, dict):
            return

        name = metadata.get("name")
        arguments = metadata.get("arguments")
        server_name = metadata.get("mcp_server_name")

        if name:
            request_data.setdefault("mcp_tool_name", name)
            request_data.setdefault("tool_name", name)
            request_data.setdefault("name", name)
        if arguments is not None:
            request_data.setdefault("mcp_arguments", arguments)
            request_data.setdefault("arguments", arguments)
        if server_name:
            request_data.setdefault("mcp_server_name", server_name)
            request_data.setdefault("server_name", server_name)

    @staticmethod
    def _normalize_mcp_response(response: object) -> Optional[Dict[str, Any]]:
        """Normalize an MCP tool response into a JSON-RPC envelope.

        Handles JSON-RPC dicts, raw content lists, MCP SDK models, and
        Pydantic-coerced ``[(field_name, value)]`` lists.
        """
        if isinstance(response, dict):
            if response.get("jsonrpc") == "2.0":
                return dict(response)
            if isinstance(response.get("result"), dict):
                return {
                    "jsonrpc": "2.0",
                    "id": response.get("id") or "litellm-mcp",
                    "result": response["result"],
                }
            content = response.get("content")
            if isinstance(content, list):
                return {
                    "jsonrpc": "2.0",
                    "id": response.get("id") or "litellm-mcp",
                    "result": _CiscoAIDefenseMcpMixin._build_mcp_result(
                        content=content, source=response
                    ),
                }
        if isinstance(response, list):
            if response and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in response
            ):
                response_fields = dict(response)
                inner_content = response_fields.get("content")
                if isinstance(inner_content, list):
                    return {
                        "jsonrpc": "2.0",
                        "id": "litellm-mcp",
                        "result": _CiscoAIDefenseMcpMixin._build_mcp_result(
                            content=inner_content, source=response_fields
                        ),
                    }
                else:
                    return None
            return {
                "jsonrpc": "2.0",
                "id": "litellm-mcp",
                "result": _CiscoAIDefenseMcpMixin._build_mcp_result(content=response),
            }
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(exclude_none=True)
            except TypeError:
                dumped = model_dump()
            if isinstance(dumped, dict):
                return _CiscoAIDefenseMcpMixin._normalize_mcp_response(dumped)
        content = getattr(response, "content", None)
        if isinstance(content, list):
            return {
                "jsonrpc": "2.0",
                "id": "litellm-mcp",
                "result": _CiscoAIDefenseMcpMixin._build_mcp_result(
                    content=content, source=response
                ),
            }
        return None

    @staticmethod
    def _build_mcp_result(
        content: List[Any],
        source: object = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "content": [_serialize_mcp_content_item(item) for item in content]
        }
        for key in ("structuredContent", "isError"):
            value = (
                source.get(key)
                if isinstance(source, dict)
                else getattr(source, key, None)
            )
            if value is not None and (key != "isError" or isinstance(value, bool)):
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # MCP redact (in-place rewrite of tool output)
    # ------------------------------------------------------------------

    @staticmethod
    def _set_mcp_tool_response_text(response_obj: object, text: str) -> bool:
        """Replace text content in any supported MCP response shape."""
        if response_obj is None:
            return False

        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is not None:
            return _CiscoAIDefenseMcpMixin._set_mcp_tool_response_text(inner, text)

        content_list = _CiscoAIDefenseMcpMixin._coerce_to_content_list(response_obj)

        replaced = False
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = text
                    replaced = True
                elif hasattr(item, "type") and getattr(item, "type", None) == "text":
                    try:
                        setattr(item, "text", text)
                        replaced = True
                    except (AttributeError, TypeError, ValueError):
                        continue

        replacement = {"result": text}
        if (
            isinstance(response_obj, list)
            and response_obj
            and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in response_obj
            )
        ):
            for index, item in enumerate(response_obj):
                if item[0] == "structuredContent":
                    response_obj[index] = (item[0], replacement)
                    replaced = True
        elif hasattr(response_obj, "structuredContent"):
            try:
                setattr(response_obj, "structuredContent", replacement)
                replaced = True
            except (AttributeError, TypeError, ValueError):
                pass
        elif isinstance(response_obj, dict):
            result = response_obj.get("result")
            target: Dict[Any, Any] = (
                result if isinstance(result, dict) else response_obj
            )
            if "structuredContent" in target:
                target["structuredContent"] = replacement
                replaced = True

        return replaced

    @staticmethod
    def _coerce_to_content_list(response_obj: object) -> Optional[List[Any]]:
        """Find the MCP content list inside supported response shapes."""
        if response_obj is None:
            return None
        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is not None:
            return _CiscoAIDefenseMcpMixin._coerce_to_content_list(inner)
        content = getattr(response_obj, "content", None)
        if isinstance(content, list):
            return content
        if isinstance(response_obj, list):
            if response_obj and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in response_obj
            ):
                inner_content = dict(response_obj).get("content")
                if isinstance(inner_content, list):
                    return inner_content
                return None
            return response_obj
        return None

    # ------------------------------------------------------------------
    # MCP-specific verdict extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sanitized_mcp_arguments(
        inspect_response: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Pull sanitized MCP tool-call arguments off the verdict.

        Cisco can return them at the top level (``params.arguments``) or
        under ``sanitized_payload`` / ``modified_payload``.
        """
        containers = [inspect_response]
        for container_key in ("result", "data"):
            container = inspect_response.get(container_key)
            if isinstance(container, dict):
                containers.append(container)

        for container in containers:
            params = container.get("params")
            if isinstance(params, dict):
                args = params.get("arguments")
                if isinstance(args, dict) and args:
                    return dict(args)
            for key in (
                "sanitized_payload",
                "sanitizedPayload",
                "modified_payload",
                "modifiedPayload",
            ):
                payload = container.get(key)
                if isinstance(payload, dict):
                    inner_params = payload.get("params")
                    if isinstance(inner_params, dict):
                        args = inner_params.get("arguments")
                        if isinstance(args, dict) and args:
                            return dict(args)
                    direct = payload.get("arguments")
                    if isinstance(direct, dict) and direct:
                        return dict(direct)
        return None
