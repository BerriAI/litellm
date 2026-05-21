# +-------------------------------------------------------------+
#
#               Cisco AI Defense MCP-specific guardrail logic
#       https://developer.cisco.com/docs/ai-defense-inspection/
#
# +-------------------------------------------------------------+
"""MCP-specific inspection logic for the Cisco AI Defense guardrail.

This module is a focused split-out of the MCP code paths from
``cisco_ai_defense.py`` — the main module grew large enough during the
PR review rounds (chat surface, Responses API support, tool-call
extraction, streaming buffer, security hardening) that it warrants
splitting the orthogonal MCP code into its own file.

The class here is a private mixin
(``_CiscoAIDefenseMcpMixin``) consumed by ``CiscoAIDefenseGuardrail``
in ``cisco_ai_defense.py``. Consumers continue to import the public
guardrail class from
``litellm.proxy.guardrails.guardrail_hooks.cisco_ai_defense`` exactly
as before — this split is purely an internal organization change.

The mixin's instance methods rely on attributes / helpers supplied by
the main class (``self.api_base``, ``self.inspect_path``,
``self.inspection_type``, ``self.guardrail_name``,
``self._post_inspection``, ``self._handle_api_error``,
``self._finalize_inspection``, ``self.should_run_guardrail``). Python's
late binding resolves these at call time, so the mixin does not need
to import the main module — keeping the dependency direction one-way
and avoiding circular imports.
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


def _serialize_mcp_content_item(item: Any) -> Dict[str, Any]:
    """Serialize an MCP content item to a JSON-friendly dict.

    The MCP SDK exposes content items as Pydantic models (TextContent,
    ImageContent, EmbeddedResource); the proxy also occasionally hands
    us raw dicts and naïve objects with a ``.text`` attribute. This
    helper produces the wire-shape Cisco's ``/inspect/mcp`` endpoint
    expects, regardless of input shape.
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

    Mixed into the main class via ``class CiscoAIDefenseGuardrail(
    _CiscoAIDefenseMcpMixin, CustomGuardrail)``. Holds the MCP post-tool
    hook, MCP request/response payload builders, JSON-RPC normalization,
    sanitized-argument extraction, and the in-place redact helper.

    All chat-surface code (pre/moderation/post-call hooks, streaming
    iterator, Responses API flattening, tool-call argument extraction)
    stays in the main module.
    """

    # ------------------------------------------------------------------
    # Attributes / methods supplied by the concrete subclass.
    #
    # The mixin's methods call ``self.api_base``, ``self._post_inspection``,
    # ``self._finalize_inspection`` etc. — all defined on
    # ``CiscoAIDefenseGuardrail`` (the chat module) or inherited from
    # ``CustomGuardrail``. At runtime Python's late binding resolves
    # these without any explicit declaration, but mypy needs the type
    # stubs below to type-check the mixin file in isolation. The stubs
    # are inside ``TYPE_CHECKING`` so they have no runtime cost and do
    # not shadow the real attributes defined by the concrete class.
    if TYPE_CHECKING:
        # Attributes / constants supplied by the concrete subclass.
        api_base: str
        inspect_path: str
        inspection_type: str
        _PROVIDER_NAME: str
        # ``guardrail_name`` is inherited from ``CustomGuardrail`` on the
        # concrete subclass and declared there as ``Optional[str]``. Match
        # that shape so the mypy "incompatible base-class definition"
        # check passes on the concrete class's MRO.
        guardrail_name: Optional[str]

        # Methods supplied by ``CustomGuardrail`` (base class of the
        # concrete subclass).
        def should_run_guardrail(
            self, data: dict, event_type: GuardrailEventHooks
        ) -> bool: ...

        # Shared helpers defined on ``CiscoAIDefenseGuardrail`` itself.
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
            surface: str,
            start_time: datetime,
            direction: str = ...,
            response_obj: Any = ...,
        ) -> Dict[str, Any]: ...

    # ------------------------------------------------------------------
    # MCP post-tool hook (dispatcher contract)
    # ------------------------------------------------------------------

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> Optional[Any]:
        """Scan the MCP tool-call response after it returns from upstream.

        LiteLLM dispatches MCP responses through this CustomLogger hook
        (not through ``async_post_call_success_hook``), so this is where
        mcp-mode guardrails must intercept tool output. ``response_obj`` is
        a ``MCPPostCallResponseObject`` whose ``mcp_tool_call_response``
        attribute carries the actual content blocks from the tool.

        Important production-path subtleties:

        1. The litellm post-MCP dispatcher
           (``litellm_logging.async_post_mcp_tool_call_hook``) wraps every
           callback in ``try: ... except Exception``, logs as non-blocking,
           and continues. So a guardrail can NOT enforce ``block`` by
           raising — that's silently swallowed and the unsafe tool output
           reaches the caller anyway. The only blocking lever the
           dispatcher exposes is its "if hook returns non-None, swap
           ``mcp_tool_call_response`` with the returned object" path
           (litellm_logging.py: ``_parse_post_mcp_call_hook_response``).
           When the verdict is ``block`` we therefore build a synthetic
           ``MCPPostCallResponseObject`` whose payload is a single text
           block describing the violation, and return it so the dispatcher
           replaces the real tool output.

        2. The hook is registered on ``litellm.success_callback`` so that
           the dispatcher reaches us, but that registration is independent
           of the user's ``mode`` setting. We must gate on
           ``should_run_guardrail`` with ``during_mcp_call`` so a user who
           configured only ``mode: pre_mcp_call`` (request-only scanning)
           doesn't get every tool response scanned as a side effect.
        """
        del start_time, end_time

        if self.inspection_type != "mcp":
            return None

        request_data: Dict[str, Any] = {}
        # Carry forward identifiers from kwargs so logging / correlation
        # work even though MCP tool calls bypass standard request_data.
        for key in (
            "litellm_call_id",
            "id",
            "user",
            "mcp_tool_name",
            "tool_name",
            "mcp_arguments",
            "arguments",
            "mcp_server_name",
            "server_name",
            # Carry metadata so should_run_guardrail can read per-request
            # guardrail opt-in / opt-out signals.
            "metadata",
            "litellm_metadata",
            "guardrails",
        ):
            if key in kwargs and kwargs[key] is not None:
                request_data[key] = kwargs[key]

        # Per product decision: either MCP mode (``pre_mcp_call`` OR
        # ``during_mcp_call``) enables response scanning. Users picking
        # ``pre_mcp_call`` expect "guard the MCP call" which means both
        # request AND response. Check both event types — if neither is
        # enabled the scan is not requested.
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

        try:
            await self._inspect_mcp_response(
                request_data=request_data,
                response=mcp_tool_response,
            )
        except HTTPException as exc:
            # The dispatcher swallows exceptions. Translate the block into
            # a synthetic MCPPostCallResponseObject — when this hook
            # returns non-None, the dispatcher swaps
            # ``mcp_tool_call_response`` with our payload, so the caller
            # gets the violation message instead of the real tool output.
            blocking_response = self._build_blocking_mcp_response(
                detail=exc.detail, original_response_obj=response_obj
            )
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
        detail: Any,
        original_response_obj: Any,
    ) -> Any:
        """Build a synthetic MCPPostCallResponseObject that replaces blocked output.

        The dispatcher swallows raised exceptions, so returning a
        non-None ``MCPPostCallResponseObject`` is the only blocking
        lever. The text content carries the SAME canonical block
        payload (JSON-encoded) used by the HTTPException and SSE paths,
        so downstream consumers see a uniform shape across surfaces.
        """
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

        # Preserve response_cost if the original captured one.
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
    def _extract_mcp_tool_call_response(response_obj: Any) -> Any:
        """Pull the raw tool-call response off a MCPPostCallResponseObject."""
        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is None and isinstance(response_obj, dict):
            inner = response_obj.get("mcp_tool_call_response")
        # The proxy gives us either a list of content blocks or an object
        # with a ``content`` attribute — both are valid inputs to
        # ``_normalize_mcp_response``.
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

        return self._finalize_inspection(
            inspect_response=inspect_response,
            request_data=data,
            surface="mcp",
            start_time=start_time,
            direction="input",
        )

    async def _inspect_mcp_response(
        self,
        request_data: dict,
        response: Any,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
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

        return self._finalize_inspection(
            inspect_response=inspect_response,
            request_data=request_data,
            surface="mcp",
            start_time=start_time,
            direction="output",
            response_obj=response,
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

        tool_name = data.get("mcp_tool_name") or data.get("tool_name")
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
        response: Any,
    ) -> Optional[Dict[str, Any]]:
        """Build the JSON-RPC response envelope sent to ``/inspect/mcp``.

        Returns the top-level JSON-RPC body — same contract as the request
        payload: no wrapper, no sibling metadata/config keys. The ``id`` is
        backfilled from the originating request when the SDK response does
        not carry one, so Cisco can correlate request/response pairs.
        """
        normalized = self._normalize_mcp_response(response)
        if normalized is None:
            return None
        if normalized.get("id") in (None, "litellm-mcp"):
            request_id = request_data.get("litellm_call_id") or request_data.get("id")
            if request_id:
                normalized["id"] = request_id
        return normalized

    @staticmethod
    def _normalize_mcp_response(response: Any) -> Optional[Dict[str, Any]]:
        """Normalize an MCP tool response into a JSON-RPC envelope.

        Handles four production input shapes:

        1. A dict already shaped as a JSON-RPC envelope
           (``{"jsonrpc": "2.0", ...}``).
        2. A dict with a nested ``result`` object.
        3. A raw list of content items — the actual production wire shape
           because ``MCPPostCallResponseObject.mcp_tool_call_response`` is
           typed ``List[Union[...]]`` and the dispatcher hands us the
           unpacked list.
        4. A Pydantic-coerced ``List[(field_name, value)]`` of tuples
           produced when a ``CallToolResult`` BaseModel is iterated under
           a ``List[...]``-typed field. The real ``content`` list is
           buried inside one of the tuples.

        Returns the JSON-RPC envelope dict, or ``None`` if no scannable
        payload can be extracted.
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
        if isinstance(response, list):
            # Detect Pydantic's iterated-BaseModel shape and reach back into
            # the real ``content`` list. Without this branch, MCP response
            # inspection silently no-ops on the production path because
            # tuples can't be serialized as content items.
            if response and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in response
            ):
                inner_content = dict(response).get("content")
                if isinstance(inner_content, list):
                    response = inner_content
                else:
                    # Iterated BaseModel without a usable ``content`` field —
                    # we have nothing meaningful to inspect.
                    return None
            return {
                "jsonrpc": "2.0",
                "id": "litellm-mcp",
                "result": {
                    "content": [_serialize_mcp_content_item(item) for item in response]
                },
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
                "result": {
                    "content": [_serialize_mcp_content_item(item) for item in content]
                },
            }
        return None

    # ------------------------------------------------------------------
    # MCP redact (in-place rewrite of tool output)
    # ------------------------------------------------------------------

    @staticmethod
    def _set_mcp_tool_response_text(response_obj: Any, text: str) -> bool:
        """Replace the text content of an MCP tool response in-place.

        ``response_obj`` arrives in any of four production shapes — must
        handle all of them or redact silently falls through to block:

        1. ``MCPPostCallResponseObject`` wrapper with a
           ``.mcp_tool_call_response`` attribute holding the content list.
        2. A test-shim with a ``.content`` attribute.
        3. A raw ``List[content_item]`` — what
           ``_extract_mcp_tool_call_response`` returns in production
           because ``MCPPostCallResponseObject.mcp_tool_call_response``
           is typed as ``List[Union[...]]``.
        4. A Pydantic-coerced ``List[(field_name, value)]`` of tuples
           produced when the dispatcher hands us a ``CallToolResult``
           BaseModel — the real ``content`` list is buried inside the
           ``("content", [...])`` tuple.

        Returns True iff at least one text item was rewritten; False
        means there was nothing to redact and the caller should fall
        back to its on_flagged_action policy.
        """
        content_list = _CiscoAIDefenseMcpMixin._coerce_to_content_list(response_obj)
        if not isinstance(content_list, list) or not content_list:
            return False

        replaced = False
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = text
                replaced = True
            elif hasattr(item, "type") and getattr(item, "type", None) == "text":
                # mcp.types.TextContent is a Pydantic BaseModel and is
                # mutable by default. ``setattr`` is safe; guard for the
                # rare frozen-model case so a failed rewrite doesn't
                # take the whole redact path down.
                try:
                    setattr(item, "text", text)
                    replaced = True
                except (AttributeError, TypeError, ValueError):
                    continue
        return replaced

    @staticmethod
    def _coerce_to_content_list(response_obj: Any) -> Optional[List[Any]]:
        """Find the MCP content list inside any of the production response shapes.

        See ``_set_mcp_tool_response_text`` for the shapes we handle.
        Returns ``None`` if no content list can be located.
        """
        if response_obj is None:
            return None
        # Shape 1: wrapper with ``.mcp_tool_call_response``.
        inner = getattr(response_obj, "mcp_tool_call_response", None)
        if inner is not None:
            return _CiscoAIDefenseMcpMixin._coerce_to_content_list(inner)
        # Shape 2: object with ``.content`` (CallToolResult / SimpleNamespace).
        content = getattr(response_obj, "content", None)
        if isinstance(content, list):
            return content
        # Shapes 3 & 4: a list — either raw content items, or Pydantic's
        # coerced ``(field_name, value)`` tuples from iterating a
        # BaseModel under a ``List[...]`` field.
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
