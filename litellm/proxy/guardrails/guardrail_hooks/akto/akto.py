import asyncio
import json
import os
from collections import Counter
from collections.abc import Awaitable, Mapping
from datetime import datetime
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from fastapi import HTTPException
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from typing_extensions import NotRequired, ReadOnly, TypedDict, Unpack, override

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.factory import get_tool_calls_from_response
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.utils import JSONLeafPath, json_string_leaves
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.akto import AktoGuardrailConfigModelOptionalParams
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs

from .akto_attachments import request_attachments, without_attachment_content

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class _CustomGuardrailKwargs(TypedDict):
    guardrail_name: NotRequired[ReadOnly[str | None]]
    event_hook: NotRequired[ReadOnly[GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None]]
    default_on: NotRequired[ReadOnly[bool]]
    mask_request_content: NotRequired[ReadOnly[bool]]
    mask_response_content: NotRequired[ReadOnly[bool]]
    violation_message_template: NotRequired[ReadOnly[str | None]]
    end_session_after_n_fails: NotRequired[ReadOnly[int | None]]
    on_violation: NotRequired[ReadOnly[str | None]]
    realtime_violation_message: NotRequired[ReadOnly[str | None]]
    on_sensitive_data: NotRequired[ReadOnly[str | None]]
    sensitive_data_route_to_model: NotRequired[ReadOnly[str | None]]
    sticky_session_routing: NotRequired[ReadOnly[bool]]
    run_in_parallel: NotRequired[ReadOnly[bool]]
    scan_raw_request: NotRequired[ReadOnly[bool]]
    only_scan_new_messages: NotRequired[ReadOnly[bool]]
    supported_event_hooks: NotRequired[ReadOnly[list[GuardrailEventHooks]]]


HTTP_PROXY_PATH: Final = "/api/http-proxy"
AKTO_CONNECTOR_NAME: Final = "litellm"
DEFAULT_STREAMING_SAMPLING_RATE: Final = 5
DEFAULT_GUARDRAIL_TIMEOUT: Final = 5
DEFAULT_FILE_GUARDRAIL_TIMEOUT: Final = 10
DEFAULT_CONTEXT_SOURCE: Final = "AGENTIC"
DEFAULT_REQUEST_PATH: Final = "/v1/chat/completions"
MCP_PATH: Final = "/mcp"
MCP_TOOL_PREFIX: Final = "mcp"
DEFAULT_BLOCK_REASON: Final = "Blocked by Akto Guardrails"
RESPONSES_API_CALL_TYPES: Final = frozenset((CallTypes.responses.value, CallTypes.aresponses.value))
MESSAGES_API_CALL_TYPES: Final = frozenset((CallTypes.anthropic_messages.value, CallTypes.aanthropic_messages.value))
UNMASKABLE_REASON: Final = "Content masked by Akto guardrail policy could not be applied"
MALFORMED_ATTACHMENT_REASON: Final = "Attachment could not be read for the Akto guardrail check"
UNREACHABLE_REASON: Final = "Akto guardrail service unreachable"
BLOCKING_BEHAVIOURS: Final = frozenset(("block", ""))
SESSION_ID_HEADER: Final = "x-akto-installer-akto_session_id"
MESSAGE_ID_HEADER: Final = "x-akto-installer-akto_message_id"
EXCLUDED_HEADERS: Final = SpecialHeaders.litellm_credential_header_names() | frozenset(
    ("cookie", "proxy-authorization", SpecialHeaders.mcp_auth.value)
)
JSON_CONTENT_TYPE: Final = MappingProxyType({"content-type": "application/json"})
AKTO_ERRORS: Final = (httpx.RequestError, httpx.HTTPStatusError, Timeout)
EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
JSON_CONTAINER: Final[TypeAdapter[dict[str, object] | list[object]]] = TypeAdapter(dict[str, object] | list[object])


class AktoVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool = Field(validation_alias=AliasChoices("Allowed", "allowed"))
    behaviour: str = Field(default="", validation_alias=AliasChoices("behaviour", "Behaviour"))
    reason: str = Field(default="", validation_alias=AliasChoices("Reason", "reason"))
    modified: bool = Field(default=False, validation_alias=AliasChoices("Modified", "modified"))
    modified_payload: str | dict[str, object] | list[object] = Field(
        default="", validation_alias=AliasChoices("ModifiedPayload", "modifiedPayload")
    )

    @model_validator(mode="before")
    @classmethod
    def null_as_default(cls, data: object) -> object:
        """Nulls take their defaults; a null or missing Allowed goes to unreachable_fallback."""
        fields: Final = as_mapping(data)
        if not fields:
            return data
        return {key: value for key, value in fields.items() if value is not None}

    @property
    def blocks(self) -> bool:
        """An empty behaviour also blocks."""
        return not self.allowed and self.behaviour.strip().lower() in BLOCKING_BEHAVIOURS


class _AktoResponseData(BaseModel):
    guardrailsResult: AktoVerdict | None = None


class _AktoResponse(BaseModel):
    data: _AktoResponseData | None = None


def as_mapping(value: object) -> Mapping[str, object]:
    try:
        return OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return EMPTY


ALLOW: Final = AktoVerdict.model_validate({"allowed": True})


def normalize_positive_setting(value: int | None, default: int) -> int:
    """Unset, zero and negative settings use the default, since none of them can work."""
    return value if value is not None and value > 0 else default


def streaming_sampling_rate_from(litellm_params: LitellmParams) -> int | None:
    """Read from optional_params, or a top-level key that LitellmParams keeps as an extra."""
    nested: Final = litellm_params.optional_params
    configured: Final = (nested.model_dump() if nested else {}).get("streaming_sampling_rate")
    extra: Final = (litellm_params.model_extra or {}).get("streaming_sampling_rate")
    return AktoGuardrailConfigModelOptionalParams.model_validate(
        {"streaming_sampling_rate": configured if configured is not None else extra}
    ).streaming_sampling_rate


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value) if isinstance(value, Mapping) else str(value)


def to_json(value: object) -> str:
    """Encodes values JSON can't, so an unusual value can't skip unreachable_fallback."""
    return json.dumps(value, default=_json_default)


def decode_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return JSON_CONTAINER.validate_json(value)
    except ValidationError:
        return value


def payload_string_leaves(raw: object) -> Mapping[JSONLeafPath, str] | None:
    """String leaves by JSON path, unwrapping {"body": ...}; None when nested too deep."""
    payload: Final = decode_json(raw)
    body: Final = decode_json(as_mapping(payload).get("body", payload))
    leaves: Final = json_string_leaves(body)
    return None if leaves is None else MappingProxyType(dict(leaves))


def masked_texts(texts: tuple[str, ...], sent: object, modified_payload: object) -> tuple[str, ...] | None:
    """texts with Akto's masking applied, or None when the masked leaves don't map back onto them one to one."""
    sent_leaves: Final = payload_string_leaves(sent)
    masked_leaves: Final = payload_string_leaves(modified_payload)
    if sent_leaves is None or masked_leaves is None or sent_leaves.keys() != masked_leaves.keys():
        return None
    changed_paths: Final = tuple(path for path in sent_leaves if sent_leaves[path] != masked_leaves[path])
    changed: Final = frozenset((sent_leaves[path], masked_leaves[path]) for path in changed_paths)
    changes: Final = MappingProxyType(dict(changed))
    if (
        not changes
        or len(changes) != len(changed)
        or not Counter(sent_leaves[path] for path in changed_paths) <= Counter(texts)
    ):
        return None
    return tuple(changes.get(text, text) for text in texts)


def scoped_message(message: object, *, only_tool_results: bool) -> object | None:
    """A Messages API message keeping only its tool_result blocks, or only the rest; None when nothing is left."""
    mapping: Final = as_mapping(message)
    content: Final = mapping.get("content")
    if not isinstance(content, list):
        return None if only_tool_results else message
    kept: Final = tuple(
        block for block in content if (as_mapping(block).get("type") == "tool_result") == only_tool_results
    )
    return {**mapping, "content": kept} if kept else None


def call_type_of(request_data: Mapping[str, object]) -> object:
    return getattr(request_data.get("litellm_logging_obj"), "call_type", None)


def client_sent(request_data: Mapping[str, object], key: str) -> bool:
    return key in as_mapping(as_mapping(request_data.get("proxy_server_request")).get("body"))


def call_details(request_data: Mapping[str, object]) -> Mapping[str, object]:
    """pre_mcp_call data lacks call ids and full headers; the logger's call details have them."""
    logger: Final[object] = request_data.get("litellm_logging_obj")
    return as_mapping(getattr(logger, "model_call_details", None))


def metadata_sources(request_data: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    details: Final = call_details(request_data)
    # LLM request data carries the logger and client-sent litellm_params; post_mcp_call hands over the logger's own
    server_params: Final = EMPTY if "litellm_logging_obj" in request_data else request_data.get("litellm_params")
    return (request_data, as_mapping(server_params), details, as_mapping(details.get("litellm_params")))


def first_value(request_data: Mapping[str, object], key: str) -> object:
    return next((value for source in (request_data, call_details(request_data)) if (value := source.get(key))), None)


INPUT_HOOKS: Final = MappingProxyType(
    {
        "request": frozenset(
            (GuardrailEventHooks.pre_call, GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.logging_only)
        ),
        "response": frozenset(
            (GuardrailEventHooks.post_call, GuardrailEventHooks.post_mcp_call, GuardrailEventHooks.logging_only)
        ),
    }
)


class AktoGuardrail(CustomGuardrail):
    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.akto import (
            AktoConfigModel,
        )

        return AktoConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.post_mcp_call,
            GuardrailEventHooks.logging_only,
        ]

    def __init__(
        self,
        akto_base_url: str | None = None,
        akto_api_key: str | None = None,
        akto_account_id: str | None = None,
        akto_vxlan_id: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        guardrail_timeout: int | None = None,
        *,
        context_source: Literal["ENDPOINT", "AGENTIC"] | None = None,
        akto_metadata: Mapping[str, object] | None = None,
        streaming_sampling_rate: int | None = None,
        file_guardrail_timeout: int | None = None,
        async_handler: AsyncHTTPHandler | None = None,
        **kwargs: Unpack[_CustomGuardrailKwargs],
    ) -> None:
        self.async_handler: AsyncHTTPHandler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        self.akto_base_url = (akto_base_url or os.environ.get("AKTO_GUARDRAIL_API_BASE", "")).rstrip("/")
        if not self.akto_base_url:
            raise ValueError("akto_base_url is required. Set AKTO_GUARDRAIL_API_BASE or pass it in litellm_params.")

        self.akto_api_key = akto_api_key or os.environ.get("AKTO_API_KEY", "")
        if not self.akto_api_key:
            raise ValueError("akto_api_key is required. Set AKTO_API_KEY or pass it in litellm_params.")

        self.akto_account_id = akto_account_id or os.environ.get("AKTO_ACCOUNT_ID", "1000000")
        self.akto_vxlan_id = akto_vxlan_id or os.environ.get("AKTO_VXLAN_ID", "0")
        self.context_source: Literal["ENDPOINT", "AGENTIC"] = context_source or DEFAULT_CONTEXT_SOURCE
        self.akto_metadata: Mapping[str, object] = akto_metadata or EMPTY
        self.streaming_sampling_rate: int = normalize_positive_setting(
            streaming_sampling_rate, DEFAULT_STREAMING_SAMPLING_RATE
        )
        self.guardrail_timeout: int = normalize_positive_setting(guardrail_timeout, DEFAULT_GUARDRAIL_TIMEOUT)
        self.file_guardrail_timeout: int = normalize_positive_setting(
            file_guardrail_timeout, DEFAULT_FILE_GUARDRAIL_TIMEOUT
        )
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback

        init_kwargs: Final[_CustomGuardrailKwargs] = {
            **kwargs,
            "supported_event_hooks": list(self.get_supported_event_hooks()),
        }
        super().__init__(**init_kwargs)

        verbose_proxy_logger.debug(
            "Akto guardrail initialized: base_url=%s fallback=%s",
            self.akto_base_url,
            self.unreachable_fallback,
        )

    def handles(self, input_type: Literal["request", "response"]) -> bool:
        if self.event_hook is None or isinstance(self.event_hook, Mode):
            return True
        configured: Final = self.event_hook if isinstance(self.event_hook, list) else (self.event_hook,)
        return any(GuardrailEventHooks(hook) in INPUT_HOOKS[input_type] for hook in configured)

    @staticmethod
    def resolve_metadata_value(request_data: Mapping[str, object] | None, key: str) -> str | None:
        if request_data is None:
            return None
        values: Final = (
            as_mapping(source.get(name)).get(key)
            for source, name in product(metadata_sources(request_data), ("litellm_metadata", "metadata"))
        )
        value: Final = next((value for value in values if value is not None), None)
        return None if value is None else str(value).strip()

    @staticmethod
    def extract_request_path(request_data: Mapping[str, object]) -> str:
        return AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_request_route") or DEFAULT_REQUEST_PATH

    def prepare_headers(self) -> Mapping[str, str]:
        return MappingProxyType({**JSON_CONTENT_TYPE, "Authorization": self.akto_api_key})

    @staticmethod
    def build_query_params(
        *, guardrails: bool, ingest_data: bool, response_guardrails: bool = False, file_guardrails: bool = False
    ) -> Mapping[str, str]:
        flags: Final = (
            ("guardrails", guardrails),
            ("response_guardrails", response_guardrails),
            ("ingest_data", ingest_data),
            ("file_guardrails", file_guardrails),
        )
        return MappingProxyType({"akto_connector": AKTO_CONNECTOR_NAME, **{name: "true" for name, on in flags if on}})

    @staticmethod
    def client_headers(request_data: Mapping[str, object]) -> Mapping[str, str]:
        """Lowercased, without credentials; full request headers win over metadata's, which pre_mcp_call trims."""
        candidates: Final = (
            as_mapping(source.get(name)).get("headers")
            for name, source in product(("proxy_server_request", "metadata"), metadata_sources(request_data))
        )
        headers: Final = next((found for found in candidates if found), None)
        return MappingProxyType(
            {
                str(key).lower(): str(val)
                for key, val in as_mapping(headers).items()
                if key and val and str(key).lower() not in EXCLUDED_HEADERS
            }
        )

    @staticmethod
    def build_request_headers(request_data: Mapping[str, object]) -> Mapping[str, str]:
        client_headers: Final = AktoGuardrail.client_headers(request_data)
        session_id: Final = (
            first_value(request_data, "litellm_session_id")
            or AktoGuardrail.resolve_metadata_value(request_data, "session_id")
            or get_chain_id_from_headers(dict(client_headers))
            or client_headers.get("mcp-session-id")
            or first_value(request_data, "litellm_trace_id")
        )
        message_id: Final = first_value(request_data, "litellm_call_id")
        trace_ids: Final = ((SESSION_ID_HEADER, session_id), (MESSAGE_ID_HEADER, message_id))
        return MappingProxyType(
            {
                **JSON_CONTENT_TYPE,
                **client_headers,
                **{name: str(value) for name, value in trace_ids if value},
            }
        )

    def messages_api_messages(self, request_data: Mapping[str, object]) -> tuple[object, ...] | None:
        """/v1/messages forwards its messages as sent, and the translated copy drops document and search_result text.

        The guardrail's skip-system, skip-tool and scan-only-tool-results scoping is applied to them here.
        """
        raw_messages: Final = request_data.get("messages")
        if call_type_of(request_data) not in MESSAGES_API_CALL_TYPES or not isinstance(raw_messages, list):
            return None
        only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(self)
        skip_tools: Final = effective_skip_tool_message_for_guardrail(self)
        skip_system: Final = only_tool_results or effective_skip_system_message_for_guardrail(self)
        system: Final = None if skip_system else request_data.get("system")
        scoped: Final = (
            (scoped_message(message, only_tool_results=only_tool_results) for message in raw_messages)
            if only_tool_results or skip_tools
            else iter(raw_messages)
        )
        return (
            *((MappingProxyType({"role": "system", "content": system}),) if system else ()),
            *(message for message in scoped if message is not None),
        )

    def build_request_body(
        self, inputs: GenericGuardrailAPIInputs, request_data: Mapping[str, object]
    ) -> Mapping[str, object]:
        texts: Final = inputs.get("texts") or ()
        scanned: Final = tuple(MappingProxyType({"role": "user", "content": text}) for text in texts)
        raw_input: Final = request_data.get("input")
        request_input: Final = (
            (MappingProxyType({"role": "user", "content": raw_input}),) if isinstance(raw_input, str) else raw_input
        )
        api_messages: Final = self.messages_api_messages(request_data)
        # The Responses API sends "input", so a "messages" key there is a decoy
        raw_messages: Final = (
            None if call_type_of(request_data) in RESPONSES_API_CALL_TYPES else request_data.get("messages")
        )
        messages: Final = (
            api_messages
            if api_messages is not None
            else inputs.get("structured_messages") or raw_messages or scanned or request_input or ()
        )
        model: Final = request_data.get("model") or inputs.get("model") or ""
        tools: Final = inputs.get("tools") or request_data.get("tools")
        tool_calls: Final = inputs.get("tool_calls")
        optional: Final = (("tools", tools), ("functions", request_data.get("functions")), ("tool_calls", tool_calls))
        return MappingProxyType(
            {
                "model": model,
                "messages": without_attachment_content(messages),
                **{key: value for key, value in optional if value},
            }
        )

    @staticmethod
    def model_response(request_data: Mapping[str, object]) -> object:
        """Translators keep a "response" already in the request, so one the client sent isn't the model's."""
        return None if client_sent(request_data, "response") else request_data.get("response")

    @staticmethod
    def build_response_body(
        inputs: GenericGuardrailAPIInputs, request_data: Mapping[str, object]
    ) -> Mapping[str, object]:
        model_response: Final = AktoGuardrail.model_response(request_data)
        if isinstance(model_response, BaseModel):
            return model_response.model_dump()
        response_mapping: Final = as_mapping(model_response)
        if response_mapping:
            return response_mapping
        tool_calls: Final = inputs.get("tool_calls")
        messages: Final = (
            *(MappingProxyType({"content": text, "role": "assistant"}) for text in inputs.get("texts") or ()),
            *((MappingProxyType({"role": "assistant", "tool_calls": tool_calls}),) if tool_calls else ()),
        )
        choices: Final = tuple(MappingProxyType({"message": message}) for message in messages)
        return MappingProxyType({"choices": choices}) if choices else EMPTY

    @staticmethod
    def build_tag_metadata(request_data: Mapping[str, object]) -> Mapping[str, str]:
        identity: Final = (
            ("user_id", AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_user_id")),
            ("team_id", AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_team_id")),
            ("user_email", AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_user_email")),
            ("team_alias", AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_team_alias")),
            ("key_alias", AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_alias")),
        )
        return MappingProxyType({"gen-ai": "Gen AI", **{key: value for key, value in identity if value}})

    def build_envelope(
        self,
        request_data: Mapping[str, object],
        *,
        path: str,
        request_payload: str,
        tag: Mapping[str, str],
        response_payload: str | None = None,
    ) -> Mapping[str, object]:
        # Only the proxy's own record, since clients control forwarding headers
        ip: Final = (self.resolve_metadata_value(request_data, "requester_ip_address") or "").split(",")[0].strip()
        tag_json: Final = to_json(tag)
        return MappingProxyType(
            {
                "path": path,
                "requestHeaders": to_json(self.build_request_headers(request_data)),
                "responseHeaders": to_json(EMPTY if response_payload is None else JSON_CONTENT_TYPE),
                "method": "POST",
                "requestPayload": request_payload,
                "responsePayload": "{}" if response_payload is None else response_payload,
                "ip": ip,
                "destIp": "127.0.0.1",
                "time": str(int(datetime.now().timestamp() * 1000)),
                "statusCode": "200",
                "type": "HTTP/1.1",
                "status": "200",
                "akto_account_id": self.akto_account_id,
                "akto_vxlan_id": self.akto_vxlan_id,
                "is_pending": "false",
                "source": "MIRRORING",
                "direction": None,
                "process_id": None,
                "socket_id": None,
                "daemonset_id": None,
                "enabled_graph": None,
                "tag": tag_json,
                "metadata": tag_json,
                "akto_metadata": to_json(self.akto_metadata),
                "contextSource": self.context_source,
            }
        )

    def build_akto_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        *,
        include_response: bool = False,
    ) -> Mapping[str, object]:
        """Bodies are sent as {"body": "<JSON string>"}."""
        # A response check's inputs are the response, so the request is taken from request_data alone
        request_inputs: Final = GenericGuardrailAPIInputs() if include_response else inputs
        request_body: Final = to_json(self.build_request_body(request_inputs, request_data))
        response_body: Final = to_json(self.build_response_body(inputs, request_data)) if include_response else None
        return self.build_envelope(
            request_data,
            path=self.extract_request_path(request_data),
            request_payload=to_json({"body": request_body}),
            tag=self.build_tag_metadata(request_data),
            response_payload=None if response_body is None else to_json({"body": response_body}),
        )

    def build_mcp_payload(
        self,
        request_data: Mapping[str, object],
        server: str,
        tool: str,
        arguments: Mapping[str, object],
        *,
        result_texts: tuple[str, ...] | None = None,
        definition: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """A JSON-RPC tools/call on /mcp; a tools/list scan sends the tool definition instead."""
        mcp_tags: Final = (
            ("mcp-server", "MCP Server"),
            ("mcp-client", AKTO_CONNECTOR_NAME),
            ("mcp_server_name", server),
            ("tool_name", tool),
            ("call_type", "tool_call" if definition is None else "tool_discovery"),
        )
        tag: Final = MappingProxyType(
            {
                key: value
                for key, value in (*self.build_tag_metadata(request_data).items(), *mcp_tags)
                if key != "gen-ai"
            }
        )
        rpc: Final = MappingProxyType(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": MappingProxyType({"name": tool, "arguments": arguments}),
                "id": 1,
            }
        )
        content: Final = tuple(MappingProxyType({"type": "text", "text": text}) for text in result_texts or ())
        rpc_result: Final = MappingProxyType(
            {"jsonrpc": "2.0", "id": 1, "result": MappingProxyType({"content": content})}
        )
        return self.build_envelope(
            request_data,
            path=MCP_PATH,
            request_payload=to_json(rpc if definition is None else {"tools": (definition,)}),
            tag=tag,
            response_payload=None if result_texts is None else to_json(rpc_result),
        )

    async def send_request(
        self,
        *,
        guardrails: bool,
        ingest_data: bool,
        payload: Mapping[str, object],
        response_guardrails: bool = False,
        file_guardrails: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        endpoint: Final = f"{self.akto_base_url}{HTTP_PROXY_PATH}"
        params: Final = self.build_query_params(
            guardrails=guardrails,
            ingest_data=ingest_data,
            response_guardrails=response_guardrails,
            file_guardrails=file_guardrails,
        )
        headers: Final = self.prepare_headers()
        return await self.async_handler.post(
            url=endpoint,
            data=to_json(payload),
            params=params,  # pyright: ignore[reportArgumentType]  # httpx accepts any Mapping
            headers=headers,  # pyright: ignore[reportArgumentType]  # httpx accepts any Mapping
            timeout=timeout or self.guardrail_timeout,
        )

    @staticmethod
    def parse_verdict(response: httpx.Response) -> AktoVerdict:
        """No verdict allows; a failed or unreadable reply raises so unreachable_fallback decides."""
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Akto returned unexpected status {response.status_code}",
                request=response.request,
                response=response,
            )
        try:
            data: Final = _AktoResponse.model_validate(response.json()).data
        except ValidationError as e:
            raise httpx.RequestError(
                f"Akto returned an unreadable verdict: {e.errors(include_input=False, include_url=False)}",
                request=response.request,
            ) from e
        except ValueError as e:
            raise httpx.RequestError("Akto returned a non-JSON body", request=response.request) from e
        return ALLOW if data is None or data.guardrailsResult is None else data.guardrailsResult

    def handle_unreachable(
        self,
        inputs: GenericGuardrailAPIInputs,
        error: Exception,
        *,
        streamed: bool = False,
    ) -> GenericGuardrailAPIInputs:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Akto unreachable (fail-open): %s",
                str(error),
                exc_info=error,
            )
            return inputs

        verbose_proxy_logger.error("Akto unreachable (fail-closed): %s", str(error))
        if streamed:
            raise HTTPException(status_code=503, detail=UNREACHABLE_REASON)
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=UNREACHABLE_REASON,
            should_wrap_with_default_message=False,
            status_code=503,
        )

    def blocked(self, reason: str, *, streamed: bool) -> Exception:
        """Once a stream started, only an HTTPException gets the endpoint's own error frame."""
        if streamed:
            return HTTPException(status_code=403, detail=reason)
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=reason,
            should_wrap_with_default_message=False,
            status_code=403,
            blocked_content=True,
        )

    @staticmethod
    def is_mcp_call(request_data: Mapping[str, object], logging_obj: "LiteLLMLoggingObj | None" = None) -> bool:
        """The logger decides when there is one, since clients can put MCP keys in a request body."""
        if logging_obj is not None:
            return logging_obj.call_type == CallTypes.call_mcp_tool.value
        return request_data.get("call_type") == CallTypes.call_mcp_tool.value or "mcp_tool_name" in request_data

    @staticmethod
    def mcp_tool_call(request_data: Mapping[str, object]) -> tuple[str, str, Mapping[str, object]]:
        call: Final = as_mapping(request_data.get("mcp_tool_call_metadata"))
        server: Final = request_data.get("mcp_server_name") or call.get("mcp_server_name") or "unknown"
        tool: Final = request_data.get("mcp_tool_name") or call.get("name") or request_data.get("name") or "unknown"
        arguments: Final = request_data.get("mcp_arguments") or call.get("arguments") or request_data.get("arguments")
        return str(server), str(tool), as_mapping(arguments)

    @staticmethod
    def response_mcp_tool_calls(response: object) -> tuple[tuple[str, str, Mapping[str, object]], ...]:
        names_and_arguments: Final = (
            ((call.get("name") or "").split("__"), call.get("arguments"))
            for call in get_tool_calls_from_response(response, include_all_choices=True)
        )
        return tuple(
            (parts[1], "__".join(parts[2:]), arguments or EMPTY)
            for parts, arguments in names_and_arguments
            if len(parts) >= 3 and parts[0] == MCP_TOOL_PREFIX and parts[1] and parts[2]
        )

    async def check_and_record(
        self,
        inputs: GenericGuardrailAPIInputs,
        payload: Mapping[str, object],
        *,
        response: bool = False,
        record: bool = True,
        can_mask: bool = True,
        streamed: bool = False,
    ) -> GenericGuardrailAPIInputs:
        """Masking that can't be applied blocks."""
        try:
            verdict: Final = self.parse_verdict(
                await self.send_request(
                    guardrails=not response,
                    response_guardrails=response,
                    ingest_data=record,
                    payload=payload,
                )
            )
        except AKTO_ERRORS as e:
            return self.handle_unreachable(inputs=inputs, error=e, streamed=streamed)

        masked: Final = (
            masked_texts(
                tuple(inputs.get("texts") or ()),
                payload.get("responsePayload" if response else "requestPayload"),
                verdict.modified_payload,
            )
            if verdict.modified and can_mask
            else None
        )
        blocked_reason: Final = (
            (verdict.reason or DEFAULT_BLOCK_REASON)
            if verdict.blocks
            else UNMASKABLE_REASON
            if verdict.modified and masked is None
            else None
        )
        if blocked_reason is None:
            return inputs if masked is None else {**inputs, "texts": list(masked)}
        raise self.blocked(blocked_reason, streamed=streamed)

    async def check_attachments(self, inputs: GenericGuardrailAPIInputs, request_data: Mapping[str, object]) -> None:
        """Attachments can't be put back masked, so masking blocks."""
        found: Final = request_attachments(request_data)
        if found.malformed_count:
            raise self.blocked(MALFORMED_ATTACHMENT_REASON, streamed=False)
        if found.unsendable_count:
            verbose_proxy_logger.warning(
                "Akto: %d attachment(s) have no inline content or URL to check", found.unsendable_count
            )
        if not found.attachments:
            return
        payload: Final = MappingProxyType(
            {
                **self.build_envelope(
                    request_data,
                    path=self.extract_request_path(request_data),
                    request_payload="{}",
                    tag=self.build_tag_metadata(request_data),
                ),
                "files": tuple(attachment.as_payload() for attachment in found.attachments),
            }
        )
        try:
            verdict: Final = self.parse_verdict(
                await self.send_request(
                    guardrails=False,
                    ingest_data=False,
                    file_guardrails=True,
                    payload=payload,
                    timeout=self.file_guardrail_timeout,
                )
            )
        except AKTO_ERRORS as e:
            self.handle_unreachable(inputs=inputs, error=e)
            return
        if verdict.blocks or verdict.modified:
            raise self.blocked(verdict.reason or DEFAULT_BLOCK_REASON, streamed=False)

    @staticmethod
    async def settle(
        main: Awaitable[GenericGuardrailAPIInputs], *others: Awaitable[object]
    ) -> GenericGuardrailAPIInputs:
        """Waits for every check; raises the first failure, main's first, else returns main's result."""
        main_task: Final = asyncio.ensure_future(main)
        results: Final[list[object]] = await asyncio.gather(main_task, *others, return_exceptions=True)
        failure: Final = next((result for result in results if isinstance(result, BaseException)), None)
        if failure is not None:
            raise failure
        return main_task.result()

    @override
    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        """Every stream check records, as the end-of-stream check can be skipped. Masking a stream blocks."""
        if not self.handles(input_type):
            return inputs

        if self.is_mcp_call(request_data, logging_obj):
            server, tool, arguments = self.mcp_tool_call(request_data)
            # Only a tools/list scan carries the input schema; it is checked, never recorded, even when blocked
            definition: Final = (
                MappingProxyType(
                    {
                        "name": tool,
                        "description": request_data.get("mcp_tool_description") or "",
                        "inputSchema": request_data.get("mcp_input_schema"),
                    }
                )
                if "mcp_input_schema" in request_data
                else None
            )
            return await self.check_and_record(
                inputs,
                self.build_mcp_payload(
                    request_data,
                    server,
                    tool,
                    arguments,
                    result_texts=tuple(inputs.get("texts") or ()) if input_type == "response" else None,
                    definition=definition,
                ),
                response=input_type == "response",
                record=definition is None,
            )

        if input_type == "request":
            return await self.settle(
                self.check_and_record(inputs, self.build_akto_payload(inputs, request_data)),
                self.check_attachments(inputs, request_data),
            )

        streamed: Final = bool(request_data.get("stream"))
        model_response: Final = self.model_response(request_data)
        # A stream's complete response arrives under "response"; a client-sent one may add checks, never skip them
        complete: Final = not streamed or model_response is not None or client_sent(request_data, "response")
        tool_call_source: Final = (
            model_response
            if model_response is not None
            else {"choices": [{"message": {"tool_calls": list(inputs.get("tool_calls") or ())}}]}
        )
        tool_calls: Final = self.response_mcp_tool_calls(tool_call_source) if complete else ()
        return await self.settle(
            self.check_and_record(
                inputs,
                self.build_akto_payload(inputs, request_data, include_response=True),
                response=True,
                can_mask=complete and not streamed,
                streamed=streamed,
            ),
            *(
                self.check_and_record(
                    inputs, self.build_mcp_payload(request_data, *call), can_mask=False, streamed=streamed
                )
                for call in tool_calls
            ),
        )
