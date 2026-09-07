"""
Server-side port of Spotify's ``shunt`` Claude Code plugin
(https://engineering.atspotify.com/2026/9/portal-by-spotify-cut-my-claude-code-token-usage-by-90),
as an auto-router preset rather than a client-side plugin.

shunt intercepts large file reads and boilerplate generation at the client via Claude Code
``PreToolUse`` hooks and delegates them to a cheap worker model. This module does the same
decision on the proxy instead: it arms via ``auto_router_shunt_min_lines`` /
``auto_router_shunt_bulk_read_model`` / ``auto_router_shunt_code_write_model`` on an
auto-router marker deployment (the same "read `litellm_params` off the resolved deployment, no
``guardrails:`` config entry needed" shape as ``auto_router_compression.py``), then a single
always-on ``ShuntGuardrail`` callback injects ``bulk_read``/``code_write`` tool definitions
pre-call and rewrites large ``Read``/``Bash``/``bulk_read``/``code_write`` tool_use blocks
post-call so the file bytes and generated code never reach the routed model's context.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.shunt_rewrite import ShuntBashRewrite
    from litellm.router import Router
    from litellm.types.utils import CallTypesLiteral, ModelResponseStream


@dataclass(frozen=True, slots=True)
class ShuntConfig:
    """An auto router's shunt settings, resolved from its marker deployment."""

    min_lines: int
    bulk_read_model: str
    code_write_model: str


def _config_from_litellm_params(
    litellm_params: Mapping[str, object], *, default_model: str | None
) -> ShuntConfig | None:
    raw_min_lines: Final = litellm_params.get("auto_router_shunt_min_lines")
    if not isinstance(raw_min_lines, int):
        return None
    fallback_model: Final = default_model or ""
    raw_bulk_read: Final = litellm_params.get("auto_router_shunt_bulk_read_model")
    raw_code_write: Final = litellm_params.get("auto_router_shunt_code_write_model")
    return ShuntConfig(
        min_lines=raw_min_lines,
        bulk_read_model=raw_bulk_read if isinstance(raw_bulk_read, str) and raw_bulk_read else fallback_model,
        code_write_model=raw_code_write if isinstance(raw_code_write, str) and raw_code_write else fallback_model,
    )


def shunt_config_for_model(
    llm_router: "Router | None",
    model_alias: str,
    team_id: str | None,
    request_tags: Sequence[str],
) -> ShuntConfig | None:
    """The shunt config of the auto router marker `model_alias` resolves to, or None.

    Mirrors `auto_router_compression.policy_for_model`'s tag-scoped resolution exactly, so an
    alias with several tag-scoped markers cannot arm shunt under one and route under another.
    """
    if llm_router is None:
        return None
    deployments: Final = llm_router.get_model_list(model_name=model_alias, team_id=team_id) or ()
    markers: Final = tuple(
        litellm_params
        for deployment in deployments
        if isinstance(litellm_params := deployment.get("litellm_params"), Mapping)  # pyright: ignore[reportUnnecessaryIsInstance]  # filters out non-Mapping
        and str(litellm_params.get("model", "")).startswith(AUTO_ROUTER_MODEL_PREFIX)
    )
    requested: Final = frozenset(request_tags)
    tag_matched: Final = tuple(
        params for params in markers if (tags := params.get("tags")) and requested.issuperset(frozenset(tags))
    )
    # Untagged only: a marker scoped to tags this request lacks describes other traffic.
    untagged: Final = tuple(params for params in markers if not params.get("tags"))
    # Lazy, so the first marker carrying a config wins and the rest are never read.
    candidates: Final = (_config_from_marker(params) for params in (*tag_matched, *untagged))
    return next((config for config in candidates if config is not None), None)


def _config_from_marker(litellm_params: Mapping[str, object]) -> ShuntConfig | None:
    default_model: Final = litellm_params.get("auto_router_default_model") or litellm_params.get(
        "complexity_router_default_model"
    )
    return _config_from_litellm_params(
        litellm_params, default_model=default_model if isinstance(default_model, str) else None
    )


# Tool descriptions carry shunt's own SKILL.md guidance: this is shunt's skills layer,
# delivered as tool metadata instead of a bundled markdown file, since there is no client-side
# plugin here for a skill file to live in.
BULK_READ_TOOL_NAME: Final = "bulk_read"
CODE_WRITE_TOOL_NAME: Final = "code_write"

# Held as JSON source, not dict literals: these are JSON Schema documents that go straight into
# the outbound provider payload, so they must stay plain JSON-serializable dicts (a
# MappingProxyType raises in the JSON encoder). Parsing the schema from the notation it is
# written in keeps one construction site instead of a suppression on every nested literal.
_TOOL_DEFINITIONS_JSON: Final = """
{
  "bulk_read": {
    "description": "Delegate reading one or more large files to a cheaper model when you only need a summary or an answer to a specific question about their contents, not exact line-level content for editing. Each call is independent; ask a follow-up by calling again with the same paths.",
    "input_schema": {
      "type": "object",
      "properties": {
        "question": {"type": "string", "description": "What to find out from the files."},
        "paths": {"type": "array", "items": {"type": "string"}, "description": "File paths to read."}
      },
      "required": ["question", "paths"]
    }
  },
  "code_write": {
    "description": "Delegate generating boilerplate code (tests, config, stubs, or anything more than 80% predictable from a reference file) to a cheaper model. Match the reference file's patterns, conventions, naming, and style exactly.",
    "input_schema": {
      "type": "object",
      "properties": {
        "spec": {"type": "string", "description": "What to generate."},
        "reference": {"type": "string", "description": "Path to a file whose patterns the output should match."},
        "target": {"type": "string", "description": "Optional path to write the generated code to."}
      },
      "required": ["spec", "reference"]
    }
  }
}
"""


def _anthropic_tool(name: str) -> Mapping[str, object]:
    """The Anthropic-shape tool definition for `name`, as a fresh plain dict.

    Fresh per call, never a shared module-level dict: these go into the outbound payload, and
    handing every request the same mutable object would let one request's downstream mutation
    (a provider transform normalizing a schema in place, say) leak into every later request.
    """
    definition: Final = json.loads(_TOOL_DEFINITIONS_JSON)[name]
    return {"name": name, **definition}  # mutable-ok: goes into the outbound provider payload as plain JSON


def _as_openai_function_tool(anthropic_tool: Mapping[str, object]) -> Mapping[str, object]:
    """The OpenAI `{"type": "function", "function": {...}}` shape of an Anthropic-style tool."""
    return {  # mutable-ok: goes into the outbound provider payload as plain JSON
        "type": "function",
        "function": {  # mutable-ok: same
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }


_SHUNT_CALL_TYPES: Final = frozenset({"completion", "acompletion", "anthropic_messages", "aanthropic_messages"})


def _request_base_url(data: Mapping[str, object]) -> str | None:
    """The scheme+host+port the client actually used to reach this proxy, or None.

    Read from `data["proxy_server_request"]["url"]` (the real request URL, stamped by
    `litellm_pre_call_utils.py` before routing) rather than an env var or a hardcoded
    localhost, so the generated curl reaches the proxy on whatever host/port/scheme the
    caller is actually using.
    """
    proxy_server_request: Final = data.get("proxy_server_request")
    if not isinstance(proxy_server_request, Mapping):
        return None
    raw_url: Final = proxy_server_request.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        return None
    from urllib.parse import urlsplit

    parts: Final = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def _request_auth_header(data: Mapping[str, object]) -> str | None:
    """The client's own `authorization` header, or None.

    Forwarded rather than a stored key: the generated curl authenticates to this proxy as the
    same caller who sent the request, so it is billed and rate-limited the same way, and this
    module never needs to hold or mint a credential of its own.
    """
    secret_fields: Final = data.get("secret_fields")
    if not isinstance(secret_fields, Mapping):
        return None
    raw_headers: Final = secret_fields.get("raw_headers")
    if not isinstance(raw_headers, Mapping):
        return None
    header: Final = raw_headers.get("authorization")
    return header if isinstance(header, str) and header else None


def _resolve_shunt_config(data: Mapping[str, object]) -> ShuntConfig | None:
    """The armed `ShuntConfig` for this request's resolved model, or None.

    Shared by both hooks so pre-call tool injection and post-call rewriting agree on whether
    shunt is armed for this request; `data["model"]` is reliably the caller-facing alias at
    both points, since the router unpacks it into a fresh local `kwargs` at the call site
    rather than mutating the request dict this guardrail was handed.
    """
    model: Final = data.get("model")
    if not isinstance(model, str) or not model:
        return None

    from litellm.proxy.guardrails.auto_router_compression import team_id_from_request
    from litellm.proxy.proxy_server import llm_router
    from litellm.router_strategy.tag_based_routing import (
        _get_tags_from_request_kwargs,  # pyright: ignore[reportPrivateUsage]  # same helper router.py and auto_router_compression.py already use
    )

    return shunt_config_for_model(
        llm_router=llm_router,
        model_alias=model,
        team_id=team_id_from_request(data),
        request_tags=_get_tags_from_request_kwargs(data),
    )


DEFAULT_BULK_READ_QUESTION: Final = "Summarize this file's exports and overall structure."


@dataclass(frozen=True, slots=True)
class _ShuntEndpoints:
    bulk_read_url: str
    code_write_url: str
    auth_header: str


def _endpoints_for_request(data: Mapping[str, object], model_alias: str) -> "_ShuntEndpoints | None":
    """Where this request's generated curl commands should point, or None if unreachable.

    None when the base URL or the caller's own auth header can't be recovered: without both,
    a generated command could not reach this proxy as this caller, so the tool_use is left
    unmodified rather than shipped with a broken command.
    """
    base_url: Final = _request_base_url(data)
    auth_header: Final = _request_auth_header(data)
    if base_url is None or auth_header is None:
        return None
    from urllib.parse import quote

    router_query: Final = f"router={quote(model_alias)}"
    return _ShuntEndpoints(
        bulk_read_url=f"{base_url}/v1/bulk_read?{router_query}",
        code_write_url=f"{base_url}/v1/code_write?{router_query}",
        auth_header=auth_header,
    )


def _string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    strings: Final = tuple(item for item in value if isinstance(item, str))
    return strings if len(strings) == len(value) else None


def _bash_replacement_for_tool_use(
    name: str, tool_input: Mapping[str, object], config: ShuntConfig, endpoints: _ShuntEndpoints
) -> "ShuntBashRewrite | None":
    """The `Bash` command a tool_use becomes, or None to leave it untouched.

    Four sources, all converging on the same generated `curl`: the model's own explicit
    `bulk_read`/`code_write` call, or an untargeted `Read`/bare `cat`/`head`/`tail`/`less`/
    `more` call this guardrail bounds on the model's behalf.
    """
    from litellm.proxy.guardrails.shunt_rewrite import (
        build_bounded_read_command,
        build_bulk_read_command,
        build_code_write_command,
        extract_bare_read_path,
        is_targeted_read,
    )

    if name == BULK_READ_TOOL_NAME:
        question: Final = tool_input.get("question")
        paths: Final = _string_list(tool_input.get("paths"))
        if not isinstance(question, str) or not question or not paths:
            return None
        return build_bulk_read_command(
            question=question,
            paths=paths,
            bulk_read_endpoint=endpoints.bulk_read_url,
            auth_header=endpoints.auth_header,
        )

    if name == CODE_WRITE_TOOL_NAME:
        spec: Final = tool_input.get("spec")
        reference: Final = tool_input.get("reference")
        if not isinstance(spec, str) or not spec or not isinstance(reference, str) or not reference:
            return None
        target: Final = tool_input.get("target")
        return build_code_write_command(
            spec=spec,
            reference=reference,
            target=target if isinstance(target, str) and target else None,
            code_write_endpoint=endpoints.code_write_url,
            auth_header=endpoints.auth_header,
        )

    if name == "Read":
        path: Final = tool_input.get("file_path")
        if not isinstance(path, str) or not path:
            return None
        if is_targeted_read(tool_input.get("offset"), tool_input.get("limit")):
            return None
        return build_bounded_read_command(
            path=path,
            question=DEFAULT_BULK_READ_QUESTION,
            min_lines=config.min_lines,
            bulk_read_endpoint=endpoints.bulk_read_url,
            auth_header=endpoints.auth_header,
        )

    if name == "Bash":
        command: Final = tool_input.get("command")
        if not isinstance(command, str) or not command:
            return None
        bare_path: Final = extract_bare_read_path(command)
        if bare_path is None:
            return None
        return build_bounded_read_command(
            path=bare_path,
            question=DEFAULT_BULK_READ_QUESTION,
            min_lines=config.min_lines,
            bulk_read_endpoint=endpoints.bulk_read_url,
            auth_header=endpoints.auth_header,
        )

    return None


def _rewrite_openai_tool_call(tool_call: object, config: ShuntConfig, endpoints: _ShuntEndpoints) -> bool:
    """Rewrite one OpenAI-shape tool call's `function` in place. Returns whether it changed.

    A custom tool call (`ChatCompletionMessageCustomToolCall`) has no `.function` attribute at
    all, so it is never a shunt-shaped call and is skipped rather than inspected. The return
    value lets a caller streaming the response skip re-serializing a stream nothing in it
    changed, matching `tool_permission.py`'s own "an allowed stream must not be re-serialized".
    """
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    if not isinstance(tool_call, ChatCompletionMessageToolCall):
        return False
    name: Final = tool_call.function.name
    if not isinstance(name, str):
        return False
    try:
        tool_input: Final = json.loads(tool_call.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(tool_input, Mapping):
        return False
    rewrite: Final = _bash_replacement_for_tool_use(name, tool_input, config, endpoints)
    if rewrite is None:
        return False
    tool_call.function = Function(  # rebind-ok: rewrites the provider response in place, mirrors tool_permission.py
        name="Bash",
        arguments=json.dumps({"command": rewrite.command}),  # mutable-ok: serialized immediately, never held
    )
    return True


def _rewrite_anthropic_content_block(block: object, config: ShuntConfig, endpoints: _ShuntEndpoints) -> bool:
    """Rewrite one Anthropic-shape content block's `name`/`input` in place. Returns whether it changed."""
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return False
    name: Final = block.get("name")
    tool_input: Final = block.get("input")
    if not isinstance(name, str) or not isinstance(tool_input, Mapping):
        return False
    rewrite: Final = _bash_replacement_for_tool_use(name, tool_input, config, endpoints)
    if rewrite is None:
        return False
    block["name"] = "Bash"  # rebind-ok: rewrites the provider response in place, mirrors tool_permission.py
    block["input"] = {"command": rewrite.command}  # rebind-ok: same  # mutable-ok: part of the response payload
    return True


def _rewrite_openai_response_in_place(response: object, config: ShuntConfig, endpoints: _ShuntEndpoints) -> bool:
    """Rewrite every shunt-shaped tool call across all choices. Returns whether any changed.

    Built as a tuple comprehension, not a short-circuiting `any(...)`: every tool call must be
    inspected and rewritten regardless of whether an earlier one already changed, so evaluation
    can't stop at the first True the way `any` would.
    """
    from litellm.types.utils import Choices

    results: Final = tuple(
        _rewrite_openai_tool_call(tool_call, config, endpoints)
        for choice in getattr(response, "choices", ())
        if isinstance(choice, Choices)
        for tool_call in choice.message.tool_calls or ()
    )
    return any(results)


def _rewrite_anthropic_response_in_place(
    response: dict,  # mutable-ok: the provider response this rewrites in place, as tool_permission.py does
    config: ShuntConfig,
    endpoints: _ShuntEndpoints,
) -> bool:
    """Rewrite every shunt-shaped tool_use block in `response["content"]`. Returns whether any changed.

    Same full-evaluation shape as `_rewrite_openai_response_in_place`, for the same reason.
    """
    content: Final = response.get("content")
    if not isinstance(content, list):
        return False
    results: Final = tuple(_rewrite_anthropic_content_block(block, config, endpoints) for block in content)
    return any(results)


def _tools_payload(
    existing: Sequence[object], added: Sequence[Mapping[str, object]]
) -> list[object]:  # mutable-ok: outbound provider payload
    """`existing` plus `added`, as the plain list the outbound provider payload must hold."""
    return [*existing, *added]  # mutable-ok: outbound provider payload; transforms downstream append to it


class ShuntGuardrail(CustomLogger):
    """Always-registered callback (see the module docstring) that arms per-request from an
    auto-router marker's `litellm_params`, never from a `guardrails:` config entry.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,  # mutable-ok: CustomLogger declares `data: dict`; narrowing breaks the override
        call_type: "CallTypesLiteral",
    ) -> dict | None:  # mutable-ok: the base class's own return type; the caller re-reads this dict
        if call_type not in _SHUNT_CALL_TYPES:
            return None

        config: Final = _resolve_shunt_config(data)
        if config is None:
            return None

        use_anthropic_format: Final = call_type in ("anthropic_messages", "aanthropic_messages")
        anthropic_tools: Final = (_anthropic_tool(BULK_READ_TOOL_NAME), _anthropic_tool(CODE_WRITE_TOOL_NAME))
        new_tools: Final = (
            anthropic_tools if use_anthropic_format else tuple(_as_openai_function_tool(t) for t in anthropic_tools)
        )
        existing_tools: Final = data.get("tools")
        # Mutating `data` is how this hook is documented to modify a request; litellm_skills/
        # main.py does the same `data["tools"] = ...` assignment.
        data["tools"] = _tools_payload(  # rebind-ok: the hook's documented way to modify a request
            existing_tools if isinstance(existing_tools, list) else (), new_tools
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,  # mutable-ok: CustomLogger declares `data: dict`; narrowing breaks the override
        user_api_key_dict: "UserAPIKeyAuth",
        response: object,
    ) -> object:
        from litellm.types.utils import ModelResponse

        config: Final = _resolve_shunt_config(data)
        if config is None:
            return response

        model: Final = data.get("model")
        endpoints: Final = _endpoints_for_request(data, model) if isinstance(model, str) and model else None
        if endpoints is None:
            return response

        if isinstance(response, ModelResponse):
            _rewrite_openai_response_in_place(response, config, endpoints)
            return response

        if isinstance(response, dict):
            _rewrite_anthropic_response_in_place(response, config, endpoints)
            return response

        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: "AsyncIterable[ModelResponseStream]",
        request_data: dict,  # mutable-ok: CustomLogger declares `request_data: dict`; narrowing breaks the override
    ) -> "AsyncGenerator[ModelResponseStream, None]":
        """Buffer the whole stream, rewrite any shunt-shaped tool_use, then replay it.

        Buffer-then-replay, not per-fragment rewriting, matching `tool_permission.py`'s own
        streaming hook: `input_json_delta` fragments split mid-token (confirmed against a real
        Anthropic trace during design), so a tool_use's `input` can only be read once its
        `content_block_stop` has arrived. Unlike that guardrail, an unparseable or unassemblable
        stream is passed through unmodified rather than raised on: shunt is an optimization, not
        a safety control, so a request should never fail because this rewrite couldn't run.

        The declared return type matches the base class and `tool_permission.py`'s own override:
        on the Anthropic path this actually yields raw `bytes` SSE frames, not
        `ModelResponseStream` objects, the same documented mismatch `tool_permission.py` carries.
        """
        from litellm.main import stream_chunk_builder
        from litellm.proxy.guardrails.anthropic_sse import (
            anthropic_sse_chunks_from_response,
            assemble_anthropic_sse_stream,
            is_raw_sse_stream,
        )
        from litellm.types.utils import ModelResponse, TextCompletionResponse

        # Declared element type matches `tool_permission.py`'s own override rather than the
        # `object` the raw-SSE path really carries: see the docstring's note on that mismatch.
        all_chunks: Final[
            list[ModelResponseStream]
        ] = [  # mutable-ok: stream_chunk_builder/is_raw_sse_stream both take a concrete list
            chunk async for chunk in response
        ]

        config: Final = _resolve_shunt_config(request_data)
        model: Final = request_data.get("model")
        endpoints: Final = (
            _endpoints_for_request(request_data, model)
            if config is not None and isinstance(model, str) and model
            else None
        )
        if config is None or endpoints is None:
            for chunk in all_chunks:
                yield chunk
            return

        if is_raw_sse_stream(all_chunks):
            assembled: Final = assemble_anthropic_sse_stream(all_chunks)
            if assembled is None:
                # Unparseable: pass the stream through exactly as received rather than fail the
                # request over an optimization that could not run.
                for chunk in all_chunks:
                    yield chunk
                return
            changed: Final = _rewrite_openai_response_in_place(assembled, config, endpoints)
            if not changed:
                for chunk in all_chunks:
                    yield chunk
                return
            for sse_chunk in anthropic_sse_chunks_from_response(assembled):
                yield sse_chunk
            return

        assembled_openai: Final[ModelResponse | TextCompletionResponse | None] = stream_chunk_builder(chunks=all_chunks)
        if not isinstance(assembled_openai, ModelResponse):
            for chunk in all_chunks:
                yield chunk
            return
        changed_openai: Final = _rewrite_openai_response_in_place(assembled_openai, config, endpoints)
        if not changed_openai:
            for chunk in all_chunks:
                yield chunk
            return
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        async for chunk in MockResponseIterator(model_response=assembled_openai):
            yield chunk
