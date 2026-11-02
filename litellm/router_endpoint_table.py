"""
The router's factory-built endpoints, in one table both router cores bind from.

Each row names a public router member, the host op that serves it (a litellm function name, or a
dotted path to one under ``litellm``: ``vector_stores.main.asearch``), the call type the routing
decisions key on (stream-timeout split, video-id pin, route-capability pin, the responses socket,
the /v1/messages refusal gate, context compaction, endpoint eligibility), and how the endpoint
dispatches: sync or async, and which helper routes it. A core binds every row and dispatches on
``dispatch`` alone, so an endpoint cannot be bound without a dispatch or dispatched without a row.

Neither core is imported here: ``PythonCore`` and ``RustCore`` both read this module.
"""

import random
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol, get_args

Dispatch = Literal[
    # async
    "assistants",
    "moderation",
    "search",
    "responses",
    "anthropic_messages",
    "generic",
    "generic_client",
    "generic_provider_client",
    "passthrough",
    "response_id",
    "container",
    "vector_store",
    "interaction",
    "managed_agent",
    # sync
    "search_sync",
    "generic_sync",
    "vector_store_sync",
    "vector_store_file_sync",
    "interaction_sync",
    "managed_agent_sync",
]
DISPATCHES: Final[frozenset[str]] = frozenset(get_args(Dispatch))
SYNC_DISPATCHES: Final[frozenset[str]] = frozenset(
    {
        "search_sync",
        "generic_sync",
        "vector_store_sync",
        "vector_store_file_sync",
        "interaction_sync",
        "managed_agent_sync",
    }
)


class EndpointSpec:
    """One factory-built endpoint: ``name`` on the router, served by the host op ``host_op``."""

    __slots__ = ("call_type", "dispatch", "host_op", "name")

    def __init__(self, name: str, host_op: str, call_type: str, dispatch: Dispatch) -> None:
        self.name: Final = name
        self.host_op: Final = host_op
        self.call_type: Final = call_type
        self.dispatch: Final[Dispatch] = dispatch

    @property
    def sync(self) -> bool:
        return self.dispatch in SYNC_DISPATCHES

    def __repr__(self) -> str:
        return f"EndpointSpec({self.name!r}, {self.host_op!r}, {self.call_type!r}, {self.dispatch!r})"


def _rows() -> Iterable[EndpointSpec]:
    def row(name: str, host_op: str, dispatch: Dispatch, call_type: str | None = None) -> EndpointSpec:
        return EndpointSpec(name, host_op, call_type or name, dispatch)

    for name in (
        "acreate_assistants",
        "adelete_assistant",
        "aget_assistants",
        "acreate_thread",
        "aget_thread",
        "a_add_message",
        "aget_messages",
        "arun_thread",
    ):
        yield row(name, name, "assistants", "assistants")
    yield row("amoderation", "amoderation", "moderation", "moderation")
    yield row("aanthropic_messages", "anthropic_messages", "anthropic_messages", "anthropic_messages")
    yield row("anthropic_messages", "anthropic_messages", "anthropic_messages")
    yield row("aresponses", "aresponses", "responses")
    yield row("responses", "responses", "generic_sync")
    for name in ("aget_responses", "acancel_responses", "acompact_responses", "adelete_responses", "alist_input_items"):
        yield row(name, name, "response_id")
    for name in ("afile_delete", "afile_content"):
        yield row(name, name, "generic_provider_client")
    yield row("afile_list", "afile_list", "generic", "alist_files")
    for name in ("acreate_realtime_client_secret", "arealtime_calls", "acreate_realtime_transcription_session"):
        yield row(name, name, "generic_client")
    for name in (
        "_arealtime",
        "_aresponses_websocket",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "alist_fine_tuning_jobs",
        "aretrieve_fine_tuning_job",
        "aimage_edit",
        "aadapter_generate_content",
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
    ):
        yield row(name, name, "generic")
    yield row("allm_passthrough_route", "allm_passthrough_route", "passthrough")
    # vector stores (`avector_store_create` is a method on both cores: it routes to one deployment)
    for name in ("search", "retrieve", "list", "update", "delete"):
        yield row(f"avector_store_{name}", f"vector_stores.main.a{name}", "vector_store")
    for name in ("search", "create", "retrieve", "list", "update", "delete"):
        yield row(f"vector_store_{name}", f"vector_stores.main.{name}", "vector_store_sync")
    for name, target in (
        ("create", "create"),
        ("list", "list"),
        ("retrieve", "retrieve"),
        ("content", "retrieve_content"),
        ("update", "update"),
        ("delete", "delete"),
    ):
        yield row(f"avector_store_file_{name}", f"vector_store_files.main.a{target}", "vector_store")
        yield row(f"vector_store_file_{name}", f"vector_store_files.main.{target}", "vector_store_file_sync")
    # Google GenAI, OCR, search, decisions, systemone, extraction
    for name in ("agenerate_content", "agenerate_content_stream"):
        yield row(name, f"google_genai.{name}", "generic")
    for name in ("generate_content", "generate_content_stream"):
        yield row(name, f"google_genai.{name}", "generic_sync")
    for module, name in (("ocr", "ocr"), ("decisions", "decisions"), ("systemone", "systemone")):
        yield row(f"a{name}", f"{module}.a{name}", "generic")
        yield row(name, f"{module}.{name}", "generic_sync")
    yield row("aextraction", "extraction.aextraction", "generic")
    yield row("extraction", "extraction.extraction", "generic_sync")
    yield row("asearch", "search.asearch", "search")
    yield row("search", "search.search", "search_sync")
    # videos
    for name in (
        "generation",
        "list",
        "status",
        "content",
        "remix",
        "create_character",
        "get_character",
        "edit",
        "extension",
    ):
        yield row(f"avideo_{name}", f"videos.avideo_{name}", "generic")
        yield row(f"video_{name}", f"videos.video_{name}", "generic_sync")
    # containers and their files
    for name in ("create_container", "list_containers", "retrieve_container", "delete_container"):
        yield row(f"a{name}", f"containers.a{name}", "container")
        yield row(name, f"containers.{name}", "generic_sync")
    for name in (
        "upload_container_file",
        "list_container_files",
        "retrieve_container_file",
        "delete_container_file",
        "retrieve_container_file_content",
    ):
        yield row(f"a{name}", f"containers.endpoint_factory._generated_endpoints.a{name}", "container")
        yield row(name, f"containers.endpoint_factory._generated_endpoints.{name}", "generic_sync")
    # Google Interactions and Managed Agents
    for name in ("create", "get", "delete", "cancel"):
        yield row(f"a{name}_interaction", f"interactions.a{name}", "interaction")
        yield row(f"{name}_interaction", f"interactions.{name}", "interaction_sync")
    for name, target in (
        ("create_agent", "create"),
        ("list_agents", "list"),
        ("get_agent", "get"),
        ("delete_agent", "delete"),
        ("list_agent_versions", "list_versions"),
    ):
        yield row(f"a{name}", f"interactions.agents.a{target}", "managed_agent")
        yield row(name, f"interactions.agents.{target}", "managed_agent_sync")


ENDPOINTS: Final[Mapping[str, EndpointSpec]] = MappingProxyType({spec.name: spec for spec in _rows()})


def _dispatch_by_call_type() -> Mapping[str, Dispatch]:
    table: Final[dict[str, Dispatch]] = {"avector_store_create": "vector_store"}  # mutable-ok: built once here
    for spec in ENDPOINTS.values():
        known = table.setdefault(spec.call_type, spec.dispatch)
        if known != spec.dispatch:
            raise RuntimeError(f"call type {spec.call_type!r} dispatches both as {known!r} and {spec.dispatch!r}")
    return MappingProxyType(table)


# `factory_function(original_function, call_type=...)`: how each call type dispatches.
DISPATCH_BY_CALL_TYPE: Final[Mapping[str, Dispatch]] = _dispatch_by_call_type()


def dispatch_for(call_type: str) -> Dispatch:
    """How an endpoint of ``call_type`` dispatches; an unknown call type is refused when bound."""
    dispatch: Final = DISPATCH_BY_CALL_TYPE.get(call_type)
    if dispatch is None:
        raise ValueError(f"no router endpoint dispatches the call type {call_type!r}")
    return dispatch


class _Invokes(Protocol):
    def invoke(self, op: str) -> Callable[..., object]: ...


class EndpointHandler:
    """An endpoint's handler: the function ``resolve()`` returns, looked up at each call, tagged with
    the endpoint's ``call_type`` so the routing decisions key on the endpoint, never on the
    function's ``__name__`` (a wrapped or renamed host function keeps them)."""

    __slots__ = ("call_type", "op", "resolve")

    def __init__(self, call_type: str, resolve: Callable[[], Callable[..., object]], op: str | None = None) -> None:
        self.call_type: Final = call_type
        self.resolve: Final = resolve
        self.op: Final = op

    def __call__(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: the endpoint's request
        return self.resolve()(*args, **kwargs)

    def __deepcopy__(self, memo: dict[int, object]) -> "EndpointHandler":  # mutable-ok: copy.deepcopy's memo
        return self

    def __repr__(self) -> str:
        return f"EndpointHandler({self.call_type!r}, op={self.op!r})"


def host_handler(host: _Invokes, op: str, call_type: str) -> EndpointHandler:
    """The host op ``op`` as the handler of an endpoint of ``call_type``, resolved at each call."""
    return EndpointHandler(call_type, lambda: host.invoke(op), op)


def as_handler(function: Callable[..., object], call_type: str) -> EndpointHandler:
    """``function`` as the handler of an endpoint of ``call_type`` (kept as given when it already is one)."""
    if isinstance(function, EndpointHandler):
        return function
    return EndpointHandler(call_type, lambda: function)


def handler_call_type(function: object) -> str | None:
    """The call type the routing decisions key on: the endpoint's, else the function's ``__name__``
    (a function a caller hands the router's generic call directly)."""
    if isinstance(function, EndpointHandler):
        return function.call_type
    name: Final[object] = getattr(function, "__name__", None)
    return name if isinstance(name, str) else None


def handler_identity(function: object) -> tuple[str | None, str | None]:
    """``(call type, module)`` for ``endpoint_selection``: an endpoint's call type names it alone."""
    if isinstance(function, EndpointHandler):
        return function.call_type, None
    module: Final[object] = getattr(function, "__module__", None)
    return handler_call_type(function), module if isinstance(module, str) else None


def with_provider(
    custom_llm_provider: str | None, kwargs: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the call's kwargs
    """The request with the wrapper's ``custom_llm_provider`` when the request names none."""
    if custom_llm_provider and "custom_llm_provider" not in kwargs:
        return {**kwargs, "custom_llm_provider": custom_llm_provider}
    return dict(kwargs)


def managed_agent_request(
    custom_llm_provider: str | None, kwargs: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the call's kwargs
    """A Managed Agents request: the given provider, else ``gemini``."""
    request: Final = with_provider(custom_llm_provider, kwargs)
    request.setdefault("custom_llm_provider", "gemini")
    return request


def interaction_request(
    custom_llm_provider: str | None, kwargs: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the call's kwargs
    """An Interactions request: the given provider, else ``gemini``; an agent name sent as the model
    is no model group, so the model is cleared and the call is not routed."""
    request: Final = managed_agent_request(custom_llm_provider, kwargs)
    if request.get("agent") and request.get("model") == request.get("agent"):
        request["model"] = None
    return request


_SEARCH_PROTECTED_PARAMS: Final = frozenset({"search_provider", "api_key", "api_base"})


def search_tool_call(
    search_tools: Iterable[Mapping[str, object]], search_tool_name: str, kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """``SearchAPIRouter.async_search_with_fallbacks_helper``'s call: one of the search tools named
    ``search_tool_name`` at random, with its provider and credentials, its params under the request's."""
    tools: Final = [tool for tool in search_tools if tool.get("search_tool_name") == search_tool_name]
    if not tools:
        raise ValueError(f"Search tool '{search_tool_name}' not found in router.search_tools")
    selected: Final = random.choice(tools)  # load balancing across the providers sharing the name
    raw_params: Final[object] = selected.get("litellm_params") or {}
    params: Final[Mapping[str, object]] = raw_params if isinstance(raw_params, Mapping) else {}  # pyright: ignore[reportUnknownVariableType]  # a configured tool's params
    search_provider: Final = params.get("search_provider")
    if not search_provider:
        raise ValueError(f"search_provider not found in litellm_params for search tool '{search_tool_name}'")
    search_params: Final = {
        key: value
        for source in (params, kwargs)
        for key, value in source.items()
        if key not in _SEARCH_PROTECTED_PARAMS and value is not None
    }  # comprehension-ok: the tool's params, then the request's
    return MappingProxyType(
        {
            "search_provider": search_provider,
            "api_key": params.get("api_key"),
            "api_base": params.get("api_base"),
            **search_params,
        }
    )


__all__ = [
    "DISPATCHES",
    "DISPATCH_BY_CALL_TYPE",
    "ENDPOINTS",
    "SYNC_DISPATCHES",
    "Dispatch",
    "EndpointHandler",
    "EndpointSpec",
    "as_handler",
    "dispatch_for",
    "handler_call_type",
    "handler_identity",
    "host_handler",
    "interaction_request",
    "managed_agent_request",
    "search_tool_call",
    "with_provider",
]
