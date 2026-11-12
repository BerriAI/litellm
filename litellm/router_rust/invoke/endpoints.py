"""
The Rust router core's endpoints: what each public router method does before, around and after
the routed call, as the Python router's endpoint methods do it.

An endpoint is either routed (the Rust core picks the deployment, retries and falls back, and the
host runs each attempt) or served directly (the Python router calls litellm's function without
routing: no model group, an id that names no deployment, a fan-out over every deployment). Every
call out goes through the host: ``host.invoke(op)``, where an op is a litellm function name or a
dotted path to one (``vector_stores.main.asearch``), looked up when it is called.
"""

import asyncio
import copy
import dataclasses
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
)
from litellm.litellm_core_utils.batch_utils import get_router_metadata_variable_name
from litellm.litellm_core_utils.core_helpers import without
from litellm.litellm_core_utils.routing_common_utils import without_router_only_kwargs
from litellm.responses.retrieve_stream_query import without_retrieve_stream_defaults
from litellm.router_api import RoutingContext
from litellm.router_rust.invoke.shaping import (
    ROUTER_CLIENTS_TARGET,
    STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS,
    Shaping,
    as_dict,
    deployment_params_with_request_reasoning_override,
    drop_unsupported_classifier_reasoning_effort,
    metadata_bucket,
    stamp_deployment_num_retries,
    stamp_failed_deployment_id,
    stamp_retry_skip_deployment_id,
    update_kwargs_with_deployment,
)
from litellm.router_strategy.complexity_router.context_compaction import (
    Surface,
    compact_to_fit,
    initialize_compaction_state,
    surface_for_call,
)
from litellm.router_utils.endpoint_eligibility import endpoint_for_call, endpoint_selection
from litellm.vector_stores.search_query import store_search_params

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.router_api import RouterRegistry
    from litellm.router_rust.core import RustCore
    from litellm.router_rust.steps.selection import StrategySelectors
    from litellm.rust_bridge._native import RouterCoreHandle
    from litellm.types.router import Deployment


class EndpointCore(Protocol):
    """What the endpoints use of the core they serve: its host, Rust handle, settings, routing
    cache, registry lookups and call drivers. ``RustCore`` provides it through ``_EndpointPort``;
    a Rust host's endpoint layer would have to provide the same list."""

    @property
    def router(self) -> "RustCore":
        """The core itself, for the provider handlers and helpers that take a router."""
        ...

    @property
    def public(self) -> object:
        """The ``Router`` that owns the core, for litellm functions that take ``router=``."""
        ...

    @property
    def host(self) -> "PythonHost": ...

    @property
    def handle(self) -> "RouterCoreHandle": ...

    @property
    def resolved(self) -> "Resolved": ...

    @property
    def shaping(self) -> Shaping: ...

    @property
    def routing_cache(self) -> "DualCache": ...

    @property
    def selectors(self) -> "StrategySelectors": ...

    @property
    def endpoints(self) -> "RustEndpoints": ...

    async def drive(
        self,
        plan: "Plan",
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        passthrough: Callable[[Mapping[str, object], str], Awaitable[object]] | None = None,
        prepared_already: bool = False,
    ) -> object: ...

    def call_sync(
        self, plan: "Plan", request: Mapping[str, object], overrides: Mapping[str, object] | None = None
    ) -> object: ...

    def routing_callbacks(self) -> Iterable[object]: ...

    def resolve_alias(self, model: str) -> str | None: ...

    def group_deployments(self, model_name: str) -> Sequence[Mapping[str, object]] | None:
        """``_core_get_model_list(model_name=...)`` on the host registry."""
        ...

    def group_members(self, model_name: str) -> Sequence[tuple[object, object]] | None: ...

    def routed_name(self, model: object) -> str:
        """``RustCore.routed_name``: what a ``model`` that is no string raises, or the name it routes under."""
        ...

    def service_logged(
        self, call_type: str, start_time: float, end_time: float, request: Mapping[str, object]
    ) -> None: ...

    async def set_response_headers(
        self, response: object, model_group: str | None, request_kwargs: Mapping[str, object]
    ) -> object: ...

    def configured_deployment(self, model_id: str) -> "Deployment | None":
        """``_core_get_deployment``: the deployment as the host registry holds it now."""
        ...

    def registry(self) -> "RouterRegistry": ...

    async def select(
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the RouterCore signature
        *,
        kind: str = "pick",
        specific_deployment: bool | None = None,
        messages: object = None,
        batch_upload: bool = False,
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]: ...

    async def acompletion(self, *args: object, **kwargs: object) -> object: ...  # kwargs-ok: endpoint payload


Kind = Literal[  # rebind-ok: rebound as the Python router rebinds it
    "chat",
    "text",
    "embedding",
    "image",
    "transcription",
    "speech",
    "rerank",
    "adapter",
    "generic",
    "generic_sync",
    "batch",
    "cancel_batch",
    "vector_store_create",
    "moderation",
    "create_file",
    "search",
]
Stamp = Literal["deployment", "retry_skip", "none"]  # rebind-ok: rebound as the Python router rebinds it
CountKey = Literal["deployment", "group", "empty"]  # rebind-ok: rebound as the Python router rebinds it

# The text of the user message each attempt helper hands its pick in place of the request's own
# (`_aimage_generation`, `_aadapter_completion`, `_acreate_batch`, ...).
_PICK_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "image": "prompt",
        "transcription": "prompt",
        "speech": "prompt",
        "adapter": "default text",
        "batch": "files-api-fake-text",
        "create_file": "files-api-fake-text",
        "cancel_batch": "batch-api-fake-text",
        "vector_store_create": "vector-store-api-fake-text",
    }
)


def _user_message(content: object) -> list[dict[str, object]]:  # mutable-ok: a fresh list, as each pick builds it
    return [{"role": "user", "content": content}]


PINNED_DEPLOYMENT_KEY: Final = "_pinned_deployment_id"

# Request keys the router reads to route the call; the deployment call never receives them.
ROUTING_KEYS: Final = frozenset(
    {
        "model",
        PINNED_DEPLOYMENT_KEY,
        "num_retries",
        "disable_fallbacks",
        "specific_deployment",
        "_router_weights",
        "_retry_skipped_deployment_ids",
        "_excluded_deployment_ids",
        "_target_order",
        "original_function",
        # The walk's own settings, which `async_function_with_fallbacks` pops before the attempt.
        "model_group_retry_policy",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
        "mock_testing_rate_limit_error",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """How one endpoint runs its routed attempts."""

    endpoint: str
    host_op: str
    kind: Kind
    sync: bool = False
    # The bucket `_update_kwargs_before_fallbacks` writes the requested group into; None for the
    # endpoints that write `metadata.model_group` alone (no trace id, no alias).
    prepare_field: str | None = "metadata"
    function_name: str | None = None
    splits_stream_timeout: bool = False
    stamp: Stamp = "retry_skip"
    total_key: CountKey | None = "deployment"
    success_key: CountKey | None = "deployment"
    fail_key: CountKey | None = "deployment"
    # One attempt, no retries and no fallbacks.
    single_attempt: bool = False
    # Whether the attempt's metadata carries the walk's keys (``original_model_group``, the retry
    # counts): not for an endpoint whose metadata goes to the provider as the caller sent it.
    stamps_walk: bool = True
    # The named parameter the endpoint takes beside **kwargs, sent between the deployment params and
    # the request (`{**data, name: value, **kwargs}`).
    named: tuple[str, ...] = ()

    @property
    def handler(self) -> str:
        """The handler name the host op ends in, which route capabilities and pre-routing strategies key on."""
        return self.host_op.rsplit(".", 1)[-1]

    @property
    def on_its_own_loop(self) -> bool:
        """``function_with_fallbacks`` runs these sync endpoints' whole walk on a private event loop."""
        return self.sync and self.kind in ("chat", "embedding", "image")

    def pick_input(self, request: Mapping[str, object]) -> tuple[object, object]:
        """The ``messages`` and ``input`` the endpoint's attempt helper hands its pick, which every
        selection stage reads: the pre-call checks run when either is set."""
        match self.kind:
            case "chat":
                return request.get("messages"), None
            case "text":
                return _user_message(request.get("prompt")), None
            case "embedding":
                return None, request.get("input")
            case "rerank" | "moderation" | "search":
                return None, None
            case "generic" | "generic_sync":
                return request.get("messages"), request.get("input")
            case _:
                return _user_message(_PICK_TEXTS[self.kind]), None

    @property
    def sets_headers(self) -> bool:
        """Whether the response gets the router's headers (the sync generic wrapper adds none)."""
        return self.kind != "generic_sync"

    @property
    def stamps_failed_deployment(self) -> bool:
        """Whether the attempt stamps ``failed_deployment_id`` on its error, which weighted failover
        reads: every endpoint but file and batch creation (whose result belongs to the deployment
        that served it), moderation's pass-through, search and vector store creation (whose
        handler calls the provider without the attempt wrapper)."""
        return self.kind not in ("batch", "cancel_batch", "create_file", "moderation", "search", "vector_store_create")

    @property
    def metadata_field(self) -> str:
        """Where `_update_kwargs_with_deployment` writes the deployment it routed to."""
        return get_router_metadata_variable_name(function_name=self.function_name)


def _generic(endpoint: str, host_op: str | None = None, *, sync: bool = False, single_attempt: bool = True) -> Plan:
    if sync:
        return Plan(
            endpoint=endpoint,
            host_op=host_op or endpoint,
            kind="generic_sync",
            sync=True,
            prepare_field="litellm_metadata",
            function_name="generic_api_call",
            splits_stream_timeout=_handler_name(host_op or endpoint) in STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS,
            stamp="none" if single_attempt else "deployment",
            fail_key="group",
            single_attempt=single_attempt,
        )
    return Plan(
        endpoint=endpoint,
        host_op=host_op or endpoint,
        kind="generic",
        prepare_field="litellm_metadata",
        function_name="_ageneric_api_call_with_fallbacks",
        splits_stream_timeout=_handler_name(host_op or endpoint) in STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS,
        stamp="deployment",
        fail_key="group",
    )


def _handler_name(host_op: str) -> str:
    return host_op.rsplit(".", 1)[-1]


def invoked_handler_name(core: EndpointCore, plan: Plan) -> str | None:
    """The generic handler's ``__name__`` (`original_generic_function.__name__`): the function the
    host serves for the op, which decides the stream-timeout split and the responses socket."""
    function: Final[object] = core.host.invoke(plan.host_op)  # pyright: ignore[reportUnknownMemberType]  # the core's host
    name: Final[object] = getattr(function, "__name__", None)
    return name if isinstance(name, str) else None


# The endpoints whose attempt helper runs inside `make_call`'s `endpoint_selection`, under the
# name of its `original_function`.
_SELECTION_CALLS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "acompletion": "_acompletion",
        "completion": "_completion",
        "atext_completion": "_atext_completion",
    }
)


def selection_endpoint(core: EndpointCore, plan: Plan) -> str | None:
    """The endpoint ``endpoint_selection`` names around the plan's pick. A sync generic call never
    reaches ``make_call``, and a generic attempt reads the handler's name and module."""
    call: Final = _SELECTION_CALLS.get(plan.endpoint)
    if call is not None:
        return endpoint_for_call(call)
    if plan.kind != "generic":
        return None
    function: Final[object] = core.host.invoke(plan.host_op)  # pyright: ignore[reportUnknownMemberType]  # the core's host
    return endpoint_for_call(getattr(function, "__name__", None), getattr(function, "__module__", None))


PLANS: Final[Mapping[str, Plan]] = MappingProxyType(
    {
        "acompletion": Plan(
            "acompletion", "acompletion", "chat", splits_stream_timeout=True, stamp="deployment", named=("messages",)
        ),
        "completion": Plan(
            "completion",
            "completion",
            "chat",
            sync=True,
            splits_stream_timeout=True,
            stamp="deployment",
            total_key=None,
            success_key=None,
            fail_key=None,
            named=("messages",),
        ),
        "atext_completion": Plan(
            "atext_completion",
            "atext_completion",
            "text",
            splits_stream_timeout=True,
            fail_key="group",
            named=("prompt",),
        ),
        # `_asearch_with_fallbacks`: the search tool name routes as the model, with no deployment
        # picked; its helper picks among the router's search tools.
        "asearch": Plan(
            "asearch",
            "search.asearch",
            "search",
            prepare_field="litellm_metadata",
            stamp="none",
            total_key=None,
            success_key=None,
            fail_key=None,
        ),
        "aembedding": Plan("aembedding", "aembedding", "embedding", named=("input",)),
        "embedding": Plan("embedding", "embedding", "embedding", sync=True, named=("input",)),
        "aimage_generation": Plan("aimage_generation", "aimage_generation", "image", named=("prompt",)),
        "image_generation": Plan(
            "image_generation",
            "image_generation",
            "image",
            sync=True,
            prepare_field=None,
            total_key="empty",
            success_key="empty",
            fail_key="empty",
            named=("prompt",),
        ),
        "atranscription": Plan(
            "atranscription",
            "atranscription",
            "transcription",
            total_key="group",
            success_key="group",
            fail_key="group",
            named=("file",),
        ),
        "aspeech": Plan(
            "aspeech",
            "aspeech",
            "speech",
            total_key="group",
            success_key="group",
            fail_key="group",
            named=("input",),
        ),
        "arerank": Plan("arerank", "arerank", "rerank"),
        "aadapter_completion": Plan(
            "aadapter_completion",
            "aadapter_completion",
            "adapter",
            prepare_field=None,
            splits_stream_timeout=True,
            fail_key="group",
        ),
        "acreate_batch": Plan(
            "acreate_batch",
            "acreate_batch",
            "batch",
            prepare_field="litellm_metadata",
            function_name="_acreate_batch",
            fail_key="group",
        ),
        "acancel_batch": Plan(
            "acancel_batch",
            "acancel_batch",
            "cancel_batch",
            prepare_field="litellm_metadata",
            function_name="_acancel_batch",
            fail_key="group",
        ),
        "acreate_file": Plan(
            "acreate_file",
            "acreate_file",
            "create_file",
            function_name="acreate_file",
            stamp="none",
            total_key=None,
            success_key=None,
            fail_key="group",
        ),
        "avector_store_create": Plan(
            "avector_store_create",
            "vector_stores.acreate",
            "vector_store_create",
            prepare_field=None,
            function_name="avector_store_create",
            stamp="none",
            fail_key="group",
            single_attempt=True,
            stamps_walk=False,
        ),
        "amoderation": Plan(
            "amoderation",
            "amoderation",
            "moderation",
            stamp="none",
            total_key=None,
            success_key=None,
            fail_key=None,
            single_attempt=True,
        ),
        # The Python router's anthropic messages host op is `anthropic_messages`.
        "aanthropic_messages": _generic("aanthropic_messages", "anthropic_messages"),
        "anthropic_messages": _generic("anthropic_messages", "anthropic_messages"),
        "aresponses": _generic("aresponses"),
        "responses": _generic("responses", sync=True, single_attempt=False),
    }
)

# --- the generic endpoints ------------------------------------------------------------------

# endpoint -> host op, for the async generic endpoints routed through
# `_ageneric_api_call_with_fallbacks` when they carry a model.
_HOST_GENERIC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "afile_delete": "afile_delete",
        "afile_content": "afile_content",
        "afile_list": "afile_list",
        "aget_responses": "aget_responses",
        "acancel_responses": "acancel_responses",
        "acompact_responses": "acompact_responses",
        "adelete_responses": "adelete_responses",
        "alist_input_items": "alist_input_items",
        "_arealtime": "_arealtime",
        "acreate_realtime_client_secret": "acreate_realtime_client_secret",
        "arealtime_calls": "arealtime_calls",
        "acreate_realtime_transcription_session": "acreate_realtime_transcription_session",
        "acreate_realtime_translation_client_secret": "acreate_realtime_translation_client_secret",
        "arealtime_translation_calls": "arealtime_translation_calls",
        "_aresponses_websocket": "_aresponses_websocket",
        "acreate_fine_tuning_job": "acreate_fine_tuning_job",
        "acancel_fine_tuning_job": "acancel_fine_tuning_job",
        "alist_fine_tuning_jobs": "alist_fine_tuning_jobs",
        "aretrieve_fine_tuning_job": "aretrieve_fine_tuning_job",
        "aimage_edit": "aimage_edit",
        "allm_passthrough_route": "allm_passthrough_route",
        "acreate_skill": "acreate_skill",
        "alist_skills": "alist_skills",
        "aget_skill": "aget_skill",
        "adelete_skill": "adelete_skill",
        "agenerate_content": "google_genai.agenerate_content",
        "agenerate_content_stream": "google_genai.agenerate_content_stream",
        "aocr": "ocr.aocr",
        "adecisions": "decisions.adecisions",
        "asystemone": "systemone.asystemone",
        "aextraction": "extraction.aextraction",
        **{f"avideo_{name}": f"videos.avideo_{name}" for name in ("generation", "list", "status", "content", "remix")},
        **{
            name: f"containers.{name}"
            for name in ("acreate_container", "alist_containers", "aretrieve_container", "adelete_container")
        },
        **{
            name: f"containers.endpoint_factory._generated_endpoints.{name}"
            for name in (
                "aupload_container_file",
                "alist_container_files",
                "aretrieve_container_file",
                "adelete_container_file",
                "aretrieve_container_file_content",
            )
        },
        **{
            f"avector_store_{name}": f"vector_stores.main.a{name}"
            for name in ("search", "retrieve", "list", "update", "delete")
        },
        **{
            f"avector_store_file_{name}": f"vector_store_files.main.a{target}"
            for name, target in (
                ("create", "create"),
                ("list", "list"),
                ("retrieve", "retrieve"),
                ("content", "retrieve_content"),
                ("update", "update"),
                ("delete", "delete"),
            )
        },
        "acreate_interaction": "interactions.acreate",
        "aget_interaction": "interactions.aget",
        "adelete_interaction": "interactions.adelete",
        "acancel_interaction": "interactions.acancel",
    }
)

_SYNC_GENERIC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "generate_content": "google_genai.generate_content",
        "generate_content_stream": "google_genai.generate_content_stream",
        "ocr": "ocr.ocr",
        # The sync wrapper routes a search tool name as a model group, unlike `asearch`.
        "search": "search.search",
        "decisions": "decisions.decisions",
        "systemone": "systemone.systemone",
        "extraction": "extraction.extraction",
        **{f"video_{name}": f"videos.video_{name}" for name in ("generation", "list", "status", "content", "remix")},
        **{
            name: f"containers.{name}"
            for name in ("create_container", "list_containers", "retrieve_container", "delete_container")
        },
        **{
            f"vector_store_{name}": f"vector_stores.main.{name}"
            for name in ("search", "create", "retrieve", "list", "update", "delete")
        },
    }
)

_ROUTER_RETRIED_SYNC_GENERIC: Final = frozenset({"generate_content", "generate_content_stream"})

_CONTAINER_FILE_ROUTED: Final = frozenset(
    {
        "aupload_container_file",
        "alist_container_files",
        "aretrieve_container_file",
        "adelete_container_file",
        "aretrieve_container_file_content",
    }
)
_VIDEO_DIRECT: Final = frozenset({"avideo_create_character", "avideo_get_character", "avideo_edit", "avideo_extension"})
_SYNC_VIDEO_UNBRANCHED: Final = frozenset(
    {"video_create_character", "video_get_character", "video_edit", "video_extension"}
)
_SYNC_CONTAINER_FILES_UNBRANCHED: Final = frozenset(
    {
        "upload_container_file",
        "list_container_files",
        "retrieve_container_file",
        "delete_container_file",
        "retrieve_container_file_content",
    }
)
_UNBRANCHED: Final = frozenset(
    {
        "aadapter_generate_content",
        *_SYNC_VIDEO_UNBRANCHED,
        *_SYNC_CONTAINER_FILES_UNBRANCHED,
        "get_interaction",
        "delete_interaction",
        "cancel_interaction",
    }
)
_ASSISTANTS: Final = frozenset(
    {
        "acreate_assistants",
        "adelete_assistant",
        "aget_assistants",
        "acreate_thread",
        "aget_thread",
        "a_add_message",
        "aget_messages",
        "arun_thread",
    }
)
_MANAGED_AGENTS: Final[Mapping[str, str]] = MappingProxyType(
    {  # comprehension-ok: a flat walk of nested rows
        f"{prefix}{name}": f"interactions.agents.{prefix}{target}"
        for prefix in ("a", "")
        for name, target in (
            ("create_agent", "create"),
            ("list_agents", "list"),
            ("get_agent", "get"),
            ("delete_agent", "delete"),
            ("list_agent_versions", "list_versions"),
        )
    }
)
_SYNC_VECTOR_STORE_FILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        f"vector_store_file_{name}": f"vector_store_files.main.{target}"
        for name, target in (
            ("create", "create"),
            ("list", "list"),
            ("retrieve", "retrieve"),
            ("content", "retrieve_content"),
            ("update", "update"),
            ("delete", "delete"),
        )
    }
)

# The endpoints ``RustEndpoints`` serves with a method of its own, by the same name.
_DIRECT_ENDPOINTS: Final = frozenset(
    {
        "avector_store_create",
        "acreate_file",
        "acreate_batch",
        "acancel_batch",
        "aretrieve_batch",
        "alist_batches",
        "aadapter_completion",
        "abatch_completion",
        "abatch_completion_one_model_multiple_requests",
        "abatch_completion_fastest_response",
        "schedule_acompletion",
        "priority_header_update",
        "aguardrail",
        "asearch",
    }
)


def is_batch_upload(request: Mapping[str, object]) -> bool:
    return request.get("purpose") == "batch"


def endpoint_names() -> frozenset[str]:
    """Every endpoint name ``RustEndpoints.member`` may serve: what ``dir()`` of a router lists."""
    return frozenset(
        {
            *PLANS,
            *_HOST_GENERIC,
            *_VIDEO_DIRECT,
            *_SYNC_GENERIC,
            *_UNBRANCHED,
            *_ASSISTANTS,
            *_MANAGED_AGENTS,
            *_SYNC_VECTOR_STORE_FILES,
            *_DIRECT_ENDPOINTS,
            "create_interaction",
        }
    )


def plan_for(endpoint: str) -> Plan:
    plan: Final = generic_plan(endpoint)
    if plan is None:
        raise KeyError(endpoint)
    return plan


def generic_plan(endpoint: str) -> Plan | None:
    if endpoint in PLANS:
        return PLANS[endpoint]
    if endpoint in _HOST_GENERIC:
        return _generic(endpoint, _HOST_GENERIC[endpoint])
    if endpoint in _VIDEO_DIRECT:
        return _generic(endpoint, f"videos.{endpoint}")
    if endpoint in _SYNC_GENERIC:
        return _generic(
            endpoint,
            _SYNC_GENERIC[endpoint],
            sync=True,
            single_attempt=endpoint not in _ROUTER_RETRIED_SYNC_GENERIC,
        )
    return None


# --- preparing the request -------------------------------------------------------------------


def prepare_request(
    plan: Plan, request: Mapping[str, object], model: str, aliases: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the call's kwargs
    """What the router adds once before the first attempt (`_update_kwargs_before_fallbacks`): a
    trace id and the requested group, in the caller's own metadata dict."""
    prepared: Final = dict(request)
    if plan.kind == "vector_store_create":
        return prepared  # routed straight to a deployment, with no group recorded
    if plan.prepare_field is None:
        metadata_bucket(prepared, "metadata").update({"model_group": model})
        return prepared
    prepared.setdefault("litellm_trace_id", str(uuid.uuid4()))
    metadata_bucket(prepared, plan.prepare_field).update(
        {"model_group": model, "model_group_alias": model if model in aliases else None}
    )
    return prepared


# --- one attempt ----------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class AttemptCall:
    """One attempt's host call: the op, its kwargs, and the request kwargs the router kept (the
    ones its response headers are read from)."""

    host_op: str
    kwargs: Mapping[str, object]
    served: Mapping[str, object]
    model_name: str
    group: str
    # `_embedding`'s batch-size refusal, raised by the driver in place of the host call.
    refused: Exception | None = None


def reports_own_success(call: AttemptCall) -> bool:
    """A realtime attempt's return says nothing about the session (a dropped one returns too), so
    the host counts its success itself (`clean_realtime_session_deployment`)."""
    return call.host_op == "_arealtime"


def clean_realtime_session_deployment(call: AttemptCall) -> str | None:
    """`_ageneric_api_call_with_fallbacks`: the deployment a finished `_arealtime` session counts a
    success against, when its success handlers ran and it did not end abnormally."""
    from litellm.constants import REALTIME_SESSION_ENDED_ABNORMALLY_KEY, REALTIME_SESSION_SUCCESS_LOGGED_KEY
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    session_logging: Final = call.kwargs.get("litellm_logging_obj")
    if not reports_own_success(call) or not isinstance(session_logging, LiteLLMLogging):
        return None
    details: Final = session_logging.model_call_details
    if details.get(REALTIME_SESSION_SUCCESS_LOGGED_KEY) is not True:
        return None
    if details.get(REALTIME_SESSION_ENDED_ABNORMALLY_KEY) is True:
        return None
    deployment_id: Final = as_dict(call.kwargs.get("model_info")).get("id")
    return None if deployment_id is None else str(deployment_id)


# `_scheduler` once `del router.scheduler` ran (None is a scheduler not built yet).
_NO_SCHEDULER: Final = object()


class Counters:
    """`total_calls`, `success_calls` and `fail_calls`, keyed as each endpoint keys them."""

    __slots__ = ("fail_calls", "success_calls", "total_calls")

    def __init__(self) -> None:
        self.total_calls: defaultdict[str, int] = defaultdict(int)  # mutable-ok: the router's live counters
        self.success_calls: defaultdict[str, int] = defaultdict(int)  # mutable-ok: same
        self.fail_calls: defaultdict[str, int] = defaultdict(int)  # mutable-ok: same

    @staticmethod
    def _key(which: CountKey, model_name: str, group: str) -> str:
        return model_name if which == "deployment" else group if which == "group" else ""

    def started(self, plan: Plan, call: AttemptCall) -> None:
        if plan.total_key is not None:
            self.total_calls[self._key(plan.total_key, call.model_name, call.group)] += 1

    def succeeded(self, plan: Plan, call: AttemptCall) -> None:
        if plan.success_key is not None:
            self.success_calls[self._key(plan.success_key, call.model_name, call.group)] += 1

    def retract_success(self, plan: Plan, call: AttemptCall) -> None:
        """A served response the router then raised over: the attempt helper never reached its count."""
        if plan.success_key is not None:
            self.success_calls[self._key(plan.success_key, call.model_name, call.group)] -= 1

    def failed(self, plan: Plan, model_name: str | None, group: str) -> None:
        if plan.fail_key is None:
            return
        if plan.fail_key == "deployment" and model_name is None:
            return
        self.fail_calls[self._key(plan.fail_key, model_name or "", group)] += 1


def _compaction_surface(core: EndpointCore, plan: Plan) -> Surface | None:
    """``surface_for_call`` over the function the Python router's fallback loop is handed."""
    if plan.kind == "generic" and surface_for_call(_handler_name(plan.host_op)) is not None:
        return surface_for_call(invoked_handler_name(core, plan) or "")
    return "chat" if plan.kind == "chat" and not plan.sync else None


def arm_compaction_state(
    core: EndpointCore,
    plan: Plan,
    request: dict[str, object],  # mutable-ok: the request kwargs, written in place as the Python router writes them
) -> None:
    """``initialize_compaction_state`` at the top of ``async_function_with_fallbacks``."""
    surface: Final = _compaction_surface(core, plan)
    if surface is not None:
        request["_context_compaction_state"] = initialize_compaction_state(request, surface)  # rebind-ok: its own


async def compacted_attempt(
    core: EndpointCore, plan: Plan, deployment: Mapping[str, object], call: "AttemptCall"
) -> Mapping[str, object]:
    """The attempt's kwargs after ``compact_to_fit``, which ``_acompletion`` and the generic helper
    run on every attempt: it is also what refuses a compactor's own oversized child call."""
    if plan.kind != "generic" and (plan.kind != "chat" or plan.sync):
        return call.kwargs
    ctx: Final = RoutingContext(core.router)  # pyright: ignore[reportArgumentType]  # the checks read registry(), settings() and the token count
    return await compact_to_fit(ctx, deployment, call.kwargs, _compaction_surface(core, plan))


_VECTOR_STORE_SEARCH_OPS: Final = frozenset({"vector_stores.main.asearch", "vector_stores.main.search"})


def _deployment_params(
    plan: Plan, deployment: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the attempt's own copy of the deployment params
    params: Final = as_dict(deployment.get("litellm_params"))
    return as_dict(store_search_params(params)) if plan.host_op in _VECTOR_STORE_SEARCH_OPS else params


def _validate_decisions_connection(plan: Plan, data: Mapping[str, object], request: Mapping[str, object]) -> None:
    if _handler_name(plan.host_op) not in {"decisions", "adecisions", "systemone", "asystemone"}:
        return
    from litellm.decisions.call import validate_router_connection

    validate_router_connection(str(data.get("model")), _generic_provider(data), request)


def attempt_call(
    core: EndpointCore,
    plan: Plan,
    request: Mapping[str, object],
    deployment: Mapping[str, object],
    *,
    group: str,
    requested: str,
    fallback_level: int = 0,
) -> AttemptCall:
    """The host call for one attempt of ``plan`` against ``deployment``: the request (as the attempt
    helper receives it), shaped for the deployment and merged with its litellm params."""
    shaping: Final = core.shaping
    kwargs: Final = without(request, *ROUTING_KEYS)
    named: Final = {name: kwargs.pop(name) for name in plan.named if name in kwargs}
    data: Final = _deployment_params(plan, deployment)
    model_name: Final = str(data.get("model"))
    _validate_decisions_connection(plan, data, request)
    cache: Final = core.routing_cache
    caching: Final = shaping.cache_responses
    # A fallback hop routes the target group in a copy of the metadata (`run_async_fallback`).
    if fallback_level > 0:
        field: Final = plan.prepare_field or "metadata"
        kwargs[field] = {**as_dict(kwargs.get(field)), "model_group": requested}
    match plan.kind:
        case "chat":
            host = core.host  # rebind-ok: rebound as the Python router rebinds it
            drop_unsupported_classifier_reasoning_effort(deployment, requested, kwargs, host)
            params = (  # rebind-ok: rebound as the Python router rebinds it
                deployment_params_with_request_reasoning_override(data, kwargs)
            )
            silent_model = params.pop("silent_model", None)  # rebind-ok: rebound as the Python router rebinds it
            start_silent_experiments(core, plan, silent_model, named.get("messages"), kwargs)
            if "messages" in named:
                kwargs.setdefault("messages", named["messages"])
            update_kwargs_with_deployment(shaping, deployment, kwargs, cache=cache, splits_stream_timeout=True)
            kwargs.pop("silent_model", None)
            final = without_router_only_kwargs(  # rebind-ok: rebound as the Python router rebinds it
                {
                    **params,
                    **named,
                    "caching": caching,
                    "client": model_client(core, plan, deployment, kwargs),
                    **kwargs,
                }
            )
            return AttemptCall(plan.host_op, final, kwargs, str(params.get("model")), group)
        case "text" | "embedding" | "image" | "transcription" | "rerank" | "adapter":
            update_kwargs_with_deployment(
                shaping, deployment, kwargs, cache=cache, splits_stream_timeout=plan.splits_stream_timeout
            )
            refused: Final = embedding_batch_refusal(plan, deployment, named.get("input"), kwargs, requested)
            if refused is not None:
                return AttemptCall(plan.host_op, kwargs, kwargs, model_name, group, refused)
            final = {  # rebind-ok: rebound as the Python router rebinds it
                **data,
                **named,
                "caching": caching,
                "client": model_client(core, plan, deployment, kwargs),
                **kwargs,
            }
            return AttemptCall(plan.host_op, without_router_only_kwargs(final), kwargs, model_name, group)
        case "speech":
            update_kwargs_with_deployment(shaping, deployment, kwargs, cache=cache)
            voice = kwargs.pop("voice", None)  # rebind-ok: rebound as the Python router rebinds it
            final = {  # rebind-ok: rebound as the Python router rebinds it
                **data,
                **named,
                "voice": data.get("voice") if voice is None else voice,
                "client": model_client(core, plan, deployment, kwargs),
                **kwargs,
            }
            return AttemptCall(plan.host_op, final, kwargs, model_name, group)
        case "batch" | "cancel_batch" | "vector_store_create":
            if plan.kind == "cancel_batch":
                data.update(_resolved_credentials(core, deployment, requested))
                data.pop("litellm_credential_name", None)
            update_kwargs_with_deployment(shaping, deployment, kwargs, cache=cache, function_name=plan.function_name)
            final = {  # rebind-ok: rebound as the Python router rebinds it
                **data,
                "custom_llm_provider": _provider(core, data),
                "caching": caching,
                "client": model_client(core, plan, deployment, kwargs),
                **kwargs,
            }
            return AttemptCall(plan.host_op, final, kwargs, str(data.get("model")), group)
        case "generic":
            return _generic_attempt(core, plan, kwargs, deployment, data, requested, group)
        case "generic_sync":
            update_kwargs_with_deployment(
                shaping,
                deployment,
                kwargs,
                cache=cache,
                function_name="generic_api_call",
                splits_stream_timeout=invoked_handler_name(core, plan) in STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS,
            )
            endpoint = kwargs.get("endpoint")  # rebind-ok: rebound as the Python router rebinds it
            if endpoint and isinstance(endpoint, str):
                kwargs["endpoint"] = endpoint.replace(requested, model_name)
            kwargs["model"] = model_name
            final = {  # rebind-ok: rebound as the Python router rebinds it
                **data,
                "custom_llm_provider": _generic_provider(data),
                "caching": caching,
                **kwargs,
            }
            return AttemptCall(plan.host_op, final, kwargs, model_name, group)
        case "moderation":
            update_kwargs_with_deployment(shaping, deployment, kwargs, cache=cache)
            kwargs["model"] = model_name
            kwargs.update(data)
            return AttemptCall(plan.host_op, kwargs, kwargs, model_name, group)
        case "search":
            return _search_attempt(core, plan, kwargs, group)
        case "create_file":
            # `_acreate_file` creates the file on every healthy deployment of the hop's name, not
            # only the picked one: the driver hands the request to `create_file_everywhere`.
            return AttemptCall(plan.host_op, kwargs, kwargs, requested, group)


def _generic_attempt(
    core: EndpointCore,
    plan: Plan,
    kwargs: dict[str, object],  # mutable-ok: the attempt's own kwargs
    deployment: Mapping[str, object],
    data: dict[str, object],  # mutable-ok: the attempt's copy of the deployment params
    requested: str,
    group: str,
) -> AttemptCall:
    """`_ageneric_api_call_with_fallbacks_helper` for one deployment."""
    from litellm.llms.base_llm.passthrough.transformation import replace_path_segment
    from litellm.router_rust.invoke.websocket import turn_admission, turn_usage_recorder

    handler: Final = invoked_handler_name(core, plan)
    socket: Final = handler == "_aresponses_websocket"
    update_kwargs_with_deployment(
        core.shaping,
        deployment,
        kwargs,
        cache=core.routing_cache,
        function_name="_ageneric_api_call_with_fallbacks",
        splits_stream_timeout=handler in STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS,
        opens_responses_socket=socket,
    )
    model_name: Final = str(data.get("model"))
    endpoint: Final = kwargs.get("endpoint")
    if endpoint:
        try:
            stripped, _, _, _ = core.host.invoke("get_llm_provider")(
                model=model_name,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                api_base=kwargs.get("api_base"),
            )
            replacement = str(stripped)  # rebind-ok: rebound as the Python router rebinds it
        except Exception:  # noqa: BLE001  # as the Python router: an unresolvable model is used as is
            replacement = model_name  # rebind-ok: rebound as the Python router rebinds it
        kwargs["endpoint"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            replace_path_segment(str(endpoint), requested, replacement)
        )
    provider: Final = _generic_provider(data)
    final: Final = {
        **without_retrieve_stream_defaults(data, kwargs),
        "caching": core.shaping.cache_responses,
        **kwargs,
        "model": model_name,
        **_session_with_model(kwargs.get("session"), model_name),
        **(
            {
                "turn_admission": turn_admission(core, requested, deployment, kwargs),
                "turn_usage_recorder": turn_usage_recorder(core, requested, kwargs),
            }
            if socket
            else {}
        ),
    }
    if provider is not None:
        final["custom_llm_provider"] = provider
    return AttemptCall(plan.host_op, final, kwargs, model_name, group)


_SEARCH_PROTECTED_PARAMS: Final = frozenset({"search_provider", "api_key", "api_base"})


def _search_attempt(
    core: EndpointCore,
    plan: Plan,
    kwargs: dict[str, object],  # mutable-ok: the attempt's own kwargs
    group: str,
) -> AttemptCall:
    """One of the search tools named ``group`` at random, called with its provider, credentials and params under the request's."""
    import random

    tools: Final = [tool for tool in core.endpoints.search_tools if as_dict(tool).get("search_tool_name") == group]
    if not tools:
        raise ValueError(f"Search tool '{group}' not found in router.search_tools")
    selected: Final = random.choice(tools)  # load balancing, as the Python router picks
    params: Final = as_dict(as_dict(selected).get("litellm_params"))
    search_provider: Final = params.get("search_provider")
    if not search_provider:
        raise ValueError(f"search_provider not found in litellm_params for search tool '{group}'")
    search_params: Final = {
        key: value
        for source in (params, kwargs)
        for key, value in source.items()
        if key not in _SEARCH_PROTECTED_PARAMS and value is not None
    }  # comprehension-ok: the tool's params, then the request's
    final: Final = {
        "search_provider": search_provider,
        "api_key": params.get("api_key"),
        "api_base": params.get("api_base"),
        **search_params,
    }
    return AttemptCall(plan.host_op, final, kwargs, group, group)


def unrouted_fields(plan: Plan) -> Mapping[str, object]:
    """A search tool name is no model group: the Rust core hands every attempt to the host as is."""
    return {"unrouted": True} if plan.kind == "search" else {}


def failover_fields(plan: Plan) -> Mapping[str, object]:
    """An endpoint whose attempt stamps no ``failed_deployment_id`` never runs weighted failover."""
    return {} if plan.stamps_failed_deployment else {"skip_weighted_failover": True}


# The Python router's handler names for the plans whose `function_name` differs from it.
_PYTHON_HANDLER_NAMES: Final[Mapping[str, str]] = MappingProxyType({"acreate_file": "_acreate_file"})


def scoped_fallback_fields(plan: Plan, request: Mapping[str, object]) -> Mapping[str, object]:
    """A request that creates or names a resource only one provider holds (a file upload, a batch
    on an uploaded file, a fine-tuning job) falls back within its own model group only
    (``run_async_fallback``'s ``same_model_group_only``), decided by the Python router's own rules."""
    from litellm.router_utils.fallback_event_handlers import (
        creates_provider_scoped_resource,
        references_provider_scoped_resource,
    )

    function_name: Final = plan.function_name or plan.host_op
    handlers: Final = {
        "original_function": _Handler(_PYTHON_HANDLER_NAMES.get(function_name, function_name)),
        "original_generic_function": _Handler(plan.host_op),
    }
    kwargs: Final = {**request, **handlers}
    scoped: Final = references_provider_scoped_resource(kwargs) or creates_provider_scoped_resource(kwargs)
    return {"same_group_fallbacks_only": True} if scoped else {}


@dataclasses.dataclass(frozen=True, slots=True)
class _Handler:
    """A stand-in for a handler function: the scoped-resource rules read only its name."""

    __name__: str


def _session_with_model(session: object, model_name: str) -> Mapping[str, object]:
    if not isinstance(session, Mapping):
        return {}
    typed: Final = as_dict(session)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
    if "model" not in typed and typed.get("type") != "transcription":
        return {}
    from litellm.realtime_api.main import with_resolved_session_model

    return {"session": with_resolved_session_model(typed, model_name)}


def _generic_provider(data: Mapping[str, object]) -> str | None:
    from litellm.litellm_core_utils.routing_common_utils import provider_for_generic_call

    return provider_for_generic_call(data)  # pyright: ignore[reportArgumentType]  # deployment params


def model_client(
    core: EndpointCore, plan: Plan, deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> object:
    """The SDK client cached for the deployment (`_get_async_openai_model_client`; `_get_client`
    with the sync keys for sync completion and embedding), dropped when the request carries
    another api key."""
    from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs

    if as_dict(deployment.get("litellm_params")).get("kubernetes_pod_discovery") is True:
        return None
    model_id: Final = as_dict(deployment.get("model_info")).get("id")
    sync_keys: Final = plan.sync and plan.kind in ("chat", "embedding")
    stream: Final = "stream_" if kwargs.get("stream") is True else ""
    key: Final = f"{model_id}_{stream}client" if sync_keys else f"{model_id}_{stream}async_client"
    span: Final = get_parent_otel_span_from_kwargs(dict(kwargs))
    client: Final = _cached_client(core, key, span, local_only=not sync_keys)
    dynamic_api_key: Final = kwargs.get("api_key")
    if dynamic_api_key is not None and client is not None and dynamic_api_key != getattr(client, "api_key", None):
        return None
    return client


def _cached_client(core: EndpointCore, key: str, span: object, *, local_only: bool) -> object:
    cache: Final = core.routing_cache
    with service_target(ROUTER_CLIENTS_TARGET):
        if local_only:
            return cache.get_cache(key=key, local_only=True, parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]  # the cache holds untyped values
        return cache.get_cache(key=key, parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]  # see above


def _provider(core: EndpointCore, data: Mapping[str, object]) -> object:
    """The deployment's declared provider, else the one its model names (`get_llm_provider`)."""
    declared: Final = data.get("custom_llm_provider")
    _, inferred, _, _ = core.host.invoke("get_llm_provider")(model=data["model"], custom_llm_provider=declared)
    return declared or inferred


def _resolved_credentials(core: EndpointCore, deployment: Mapping[str, object], requested: str) -> Mapping[str, object]:
    """`get_deployment_credentials_with_provider` for the picked deployment (`_acancel_batch`)."""
    registry: Final = core.registry()
    model_id: Final = as_dict(deployment.get("model_info")).get("id")
    resolved: Final = registry.credentials(str(model_id or requested), form="with_provider")
    return as_dict(resolved)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values


def embedding_batch_refusal(
    plan: Plan, deployment: Mapping[str, object], input: object, kwargs: Mapping[str, object], model: str
) -> Exception | None:
    """``_embedding``'s check, past the deployment's kwargs shaping: an input the deployment admits
    drops the mark an earlier rejected attempt left on the shared logging object."""
    if plan.kind != "embedding":
        return None
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.router_utils.embedding_batch_size_check import (
        clear_rejected_before_provider_call,
        embedding_batch_too_large_error,
    )

    error: Final = embedding_batch_too_large_error(
        deployment=deployment,
        input=input,
        request_kwargs=kwargs,
        model=model,  # pyright: ignore[reportArgumentType]  # opaque values
    )
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if error is None and isinstance(logging_obj, LiteLLMLogging):
        clear_rejected_before_provider_call(logging_obj)
    return error


def log_refused_call_sync(call: AttemptCall, refused: Exception) -> None:
    """``_embedding`` logs its refusal before raising it: no provider call will."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.router_utils.embedding_batch_size_check import log_rejection_before_provider_call

    logging_obj: Final = call.kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, LiteLLMLogging):
        log_rejection_before_provider_call(logging_obj, refused)


async def log_refused_call(call: AttemptCall, refused: Exception) -> None:
    """``log_refused_call_sync`` for ``_aembedding``."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.router_utils.embedding_batch_size_check import async_log_rejection_before_provider_call

    logging_obj: Final = call.kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, LiteLLMLogging):
        await async_log_rejection_before_provider_call(logging_obj, refused)


def stamp_error(
    plan: Plan, error: BaseException, deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> None:
    """What each attempt helper stamps on the error it raises."""
    match plan.stamp:
        case "deployment":
            stamp_deployment_num_retries(error, deployment)
            stamp_failed_deployment_id(error, kwargs.get("model_info") or deployment.get("model_info"))
        case "retry_skip":
            stamp_retry_skip_deployment_id(error, kwargs)
        case "none":
            pass
    if plan.stamp != "deployment" and plan.stamps_failed_deployment:
        # `_stamp_failed_attempt`: the deployment `_update_kwargs_with_deployment` put on the kwargs.
        stamp_failed_deployment_id(error, kwargs.get("model_info"))


# --- silent experiments ---------------------------------------------------------------------


_SilentModel: TypeAlias = str | list[str]
_SILENT_MODEL_ADAPTER: Final[TypeAdapter[_SilentModel]] = TypeAdapter(_SilentModel)


def _silent_targets(silent_model: object) -> tuple[str, ...]:
    """`_silent_experiment_targets`: none inside a mirrored call or for a malformed value."""
    from litellm._internal_context import in_mirrored_call

    if silent_model is None or in_mirrored_call():
        return ()
    try:
        targets: Final = _SILENT_MODEL_ADAPTER.validate_python(silent_model)
    except ValidationError:
        verbose_router_logger.warning(
            "silent_model must be a model name or a list of model names, got %r; skipping shadow traffic",
            silent_model,
        )
        return ()
    return (targets,) if isinstance(targets, str) else tuple(targets)


def _silent_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: the experiment's own kwargs
    """`_get_silent_experiment_kwargs`."""
    from litellm.litellm_core_utils.core_helpers import (
        safe_deep_copy,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    )

    silent: Final = safe_deep_copy(dict(kwargs))  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    original_metadata: Final = kwargs.get("metadata")
    if original_metadata is not None and silent.get("metadata") is original_metadata:  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
        silent["metadata"] = dict(original_metadata)  # pyright: ignore[reportArgumentType, reportCallIssue]  # the request's metadata dict
    if "metadata" not in silent:
        silent["metadata"] = {}
    metadata: Final = silent["metadata"]  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    if isinstance(metadata, dict):
        metadata.pop("litellm_parent_otel_span", None)  # pyright: ignore[reportUnknownMemberType]  # the request's metadata dict
        metadata["is_silent_experiment"] = True
    silent.pop("litellm_call_id", None)  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
    silent.pop("litellm_logging_obj", None)  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
    silent.pop("standard_logging_object", None)  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
    return silent  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values


def _snapshot(kwargs: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: a snapshot handed to a background task
    snapshot: Final = dict(kwargs)
    metadata: Final = snapshot.get("metadata")
    if isinstance(metadata, Mapping):
        snapshot["metadata"] = dict(as_dict(metadata))  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
    return snapshot


async def _run_silent(core: EndpointCore, silent_model: str, messages: object, kwargs: Mapping[str, object]) -> None:
    from litellm._internal_context import mirrored_call

    remaining: Final = without(kwargs, "stream")
    with mirrored_call():
        response: Final = await core.acompletion(
            model=silent_model, messages=messages, stream=bool(kwargs.get("stream", False)), **remaining
        )
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        if not isinstance(response, CustomStreamWrapper):
            return
        async for _ in response:
            pass


async def _silent_async(core: EndpointCore, silent_model: str, messages: object, kwargs: Mapping[str, object]) -> None:
    try:
        if as_dict(kwargs.get("metadata")).get("is_silent_experiment", False):
            return
        silent: Final = _silent_kwargs(kwargs)
        as_dict_metadata: Final = silent["metadata"]
        if isinstance(as_dict_metadata, dict):
            as_dict_metadata["model_group"] = silent_model
        await _run_silent(core, silent_model, copy.deepcopy(messages), silent)
    except Exception:  # noqa: BLE001, S110  # a silent experiment never fails the primary call
        pass


def _silent_thread(core: EndpointCore, silent_model: str, messages: object, kwargs: Mapping[str, object]) -> None:
    try:
        if as_dict(kwargs.get("metadata")).get("is_silent_experiment", False):
            return
        silent: Final = _silent_kwargs(kwargs)
        metadata: Final = silent["metadata"]
        if isinstance(metadata, dict):
            metadata["model_group"] = silent_model
        loop: Final = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:

            async def run() -> None:
                await _run_silent(core, silent_model, copy.deepcopy(messages), silent)
                pending: Final = asyncio.all_tasks()
                current: Final = asyncio.current_task()
                if current is not None:
                    pending.discard(current)
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

            loop.run_until_complete(run())
        finally:
            loop.close()
    except Exception:  # noqa: BLE001, S110  # a silent experiment never fails the primary call
        pass


def start_silent_experiments(
    core: EndpointCore, plan: Plan, silent_model: object, messages: object, kwargs: Mapping[str, object]
) -> None:
    """Mirror the call to each silent model in the background: a task on the async path, a daemon
    thread on the sync one."""
    for target in _silent_targets(silent_model):
        if plan.sync:
            threading.Thread(
                target=_silent_thread, args=(core, target, messages, _snapshot(kwargs)), daemon=True
            ).start()
        else:
            _ = asyncio.create_task(
                _silent_async(core, target, messages, _snapshot(kwargs))
            )  # fire and forget, as the Python router


# --- the endpoints ----------------------------------------------------------------------------

_KEEPS_CLIENT: Final = frozenset(
    {
        "acreate_realtime_client_secret",
        "arealtime_calls",
        "acreate_realtime_transcription_session",
        "acreate_realtime_translation_client_secret",
        "arealtime_translation_calls",
    }
)
_KEEPS_PROVIDER_AND_CLIENT: Final = frozenset({"afile_delete", "afile_content"})
_RESPONSE_ID_ENDPOINTS: Final = frozenset(
    {"aget_responses", "acancel_responses", "acompact_responses", "adelete_responses", "alist_input_items"}
)
_CONTAINER_ENDPOINTS: Final = frozenset(
    {"acreate_container", "alist_containers", "aretrieve_container", "adelete_container", *_CONTAINER_FILE_ROUTED}
)
_VECTOR_STORE_ENDPOINTS: Final = frozenset(
    {
        *(f"avector_store_{name}" for name in ("search", "create", "retrieve", "list", "update", "delete")),
        *(f"avector_store_file_{name}" for name in ("create", "list", "retrieve", "content", "update", "delete")),
    }
)
_INTERACTION_ENDPOINTS: Final = frozenset(
    {"acreate_interaction", "aget_interaction", "adelete_interaction", "acancel_interaction"}
)
_MISSING: Final = object()


def pinned_video_deployment(plan: Plan, request: Mapping[str, object]) -> Mapping[str, object]:
    """`_video_call_pin`: the deployment a video id names, for a call on that video only."""
    from litellm.types.videos.utils import VIDEO_ID_CALL_TYPES, decode_video_id_with_provider

    video_id: Final = request.get("video_id")
    if plan.endpoint not in VIDEO_ID_CALL_TYPES or not isinstance(video_id, str):
        return {}
    deployment_id: Final = decode_video_id_with_provider(video_id).get("deployment_id")
    return {"pinned_deployment_id": deployment_id} if deployment_id else {}


def pinned_vector_store_file_deployment(plan: Plan, request: Mapping[str, object]) -> Mapping[str, object]:
    deployment_id: Final = request.get(PINNED_DEPLOYMENT_KEY)
    if not plan.endpoint.startswith("avector_store_file_") or not isinstance(deployment_id, str) or not deployment_id:
        return {}
    return {"pinned_deployment_id": deployment_id}


async def _await(value: object) -> object:
    """``await value``, raising the TypeError Python raises for a value that is not awaitable."""
    return await value  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # deliberately awaited as the Python router does


def _missing_model() -> TypeError:
    return TypeError("_ageneric_api_call_with_fallbacks() missing 1 required positional argument: 'model'")


class RustEndpoints:
    """The public endpoint methods of one Rust core, beyond the routed call loop the core drives."""

    def __init__(self, core: EndpointCore, config: "RouterConfig") -> None:
        from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
        from litellm.router_utils.request_priority import InvalidPriority, parse_default_priority

        self._core: Final = core
        self.counters: Final = Counters()
        explicit: Final = config.get("timeout")
        self._explicit_timeout: Final = explicit
        self._request_timeout: Final = get_configured_request_timeout() if explicit is not None else None
        self._stream_timeout: Final = config.get("stream_timeout")
        # apply() flips it after construction, as on the Python core
        self._cache_responses: bool = bool(config.get("cache_responses"))
        self._assistants_config: Final = (
            as_dict(config.get("assistants_config")) if config.get("assistants_config") else None
        )
        # These three are assigned by apply() after construction, as on the Python core.
        tools: Final = list(_as_items(config.get("search_tools")))
        self._search_tools: list[object] = tools  # mutable-ok: the router's live list
        self._guardrails: list[object] = list(_as_items(config.get("guardrail_list")))  # mutable-ok: same
        parsed: Final = parse_default_priority(config.get("default_priority"))
        if isinstance(parsed, InvalidPriority):
            from litellm._logging import verbose_router_logger

            verbose_router_logger.warning(
                "router_settings.default_priority is ignored, no request is queued by default: %s",
                parsed.message,
            )
        # apply() assigns it after construction, kept as given, as the Python core's attribute is.
        self._default_priority: int | None = None if isinstance(parsed, InvalidPriority) else parsed
        self._polling_interval: Final = config.get("polling_interval")
        self._caching_groups: Final = config.get("caching_groups")
        self._default_max_parallel_requests: object = config.get("default_max_parallel_requests")
        self._scheduler: object = None

    @property
    def cache_responses(self) -> bool:
        return self._cache_responses

    def set_cache_responses(self, value: bool) -> None:
        """``cache_responses`` assigned after construction (the proxy turns it on once its cache
        exists): later calls pass ``caching=value``, as the Python core's flag does."""
        self._cache_responses = value

    def assign_setting(self, name: str, value: object) -> None:
        """``search_tools``, ``guardrail_list``, ``default_max_parallel_requests`` or
        ``default_priority`` assigned after construction, kept as given: later searches, guardrail
        picks, slot limits and queued calls read it, as the Python core's attributes are read."""
        if name == "search_tools":
            self._search_tools = value  # pyright: ignore[reportAttributeAccessIssue]  # stored as given, as the Python core stores it
        elif name == "guardrail_list":
            self._guardrails = value  # pyright: ignore[reportAttributeAccessIssue]  # same
        elif name == "default_max_parallel_requests":
            self._default_max_parallel_requests = value
        elif name == "default_priority":
            self._default_priority = value  # pyright: ignore[reportAttributeAccessIssue]  # stored as given, as the Python core stores it
        else:
            raise ValueError(f"not an endpoint setting: {name}")

    @property
    def default_max_parallel_requests(self) -> object:
        """The setting as given: ``settings()`` reports it and the slot limits read it."""
        return self._default_max_parallel_requests

    @property
    def search_tools(self) -> list[object]:  # mutable-ok: the router's live list, as the Python router reports it
        return self._search_tools

    @property
    def guardrail_list(self) -> list[object]:  # mutable-ok: same
        return self._guardrails

    @staticmethod
    def prepare_host(config: "RouterConfig", host: "PythonHost") -> None:
        """`cache_responses` sets the process cache, unless one is set already (the proxy's)."""
        if not config.get("cache_responses") or host.global_cache() is not None:
            return
        from litellm.caching.caching import Cache

        redis_url: Final = config.get("redis_url")
        redis_host: Final = config.get("redis_host")
        redis_port: Final = config.get("redis_port")
        cache_config: Final[dict[str, object]] = {}  # mutable-ok: the request kwargs the Python router mutates in place
        cache_type: str = "local"  # rebind-ok: redis when the router is given one
        if redis_url is not None or (redis_host is not None and redis_port is not None):
            cache_type = "redis"  # rebind-ok: rebound as the Python router rebinds it
            if redis_url is not None:
                cache_config["url"] = redis_url
            if redis_host is not None:
                cache_config["host"] = redis_host
            if redis_port is not None:
                cache_config["port"] = str(redis_port)
            if config.get("redis_password") is not None:
                cache_config["password"] = config.get("redis_password")
            if config.get("redis_db") is not None:
                cache_config["db"] = str(config.get("redis_db"))
            cache_config.update(as_dict(config.get("cache_kwargs")))
        host.set_global_cache(Cache(type=cache_type, **cache_config))  # pyright: ignore[reportArgumentType]  # the Python router's cache config

    def shaping(self, resolved: "Resolved") -> Shaping:
        return Shaping(
            default_litellm_params=self.default_litellm_params(resolved),
            timeout=resolved.timeout,
            explicit_timeout=self._explicit_timeout,
            request_timeout=self._request_timeout,
            stream_timeout=self._stream_timeout,
            cache_responses=self._cache_responses,
        )

    def default_litellm_params(self, resolved: "Resolved") -> Mapping[str, object]:
        """The defaults the Python constructor completes before any call: the router ``timeout``
        argument, no client retries, and the caching groups in the default metadata."""
        configured: Final = resolved.default_litellm_params
        metadata: Final = configured.get("metadata")
        return MappingProxyType(
            {
                "timeout": self._explicit_timeout,
                "max_retries": 0,
                **configured,
                "metadata": {
                    **(as_dict(metadata) if isinstance(metadata, Mapping) else {}),  # pyright: ignore[reportUnknownArgumentType]  # the configured metadata
                    "caching_groups": self._caching_groups,
                },
            }
        )

    def prepare(
        self, plan: Plan, request: Mapping[str, object], model: str
    ) -> dict[str, object]:  # mutable-ok: the call's kwargs
        return prepare_request(plan, request, model, self._core.resolved.model_group_alias)

    # -- host and registry ---------------------------------------------------------------------

    def _invoke(self, op: str) -> Callable[..., object]:
        return self._core.host.invoke(op)

    async def _direct(self, op: str, kwargs: Mapping[str, object]) -> object:
        return await _await(self._invoke(op)(**kwargs))

    def _group(self, name: str) -> list[Mapping[str, object]]:  # mutable-ok: a fresh list
        """`_core_get_model_list(model_name=name)`: the deployments of the group ``name``."""
        records: Final = self._core.handle.deployments()
        return [record for record in (as_dict(r) for r in records) if record.get("model_name") == name]

    def _is_model_group(self, name: str) -> bool:
        """`_core_get_model_list(model_name=name)`: an alias, a routing group or a wildcard counts."""
        return bool(self._core.group_deployments(name))

    # -- members ------------------------------------------------------------------------------

    def member(self, name: str) -> Callable[..., object] | None:
        """The public endpoint ``name``, bound to this core, or None when the core has none."""
        if name in _UNBRANCHED:
            return _answers_none
        if name in _ASSISTANTS:
            return self._assistants(name)
        if name in _MANAGED_AGENTS:
            return self._managed_agents(name)
        if name in _SYNC_VECTOR_STORE_FILES:
            return self._sync_vector_store_files(name)
        if name == "create_interaction":
            return self._factory("create_interaction", "interactions.create")
        if name in _DIRECT_ENDPOINTS:
            direct: Final[Callable[..., object]] = getattr(self, name)  # pyright: ignore[reportAny]  # one of this class's own endpoint methods
            return direct
        if name == "responses":
            return self._sync_factory(name, PLANS["responses"])
        plan: Final = generic_plan(name)
        if plan is None or name in PLANS:
            return None
        if plan.sync:
            return self._sync_factory(name, plan)
        return self._factory(name, plan.host_op)

    # -- the factory wrappers ------------------------------------------------------------------

    def _factory(self, name: str, host_op: str) -> Callable[..., Awaitable[object]]:
        async def async_wrapper(
            custom_llm_provider: str | None = None,
            client: object = None,
            **kwargs: object,  # kwargs-ok: the endpoint's request
        ) -> object:
            return await self._wrapped(name, host_op, custom_llm_provider, client, dict(kwargs))

        async_wrapper.__name__ = name
        return async_wrapper

    async def _wrapped(
        self,
        name: str,
        host_op: str,
        custom_llm_provider: str | None,
        client: object,
        kwargs: dict[str, object],  # mutable-ok: the wrapper's own kwargs
    ) -> object:
        listed: Final = generic_plan(name)
        # The endpoint's own plan, unless the wrapper calls another litellm function for it
        # (``avector_store_create`` without a model calls ``vector_stores.main.acreate``).
        plan: Final = listed if listed is not None and listed.host_op == host_op else _generic(name, host_op)
        if name in _CONTAINER_ENDPOINTS:
            return await self._containers(plan, custom_llm_provider, kwargs)
        if name in _RESPONSE_ID_ENDPOINTS:
            from litellm.responses.utils import ResponsesAPIRequestUtils

            model_id: Final = ResponsesAPIRequestUtils.get_model_id_from_response_id(kwargs.get("response_id"))  # pyright: ignore[reportArgumentType]  # any id the caller sends
            if model_id is not None:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "model"
                ] = model_id
            return await self._ageneric(plan, kwargs)
        if name in _VECTOR_STORE_ENDPOINTS:
            return await self._vector_stores(name, plan, custom_llm_provider, kwargs)
        if name in _INTERACTION_ENDPOINTS or name == "create_interaction":
            return await self._interactions(plan, custom_llm_provider, kwargs)
        if name in _KEEPS_CLIENT:
            return await self._ageneric(plan, {**kwargs, "client": client})
        if name in _KEEPS_PROVIDER_AND_CLIENT:
            return await self._ageneric(plan, {**kwargs, "custom_llm_provider": custom_llm_provider, "client": client})
        if name == "allm_passthrough_route":
            if client:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "client"
                ] = client
            return await self._ageneric(plan, kwargs, passthrough=True)
        return await self._ageneric(plan, kwargs)

    async def generic(self, plan: Plan, request: Mapping[str, object]) -> object:
        """A dispatched generic endpoint called with its whole request (the wrapper's
        ``custom_llm_provider`` and ``client`` are consumed by the wrapper)."""
        kwargs: Final = without(request, "custom_llm_provider", "client")
        return await self._ageneric(plan, kwargs)

    async def _ageneric(
        self,
        plan: Plan,
        kwargs: Mapping[str, object],
        *,
        passthrough: bool = False,
    ) -> object:
        """`_ageneric_api_call_with_fallbacks`."""
        if "model" not in kwargs:
            raise _missing_model()

        async def passthrough_call(prepared: Mapping[str, object], name: str) -> object:
            sent: Final = without(prepared, *ROUTING_KEYS)
            return await self._direct(plan.host_op, {"model": name, **sent})

        return await self._core.drive(plan, kwargs, passthrough=passthrough_call if passthrough else None)

    def _sync_factory(self, name: str, plan: Plan) -> Callable[..., object]:
        is_vector_store: Final = name.startswith("vector_store_")
        attempts: Final = _SINGLE_ATTEMPT if plan.single_attempt else None

        def sync_wrapper(
            custom_llm_provider: str | None = None,
            client: object = None,
            **kwargs: object,  # kwargs-ok: the endpoint's request
        ) -> object:
            if not is_vector_store:
                if "model" not in kwargs:
                    raise TypeError(
                        "_generic_api_call_with_fallbacks() missing 1 required positional argument: 'model'"
                    )
                return self._core.call_sync(plan, kwargs, attempts)
            provider_kwargs: dict[  # mutable-ok: the request kwargs the Python router mutates in place  # rebind-ok: rebound as the Python router rebinds it
                str, object
            ] = (
                {**kwargs, "custom_llm_provider": custom_llm_provider}
                if custom_llm_provider and "custom_llm_provider" not in kwargs
                else dict(kwargs)
            )
            if name == "vector_store_search":
                provider_kwargs = {  # rebind-ok: rebound as the Python router rebinds it
                    **provider_kwargs,
                    "_direct_vector_store_embedding_executor": self._embedding_executor(kwargs),
                }
            if provider_kwargs.get("model"):
                return self._core.call_sync(plan, provider_kwargs, attempts)
            if name == "vector_store_search":
                return self._invoke(plan.host_op)(**{**provider_kwargs, "router": self._core.public})
            return self._invoke(plan.host_op)(**provider_kwargs)

        sync_wrapper.__name__ = name
        return sync_wrapper

    def _embedding_executor(self, kwargs: Mapping[str, object]) -> object:
        from litellm.llms.base_llm.vector_store.transformation import (
            RouterVectorStoreEmbeddingExecutor,
            vector_store_request_metadata,
        )

        return RouterVectorStoreEmbeddingExecutor(
            router=self._core.router,  # pyright: ignore[reportArgumentType]  # the router serves aembedding
            metadata=vector_store_request_metadata(kwargs),
        )

    async def _containers(
        self,
        plan: Plan,
        custom_llm_provider: str | None,
        kwargs: dict[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    ) -> object:
        """`_init_containers_api_endpoints`."""
        from litellm.responses.utils import ResponsesAPIRequestUtils

        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                "custom_llm_provider"
            ] = custom_llm_provider
        container_id: Final = kwargs.get("container_id")
        forwarded: Final = kwargs.get("model_id")
        if isinstance(container_id, str):
            decoded: Final = ResponsesAPIRequestUtils.decode_container_id(container_id)
            original_id: Final = decoded.get("response_id", container_id)
            if original_id != container_id:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "container_id"
                ] = original_id
            decoded_provider: Final = decoded.get("custom_llm_provider")
            if decoded_provider and kwargs.get("custom_llm_provider") == "openai":
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "custom_llm_provider"
                ] = decoded_provider
            model_id: Final = decoded.get("model_id") or (
                forwarded.strip() if isinstance(forwarded, str) and forwarded.strip() else None
            )
            if model_id:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "model"
                ] = model_id
                return await self._ageneric(plan, kwargs)
        requested: Final = kwargs.get("model")
        if isinstance(requested, str) and requested.strip():
            return await self._ageneric(plan, kwargs, passthrough=True)
        return await self._direct(plan.host_op, kwargs)

    async def _vector_stores(
        self,
        name: str,
        plan: Plan,
        custom_llm_provider: str | None,
        kwargs: dict[str, object],  # mutable-ok: see _wrapped
    ) -> object:
        """`_init_vector_store_api_endpoints`."""
        if name == "avector_store_search":
            kwargs = {  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                **kwargs,
                "_direct_vector_store_embedding_executor": self._embedding_executor(kwargs),
            }
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                "custom_llm_provider"
            ] = custom_llm_provider
        if kwargs.get("model"):
            return await self._ageneric(plan, kwargs)
        if name == "avector_store_search":
            return await self._direct(plan.host_op, {**kwargs, "router": self._core.public})
        return await self._direct(plan.host_op, kwargs)

    async def _interactions(
        self,
        plan: Plan,
        custom_llm_provider: str | None,
        kwargs: dict[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    ) -> object:
        """`_init_interactions_api_endpoints`."""
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                "custom_llm_provider"
            ] = custom_llm_provider
        if "custom_llm_provider" not in kwargs:
            kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                "custom_llm_provider"
            ] = "gemini"
        if kwargs.get("agent") and kwargs.get("model") == kwargs.get("agent"):
            kwargs["model"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                None
            )
        if kwargs.get("model"):
            return await self._ageneric(plan, kwargs)
        return await self._direct(plan.host_op, kwargs)

    def _managed_agents(self, name: str) -> Callable[..., object]:
        op: Final = _MANAGED_AGENTS[name]

        def prepared(
            custom_llm_provider: str | None,
            kwargs: dict[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
        ) -> dict[str, object]:  # mutable-ok: the call's kwargs
            if custom_llm_provider and "custom_llm_provider" not in kwargs:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "custom_llm_provider"
                ] = custom_llm_provider
            if "custom_llm_provider" not in kwargs:
                kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                    "custom_llm_provider"
                ] = "gemini"
            return kwargs

        if name.startswith("a"):

            async def managed_agents_wrapper(
                custom_llm_provider: str | None = None,
                client: object = None,
                **kwargs: object,  # kwargs-ok: the request
            ) -> object:
                return await self._direct(op, prepared(custom_llm_provider, dict(kwargs)))

            return managed_agents_wrapper

        def managed_agents_sync_wrapper(
            custom_llm_provider: str | None = None,
            client: object = None,
            **kwargs: object,  # kwargs-ok: the request
        ) -> object:
            return self._invoke(op)(**prepared(custom_llm_provider, dict(kwargs)))

        return managed_agents_sync_wrapper

    def _sync_vector_store_files(self, name: str) -> Callable[..., object]:
        op: Final = _SYNC_VECTOR_STORE_FILES[name]

        def vector_store_file_sync_wrapper(
            custom_llm_provider: str | None = None,
            client: object = None,
            **kwargs: object,  # kwargs-ok: the request
        ) -> object:
            return self._invoke(op)(custom_llm_provider=custom_llm_provider, client=client, **kwargs)

        return vector_store_file_sync_wrapper

    def _assistants(self, name: str) -> Callable[..., Awaitable[object]]:
        async def assistants_wrapper(
            custom_llm_provider: str | None = None,
            client: object = None,
            **kwargs: object,  # kwargs-ok: the request
        ) -> object:
            """`_pass_through_assistants_endpoint_factory`."""
            provider: object = custom_llm_provider  # rebind-ok: the assistants config's when none is given
            sent: Final = dict(kwargs)
            if provider is None:
                if self._assistants_config is None:
                    raise Exception(  # noqa: TRY002  # the Python router raises a bare Exception
                        "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                    )
                provider = self._assistants_config[  # rebind-ok: rebound as the Python router rebinds it
                    "custom_llm_provider"
                ]
                sent.update(as_dict(self._assistants_config.get("litellm_params")))
            return await self._direct(name, {"custom_llm_provider": provider, "client": client, **sent})

        return assistants_wrapper

    # -- completions, priority and prompt management ----------------------------------------

    def _drops_params(self, model: str, kwargs: Mapping[str, object]) -> bool:
        from litellm.litellm_core_utils.routing_common_utils import get_request_team_id
        from litellm.router_utils.request_priority import request_drops_params

        def deployment_params() -> tuple[Mapping[str, object], ...]:
            names: Final = {as_dict(r).get("model_name") for r in self._core.handle.deployments()}
            pinned: Final = (
                self._core.handle.deployment(model) if model not in names and isinstance(model, str) else None  # pyright: ignore[reportUnnecessaryIsInstance]  # a caller's value, typed for the common case
            )
            if pinned is not None:
                return (as_dict(as_dict(pinned).get("litellm_params")),)
            if kwargs.get("specific_deployment") is True:
                candidates = [  # rebind-ok: rebound as the Python router rebinds it
                    as_dict(r)
                    for r in self._core.handle.deployments()
                    if as_dict(as_dict(r).get("litellm_params")).get("model") == model
                ]
            else:
                # `_core_get_model_list`: an alias's target, a routing group's members and a
                # wildcard's deployments list here too.
                candidates = [  # rebind-ok: rebound as the Python router rebinds it
                    as_dict(record)
                    for record in self._core.registry().deployments(model, team_id=get_request_team_id(kwargs)) or ()
                ]
            return tuple(as_dict(candidate.get("litellm_params")) for candidate in candidates)

        return request_drops_params(
            kwargs,
            self._core.resolved.default_litellm_params,
            deployment_params(),
        )

    def _split_priority(
        self, model: str, kwargs: Mapping[str, object]
    ) -> tuple[int | None, dict[str, object]]:  # mutable-ok: the call's kwargs
        """`_core_split_request_priority`."""
        from litellm.router_utils.request_priority import InvalidPriority, priority_error, resolve_request_priority

        priority: Final = resolve_request_priority(
            requested=kwargs.get("priority"),
            default_priority=self._default_priority,
            drops_params=lambda: self._drops_params(model=model, kwargs=kwargs),
        )
        if isinstance(priority, InvalidPriority):
            raise priority_error(model=model, message=priority.message)
        return priority, without(kwargs, "priority")

    def priority_header_update(
        self, model: str, kwargs: Mapping[str, object], header: str | None
    ) -> Mapping[str, int | str]:
        from litellm.router_utils.request_priority import priority_header_fields

        return priority_header_fields(header, drops_params=lambda: self._drops_params(model=model, kwargs=kwargs))

    def set_scheduler(self, scheduler: object) -> None:
        """``router.scheduler = ...``, which the next prioritized call waits on."""
        self._scheduler = scheduler

    def delete_scheduler(self) -> None:
        """``del router.scheduler``: a prioritized call raises from then on, as on the Python core."""
        self._scheduler = _NO_SCHEDULER

    def _get_scheduler(self) -> object:
        if self._scheduler is _NO_SCHEDULER:
            raise AttributeError("'RustCore' object has no attribute 'scheduler'", name="scheduler")
        if self._scheduler is None:
            from litellm.scheduler import Scheduler

            self._scheduler = Scheduler(
                polling_interval=self._polling_interval,  # pyright: ignore[reportArgumentType]  # as configured
                redis_cache=self._core.routing_cache.redis_cache,
            )
        return self._scheduler

    async def _schedule(
        self, model: str, priority: int, run: Callable[[], Awaitable[object]], request: Mapping[str, object]
    ) -> object:
        """`_schedule_factory`: wait for the scheduler's turn, run, mark the response."""
        from litellm.litellm_core_utils.hidden_params import get_hidden_params
        from litellm.scheduler import FlowItem, Scheduler

        scheduler: Final = self._get_scheduler()
        assert isinstance(scheduler, Scheduler)

        async def healthy_deployments() -> Sequence[object]:
            try:
                healthy: Final = await self._core.select(model, dict(request), kind="healthy")
            except Exception:  # noqa: BLE001  # `_async_get_healthy_deployments` lists none on any lookup failure
                # It lists none while every deployment cools down, or for a group it cannot look up:
                # the request keeps its place in the queue, and the call itself raises the error.
                return []
            return list(healthy) if isinstance(healthy, Sequence) else [healthy]

        await scheduler.wait_for_turn(
            request=FlowItem(priority=priority, request_id=str(uuid.uuid4()), model_name=model),
            timeout=self._core.resolved.timeout,  # pyright: ignore[reportArgumentType]  # the router timeout
            get_healthy_deployments=healthy_deployments,
        )
        try:
            response: Final = await run()
        except Exception as error:
            setattr(error, "priority", priority)  # noqa: B010  # the Python router stamps the same attribute
            raise
        hidden: Final = get_hidden_params(response)
        if hidden is not None:
            headers: Final = hidden.setdefault("additional_headers", {})
            if isinstance(headers, dict):
                headers.update({"x-litellm-request-prioritization-used": True})  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
        return response

    async def acompletion(self, request: Mapping[str, object]) -> object:
        model: Final = request.get("model")
        if model is None and "model" in request:
            return await self._core.drive(PLANS["acompletion"], request)
        if "model" not in request:
            raise TypeError("acompletion() needs a model name")
        if not isinstance(model, str):
            _ = self._core.routed_name(model)
        kwargs: Final = {**request, "stream": request.get("stream", False)}
        start_time: Final = time.time()
        response: Final = await self._routed_acompletion(model, kwargs)  # pyright: ignore[reportArgumentType]  # the name as given
        self._core.service_logged("acompletion", start_time, time.time(), kwargs)
        return response

    async def _routed_acompletion(self, model: str, kwargs: Mapping[str, object]) -> object:
        plan: Final = PLANS["acompletion"]
        priority, rest = self._split_priority(model, kwargs)
        from litellm.router_rust.invoke.prompt_management import (
            is_prompt_management_model,
            prompt_management_acompletion,
        )

        if is_prompt_management_model(self._core, model):
            prepared: Final = self.prepare(plan, rest, model)
            messages: Final = rest.get("messages")

            def factory() -> Awaitable[object]:
                return prompt_management_acompletion(self._core, model, messages, prepared)

            if priority is None:
                return await factory()
            return await self._schedule(model, priority, factory, prepared)
        if priority is None:
            return await self._core.drive(plan, rest)
        return await self._schedule(model, priority, lambda: self._core.drive(plan, rest), rest)

    async def atext_completion(self, request: Mapping[str, object]) -> object:
        model: Final = request.get("model")
        if model is None and "model" in request:
            return await self._core.drive(PLANS["atext_completion"], request)
        if "model" not in request:
            raise TypeError("atext_completion() needs a model name")
        if not isinstance(model, str):
            _ = self._core.routed_name(model)
        kwargs: Final = without(request, "is_retry", "is_fallback", "is_async")
        plan: Final = PLANS["atext_completion"]
        priority, rest = self._split_priority(model, kwargs)  # pyright: ignore[reportArgumentType]  # the name as given
        if priority is None:
            return await self._core.drive(plan, rest)
        return await self._schedule(model, priority, lambda: self._core.drive(plan, rest), rest)  # pyright: ignore[reportArgumentType]  # see above

    async def schedule_acompletion(
        self,
        model: str,
        messages: object,
        priority: int | None = None,
        stream: bool = False,
        **kwargs: object,  # kwargs-ok: the request
    ) -> object:
        from litellm.router_utils.request_priority import priority_error, unqueueable_priority_message

        queued, _ = self._split_priority(model, {**kwargs, "priority": priority})
        if queued is None:
            raise priority_error(model=model, message=unqueueable_priority_message(priority))
        return await self._core.acompletion(model=model, messages=messages, stream=stream, priority=queued, **kwargs)

    # -- batch completions -------------------------------------------------------------------

    async def _no_exceptions(self, **kwargs: object) -> object:  # kwargs-ok: the request
        try:
            return await self._core.acompletion(**kwargs)
        except Exception as error:  # noqa: BLE001  # returned in place, as the Python router does
            return error

    async def abatch_completion(
        self,
        models: list[str],  # mutable-ok: the request kwargs the Python router mutates in place
        messages: object,
        **kwargs: object,  # kwargs-ok: the endpoint's open request
    ) -> object:
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):  # pyright: ignore[reportUnknownVariableType]  # checked item by item
            return await asyncio.gather(
                *(self._no_exceptions(model=model, messages=messages, **kwargs) for model in models)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
            )
        if isinstance(messages, list) and all(isinstance(m, list) for m in messages):  # pyright: ignore[reportUnknownVariableType]  # same
            rows: Final[list[object]] = list(messages)  # pyright: ignore[reportUnknownArgumentType]  # same  # mutable-ok: the request kwargs the Python router mutates in place

            async def indexed(model: str, message: object, idx: int) -> tuple[object, int]:
                return await self._no_exceptions(model=model, messages=message, **kwargs), idx

            responses: Final = await asyncio.gather(
                *(
                    indexed(model, message, idx) for idx, message in enumerate(rows) for model in models
                )  # comprehension-ok: a flat walk of nested rows
            )
            final: Final[list[list[object]]] = [  # mutable-ok: the request kwargs the Python router mutates in place
                [] for _ in rows
            ]
            for response, idx in responses:
                final[idx].append(response)
            return final
        return None

    async def abatch_completion_one_model_multiple_requests(
        self,
        model: str,
        messages: list[object],  # mutable-ok: the request kwargs the Python router mutates in place
        **kwargs: object,  # kwargs-ok: the endpoint's open request
    ) -> object:
        return await asyncio.gather(*(self._no_exceptions(model=model, messages=m, **kwargs) for m in messages))

    async def abatch_completion_fastest_response(
        self,
        model: str,
        messages: object,
        stream: bool = False,
        **kwargs: object,  # kwargs-ok: the request
    ) -> object:
        from litellm.litellm_core_utils.hidden_params import get_hidden_params
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
        from litellm.types.utils import ModelResponse

        models: Final = [m.strip() for m in model.split(",")]

        async def no_exceptions(model_name: str) -> object:
            try:
                return await self._core.acompletion(model=model_name, messages=messages, stream=stream, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001  # see _no_exceptions
                return error

        pending: set[  # mutable-ok: the request kwargs the Python router mutates in place  # rebind-ok: rebound as the Python router rebinds it
            asyncio.Task[object]
        ] = {asyncio.create_task(no_exceptions(m)) for m in models}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                if isinstance(result, (ModelResponse, CustomStreamWrapper)):
                    for other in pending:
                        other.cancel()
                    hidden = get_hidden_params(result)
                    if hidden is not None:
                        hidden["fastest_response_batch_completion"] = True
                    return result
        raise Exception("All tasks failed")  # noqa: TRY002  # the Python router raises a bare Exception

    # -- moderation, text completion and adapter --------------------------------------------

    async def amoderation(self, request: Mapping[str, object]) -> object:
        """`_pass_through_moderation_endpoint_factory`: one pick when the model is a group."""
        kwargs: Final = without(request, "custom_llm_provider", "client")
        plan: Final = PLANS["amoderation"]
        model: Final = kwargs.get("model", "")
        if not isinstance(model, str):
            _ = self._core.routed_name(model)
        if model and isinstance(model, str) and self._is_model_group(model):
            return await self._core.drive(plan, kwargs, _SINGLE_ATTEMPT)
        prepared: Final = prepare_request(plan, kwargs, str(model), self._core.resolved.model_group_alias)
        return await self._direct("amoderation", prepared)

    def text_completion(self, request: Mapping[str, object]) -> object:
        """The sync `text_completion`: one pick, no retries, the group name as the model."""
        from litellm.litellm_core_utils.routing_common_utils import (
            deployment_model_id_pin,
            metadata_defaults_under_request,
        )

        kwargs: Final = without(request, "is_retry", "is_fallback", "is_async")
        model: Final = kwargs.get("model")
        prompt: Final = kwargs.get("prompt")
        if "model" not in kwargs:
            raise TypeError("text_completion() missing 1 required positional argument: 'model'")
        kwargs["num_retries"] = kwargs.get("num_retries", self._core.resolved.num_retries)
        metadata_bucket(kwargs, "metadata").update({"model_group": model})
        with endpoint_selection("text_completion", model=model):
            deployment: Final = self._core.router.get_available_deployment(
                model=model,  # pyright: ignore[reportArgumentType]  # the name as given
                messages=_user_message(prompt),
                specific_deployment=kwargs.pop("specific_deployment", None),  # pyright: ignore[reportArgumentType]  # the request's value, as given
                request_kwargs=kwargs,
            )
        data: Final = as_dict(deployment.get("litellm_params"))
        for key, value in self._core.resolved.default_litellm_params.items():
            if key not in kwargs:
                kwargs[key] = value
            elif key == "metadata":
                metadata_bucket(kwargs, key).update(
                    metadata_defaults_under_request(defaults=as_dict(value), request_metadata=as_dict(kwargs[key]))
                )
        pinned: Final = deployment_model_id_pin(data)
        if pinned is not None:
            kwargs["model_id"] = pinned
        return self._invoke("text_completion")(**{**data, "prompt": prompt, "caching": self._cache_responses, **kwargs})

    async def aadapter_completion(
        self,
        adapter_id: str,
        model: str,
        is_retry: bool | None = False,
        is_fallback: bool | None = False,
        is_async: bool | None = False,
        **kwargs: object,  # kwargs-ok: the request
    ) -> object:
        return await self._core.drive(
            PLANS["aadapter_completion"], {**kwargs, "model": model, "adapter_id": adapter_id}
        )

    # -- files, batches and vector stores ---------------------------------------------------

    async def acreate_batch(self, model: str, **kwargs: object) -> object:  # kwargs-ok: the request
        return await self._core.drive(PLANS["acreate_batch"], {**kwargs, "model": model})

    async def acancel_batch(self, model: str, **kwargs: object) -> object:  # kwargs-ok: the request
        return await self._core.drive(PLANS["acancel_batch"], {**kwargs, "model": model})

    async def avector_store_create(self, model: str | None, **kwargs: object) -> object:  # kwargs-ok: the request
        if model is None:
            return await self._factory("avector_store_create", "vector_stores.main.acreate")(**kwargs)
        return await self._core.drive(PLANS["avector_store_create"], {**kwargs, "model": model}, _SINGLE_ATTEMPT)

    async def aretrieve_batch(self, model: str | None = None, **kwargs: object) -> object:  # kwargs-ok: the request
        """Ask every deployment of the group (or of the router) for the batch."""
        from litellm.litellm_core_utils.core_helpers import (
            safe_deep_copy,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        )
        from litellm.types.utils import LiteLLMBatch

        shaping: Final = self._core.shaping
        metadata_variable_name: Final = get_router_metadata_variable_name(function_name="aretrieve_batch")
        if model is not None:
            specific: Final = kwargs.pop("specific_deployment", None)
            healthy = await self._core.select(  # rebind-ok: rebound as the Python router rebinds it
                model,
                dict(kwargs),
                kind="healthy",
                specific_deployment=bool(specific) or None,
                messages=_user_message("retrieve-api-fake-text"),
            )
        else:
            healthy = [as_dict(r) for r in self._core.handle.deployments()]  # rebind-ok: as Python rebinds it
        errors: Final[list[BaseException]] = []  # mutable-ok: the request kwargs the Python router mutates in place

        async def try_retrieve(deployment: Mapping[str, object]) -> object:
            try:
                data: Final = as_dict(deployment.get("litellm_params"))
                deployment_model: Final = data.get("model")
                provider: object = data.get("custom_llm_provider")  # rebind-ok: inferred when not declared
                if deployment_model is None:
                    raise Exception(f"Model not found in litellm_params for deployment: {deployment}")  # noqa: TRY002  # as the Python router
                if not provider:
                    lookup: Final = self._core.host.invoke("get_llm_provider")
                    _, provider, _, _ = lookup(model=deployment_model)  # rebind-ok: as Python rebinds it
                new_kwargs: Final = safe_deep_copy(dict(kwargs))  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
                update_kwargs_with_deployment(
                    shaping,
                    deployment,
                    new_kwargs,  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                    cache=self._core.routing_cache,
                    function_name="aretrieve_batch",
                )
                bucket: Final = metadata_bucket(new_kwargs, metadata_variable_name)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                if not bucket.get("model_group"):
                    bucket["model_group"] = model or deployment.get("model_name")
                new_kwargs.pop("custom_llm_provider", None)  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values
                data.pop("custom_llm_provider", None)
                return await self._direct("aretrieve_batch", {**data, "custom_llm_provider": provider, **new_kwargs})  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
            except Exception as error:  # noqa: BLE001  # collected; the first one is raised when no deployment has the batch
                errors.append(error)
                return None

        deployments: Final = list(healthy) if isinstance(healthy, Sequence) else [healthy]
        if not deployments:
            raise Exception("No healthy deployments found.")  # noqa: TRY002  # as the Python router
        results: Final = await asyncio.gather(*(try_retrieve(as_dict(d)) for d in deployments), return_exceptions=True)
        for result in results:
            if isinstance(result, LiteLLMBatch):
                return result
        if errors:
            raise errors[0]
        raise Exception(f"Unable to find batch in any model. Received errors - {errors}")  # noqa: TRY002  # as the Python router

    async def alist_batches(self, model: str, **kwargs: object) -> object:  # kwargs-ok: the request
        deployments: Final = self._group(model)

        async def try_list(deployment: Mapping[str, object]) -> object:
            try:
                return await self._direct("alist_batches", {**as_dict(deployment.get("litellm_params")), **kwargs})
            except Exception:  # noqa: BLE001  # a failing deployment is left out, as the Python router does
                return None

        results: Final = await asyncio.gather(*(try_list(d) for d in deployments))
        final: Final[dict[str, object]] = {  # mutable-ok: the request kwargs the Python router mutates in place
            "object": "list",
            "data": [],
            "first_id": None,
            "last_id": None,
            "has_more": False,
        }
        data: Final[list[object]] = []  # mutable-ok: the request kwargs the Python router mutates in place
        final["data"] = data
        for result in results:
            if result is None:
                continue
            if final["first_id"] is None and hasattr(result, "first_id"):
                final["first_id"] = getattr(result, "first_id")  # noqa: B009  # as the Python router
            final["last_id"] = getattr(result, "last_id")  # noqa: B009  # same
            data.extend(getattr(result, "data"))  # noqa: B009  # same  # pyright: ignore[reportAny]  # litellm's untyped helpers and request values
            if getattr(result, "has_more", False) is True:
                final["has_more"] = True
        return final

    async def acreate_file(self, model: str, **kwargs: object) -> object:  # kwargs-ok: the request
        return await self._core.drive(PLANS["acreate_file"], {**kwargs, "model": model})

    async def create_file_everywhere(self, kwargs: Mapping[str, object], requested: str, specific: object) -> object:
        """`_acreate_file`: create the file on every healthy deployment of the group."""
        from litellm.litellm_core_utils import batch_utils
        from litellm.litellm_core_utils.core_helpers import (
            safe_deep_copy,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        )
        from litellm.litellm_core_utils.hidden_params import get_hidden_params
        from litellm.litellm_core_utils.routing_common_utils import (
            add_model_file_id_mappings,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        )
        from litellm.types.utils import LlmProviders

        healthy: Final = await self._core.select(
            requested,
            dict(kwargs),
            kind="healthy",
            specific_deployment=bool(specific) or None,
            messages=_user_message(_PICK_TEXTS["create_file"]),
            batch_upload=is_batch_upload(kwargs),
        )
        deployments: Final = list(healthy) if isinstance(healthy, Sequence) else [healthy]

        async def create_for(deployment: Mapping[str, object]) -> object:
            kwargs_copy: Final = safe_deep_copy(dict(kwargs))  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
            update_kwargs_with_deployment(
                self._core.shaping,
                deployment,
                kwargs_copy,  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                cache=self._core.routing_cache,
                function_name="acreate_file",
            )
            data: Final = as_dict(deployment.get("litellm_params"))
            model_name: Final = str(data.get("model"))
            self.counters.total_calls[model_name] += 1
            declared: Final = data.get("custom_llm_provider")
            stripped, inferred, _, _ = self._core.host.invoke("get_llm_provider")(
                model=data["model"], custom_llm_provider=declared
            )
            provider: Final = declared or inferred
            purpose: Final = kwargs.get("purpose")
            file: object = kwargs.get("file")  # rebind-ok: rewritten for the deployment
            if not file or not purpose:
                raise Exception("file and file_purpose are required for create_file")  # noqa: TRY002  # as the Python router
            replaces: Final = batch_utils.should_replace_model_in_jsonl(
                purpose=purpose,  # pyright: ignore[reportArgumentType]  # the caller's purpose
                passthrough=kwargs.get("passthrough") is True,
            )
            if replaces:
                file = batch_utils.replace_model_in_jsonl(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # an untyped helper  # rebind-ok: rewritten for the deployment
                    file_content=file,  # pyright: ignore[reportArgumentType]  # the caller's file
                    new_model_name=stripped,
                    keep_line_numbers=provider == LlmProviders.VERTEX_AI.value,
                )
                kwargs_copy["file"] = file
            if provider == LlmProviders.LITELLM_PROXY.value:
                kwargs_copy["extra_body"] = MappingProxyType(
                    {**as_dict(kwargs_copy.get("extra_body")), "target_model_names": stripped}  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # litellm's untyped helpers and request values
                )
            if "gcs_bucket_name" in data:
                metadata_bucket(kwargs_copy, "litellm_metadata")["gcs_bucket_name"] = data["gcs_bucket_name"]  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
            from litellm.router_rust.invoke.slots import deployment_slot

            with deployment_slot(self._core, deployment, kwargs_copy):  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                response: Final = await self._direct(
                    "acreate_file",
                    {
                        **data,
                        "custom_llm_provider": provider,
                        "caching": self._cache_responses,
                        "client": model_client(self._core, PLANS["acreate_file"], deployment, kwargs_copy),  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                        **kwargs_copy,
                    },  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
                )
            self.counters.success_calls[model_name] += 1
            return response

        responses: Final = await asyncio.gather(*(create_for(as_dict(d)) for d in deployments))
        if len(responses) == 0:
            raise Exception("No healthy deployments found.")  # noqa: TRY002  # as the Python router
        mapping: Final = add_model_file_id_mappings(healthy_deployments=deployments, responses=responses)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]  # deployment records
        returned: Final = responses[0]
        hidden: Final = get_hidden_params(returned)
        if hidden is not None:
            hidden["model_file_id_mapping"] = mapping
        return returned

    # -- search tools -------------------------------------------------------------------------

    async def asearch(
        self,
        custom_llm_provider: str | None = None,
        client: object = None,
        **kwargs: object,  # kwargs-ok: the request
    ) -> object:
        """`_asearch_with_fallbacks`: the search tool name (``search_tool_name``, else ``model``)
        routes as the model; the factory wrapper drops ``custom_llm_provider`` and ``client``."""
        request: Final = dict(kwargs)
        search_tool_name: Final = request.get("search_tool_name", request.get("model"))
        if not search_tool_name:
            raise ValueError("search_tool_name or model parameter is required for search")
        request["model"] = search_tool_name
        return await self._core.drive(PLANS["asearch"], request)

    # -- guardrails ---------------------------------------------------------------------------

    async def aguardrail(
        self,
        guardrail_name: str,
        original_function: Callable[..., Awaitable[object]],
        **kwargs: object,  # kwargs-ok: the request
    ) -> object:
        """Pick one of the guardrails named ``guardrail_name`` (weighted) and run it. A guardrail name
        is no model group, so the router's retries never apply."""
        request: Final = dict(kwargs)
        request.setdefault("litellm_trace_id", str(uuid.uuid4()))
        metadata_bucket(request, "litellm_metadata").update(
            {
                "model_group": guardrail_name,
                "model_group_alias": guardrail_name
                if guardrail_name in self._core.resolved.model_group_alias
                else None,
            }
        )
        verdict: Final = await original_function(**request, selected_guardrail=self.available_guardrail(guardrail_name))
        # `make_call`, `async_function_with_retries` and `async_function_with_fallbacks` stamp the result.
        headed: Final = await self._core.set_response_headers(verdict, guardrail_name, request)
        return add_fallback_headers_to_response(
            response=add_retry_headers_to_response(response=headed, attempted_retries=0, max_retries=None),
            attempted_fallbacks=0,
        )

    def available_guardrail(self, guardrail_name: str) -> object:
        """``_core_get_available_guardrail``: the one guardrail configured under the name, else one of
        them picked by weight (``simple_shuffle``, the name resolved through the alias map)."""
        from litellm.router_strategy.simple_shuffle import simple_shuffle

        candidates: Final = [g for g in self._guardrails if as_dict(g).get("guardrail_name") == guardrail_name]
        if not candidates:
            raise ValueError(f"No guardrail found with name: {guardrail_name}")
        if len(candidates) == 1:
            return candidates[0]
        return simple_shuffle(  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helper
            resolve_model_alias=self._core.resolve_alias,
            healthy_deployments=candidates,  # pyright: ignore[reportArgumentType]  # guardrail configs
            model=guardrail_name,
            request_kwargs=None,
        )

    # -- attempts -----------------------------------------------------------------------------

    def attempt_failed(
        self,
        plan: Plan,
        error: BaseException,
        deployment: Mapping[str, object],
        call: AttemptCall | None,
        requested: str,
    ) -> None:
        from litellm import exceptions as litellm_exceptions
        from litellm.types.router import LiteLLM_Params

        # `_acompletion`'s Timeout branch counts no failure and adds the deployment's timeouts; the
        # sync `_completion` has no such branch.
        timed_out: Final = plan.kind == "chat" and not plan.sync and isinstance(error, litellm_exceptions.Timeout)
        if not timed_out:
            self.counters.failed(plan, call.model_name if call is not None else None, requested)
        elif isinstance(error, litellm_exceptions.Timeout) and self._core.host.dynamic().expose_router_debug_in_errors:
            params: Final = as_dict(deployment.get("litellm_params"))
            try:
                typed: Mapping[str, object] = (  # rebind-ok: rebound as the Python router rebinds it
                    LiteLLM_Params.model_validate(params).model_dump()
                )
            except Exception:  # noqa: BLE001  # the params as configured, when they do not validate
                typed = params  # rebind-ok: rebound as the Python router rebinds it
            error.message += (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
                f"\n\nDeployment Info: request_timeout: {typed.get('request_timeout')}"
                f"\ntimeout: {typed.get('timeout')}"
                f"\nstream_timeout: {typed.get('stream_timeout')}"
            )

    def check_response(self, plan: Plan, response: object, call: AttemptCall) -> None:
        """What the attempt helper checks of a response before it counts it as served."""

    def sync_pre_call_checks(self, plan: Plan, deployment: Mapping[str, object], request: Mapping[str, object]) -> None:
        """`routing_strategy_pre_call_checks` on the sync path: the process callbacks' pre-call checks.
        Sync completion runs them for a group but not for a deployment id."""
        from litellm.integrations.custom_logger import CustomLogger

        if plan.kind == "chat":
            model: Final = str(request.get("model"))
            names: Final = {as_dict(r).get("model_name") for r in self._core.handle.deployments()}
            if model not in names and self._core.handle.deployment(model) is not None:
                return
        for callback in tuple(self._core.routing_callbacks()):
            if isinstance(callback, CustomLogger):
                callback.pre_call_check(dict(deployment))  # pyright: ignore[reportUnknownMemberType]  # litellm's untyped helpers and request values


_SINGLE_ATTEMPT: Final[Mapping[str, object]] = MappingProxyType({"num_retries": 0, "disable_fallbacks": True})


async def _answers_none(*_args: object, **_kwargs: object) -> None:  # kwargs-ok: any request
    """An endpoint the Python router's async wrapper has no branch for."""
    return


def _as_items(value: object) -> list[object]:  # mutable-ok: a fresh list
    return list(value) if isinstance(value, list) else []  # pyright: ignore[reportUnknownArgumentType]  # a config list


if TYPE_CHECKING:
    from litellm.router_config import RouterConfig
    from litellm.router_python_host import PythonHost
    from litellm.router_rust.core import Resolved


__all__ = [
    "PINNED_DEPLOYMENT_KEY",
    "PLANS",
    "ROUTING_KEYS",
    "AttemptCall",
    "Counters",
    "Plan",
    "RustEndpoints",
    "Shaping",
    "arm_compaction_state",
    "attempt_call",
    "clean_realtime_session_deployment",
    "compacted_attempt",
    "endpoint_names",
    "failover_fields",
    "generic_plan",
    "log_refused_call",
    "log_refused_call_sync",
    "pinned_vector_store_file_deployment",
    "pinned_video_deployment",
    "plan_for",
    "prepare_request",
    "reports_own_success",
    "selection_endpoint",
    "stamp_error",
    "unrouted_fields",
]
