"""
Helpers for the endpoint characterization tests (``test_endpoints_characterization*.py``).

``EndpointHost`` is a ``ScriptedHost`` that also scripts the host ops ``ScriptedHost`` leaves to
litellm (assistants, skills, realtime, fine-tuning, batches, ...), so every ``host.invoke`` the core
makes is answered and recorded.

Some endpoints are served by a host op named by its dotted path under ``litellm`` (vector stores,
Google GenAI, OCR, search, videos, containers, interactions, managed agents), which ``host.invoke``
resolves from litellm's modules at call time. ``patch_direct_endpoints`` replaces those litellm
functions with recording stand-ins that answer from the same host script. A call to one of them is
recorded with the function's name as its ``op``.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Final

import pytest

from tests.unit.test_router.core.scripted_host import INFERENCE_OPS, ScriptedHost

EXTRA_HOST_OPS: Final = frozenset(
    {
        "aadapter_completion",
        "acreate_assistants",
        "adelete_assistant",
        "aget_assistants",
        "acreate_thread",
        "aget_thread",
        "a_add_message",
        "aget_messages",
        "arun_thread",
        "acompact_responses",
        "_arealtime",
        "acreate_realtime_client_secret",
        "arealtime_calls",
        "acreate_realtime_transcription_session",
        "_aresponses_websocket",
        "acancel_fine_tuning_job",
        "alist_fine_tuning_jobs",
        "aretrieve_fine_tuning_job",
        "afile_list",
        "aimage_edit",
        "allm_passthrough_route",
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
        "aretrieve_batch",
        "acancel_batch",
        "alist_batches",
    }
)


class EndpointHost(ScriptedHost):
    """``ScriptedHost`` that scripts every host op the core's endpoints invoke."""

    __slots__ = ()

    def invoke(self, op: str) -> Callable[..., object]:
        if op in INFERENCE_OPS or op not in EXTRA_HOST_OPS:
            return super().invoke(op)
        return recording_endpoint(self, op, is_async=True)


def recording_endpoint(host: ScriptedHost, name: str, *, is_async: bool) -> Callable[..., object]:
    """A stand-in for litellm's ``name`` that records the call on ``host`` and answers from its script."""
    if is_async:

        async def endpoint(*args: object, **kwargs: object) -> object:  # kwargs-ok: the endpoint's request
            return await host._answer_async(name, args, kwargs)  # pyright: ignore[reportPrivateUsage]  # the test host

        endpoint.__name__ = name
        endpoint.__qualname__ = name
        return endpoint

    def sync_endpoint(*args: object, **kwargs: object) -> object:  # kwargs-ok: the endpoint's request
        return host._answer_sync(name, args, kwargs)  # pyright: ignore[reportPrivateUsage]  # the test host

    sync_endpoint.__name__ = name
    sync_endpoint.__qualname__ = name
    return sync_endpoint


# module -> {attribute on the module: the function's own name}
DIRECT_ENDPOINTS: Final[Mapping[str, Mapping[str, str]]] = {
    "litellm.vector_stores.main": {
        name: name
        for name in (
            "asearch",
            "search",
            "acreate",
            "create",
            "aretrieve",
            "retrieve",
            "alist",
            "list",
            "aupdate",
            "update",
            "adelete",
            "delete",
        )
    },
    "litellm.vector_stores": {"acreate": "acreate"},
    "litellm.vector_store_files.main": {
        name: name
        for name in (
            "acreate",
            "create",
            "alist",
            "list",
            "aretrieve",
            "retrieve",
            "aretrieve_content",
            "retrieve_content",
            "aupdate",
            "update",
            "adelete",
            "delete",
        )
    },
    "litellm.google_genai": {
        name: name
        for name in ("agenerate_content", "generate_content", "agenerate_content_stream", "generate_content_stream")
    },
    "litellm.ocr": {"aocr": "aocr", "ocr": "ocr"},
    "litellm.search": {"asearch": "asearch", "search": "search"},
    "litellm.decisions": {"adecisions": "adecisions", "decisions": "decisions"},
    "litellm.systemone": {"asystemone": "asystemone", "systemone": "systemone"},
    "litellm.extraction": {"aextraction": "aextraction", "extraction": "extraction"},
    "litellm.videos": {
        f"{prefix}video_{name}": f"{prefix}video_{name}"
        for prefix in ("a", "")
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
        )
    },
    "litellm.containers": {
        f"{prefix}{name}": f"{prefix}{name}"
        for prefix in ("a", "")
        for name in ("create_container", "list_containers", "retrieve_container", "delete_container")
    },
    "litellm.interactions": {
        f"{prefix}{name}": f"{prefix}{name}" for prefix in ("a", "") for name in ("create", "get", "delete", "cancel")
    },
    "litellm.interactions.agents": {
        f"{prefix}{name}": f"{prefix}{name}"
        for prefix in ("a", "")
        for name in ("create", "get", "delete", "list", "list_versions")
    },
}

# Recorded op names for the direct endpoints whose bare names collide across modules.
RECORDED_NAMES: Final[Mapping[tuple[str, str], str]] = {
    **{("litellm.vector_stores.main", name): f"vs.{name}" for name in DIRECT_ENDPOINTS["litellm.vector_stores.main"]},
    ("litellm.vector_stores", "acreate"): "vs.acreate",
    **{
        ("litellm.vector_store_files.main", name): f"vsf.{name}"
        for name in DIRECT_ENDPOINTS["litellm.vector_store_files.main"]
    },
    **{("litellm.interactions", name): f"interactions.{name}" for name in DIRECT_ENDPOINTS["litellm.interactions"]},
    **{
        ("litellm.interactions.agents", name): f"agents.{name}"
        for name in DIRECT_ENDPOINTS["litellm.interactions.agents"]
    },
}


def patch_direct_endpoints(monkeypatch: pytest.MonkeyPatch, host: ScriptedHost) -> None:
    """Replace litellm's directly bound endpoint functions with recording stand-ins on ``host``.

    The call is recorded under ``RECORDED_NAMES`` when the bare name is ambiguous, else under the
    function's name."""
    for module_name, attributes in DIRECT_ENDPOINTS.items():
        module: Final = importlib.import_module(module_name)
        for attribute, function_name in attributes.items():
            original: Final[object] = getattr(module, attribute)
            is_async: Final = function_name.startswith("a")
            recorded: Final = RECORDED_NAMES.get((module_name, attribute), function_name)
            stand_in: Final = recording_endpoint(host, recorded, is_async=is_async)
            stand_in.__name__ = getattr(original, "__name__", function_name)
            monkeypatch.setattr(module, attribute, stand_in)
    from litellm.containers import endpoint_factory

    for name in list(endpoint_factory._generated_endpoints):  # pyright: ignore[reportPrivateUsage]  # litellm's registry
        monkeypatch.setitem(
            endpoint_factory._generated_endpoints,  # pyright: ignore[reportPrivateUsage]  # litellm's registry
            name,
            recording_endpoint(host, name, is_async=name.startswith("a")),
        )


__all__ = [
    "DIRECT_ENDPOINTS",
    "EXTRA_HOST_OPS",
    "EndpointHost",
    "patch_direct_endpoints",
    "recording_endpoint",
]
