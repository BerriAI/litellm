"""``with_service_target`` and ``service_caller`` carry the purpose and the caller of a datastore call
to code that cannot see them from its own frames, and every Redis producer on the proxy request path
declares a key family so no request-path span renders as a bare ``redis.get``."""

import ast
import asyncio
import contextvars
import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from litellm._internal_context import (
    current_service_caller,
    current_service_target,
    service_caller,
    service_target,
    with_service_target,
)

_REPO: Final = Path(__file__).resolve().parents[2]

_REDIS_PRODUCER_ROOTS: Final = ("litellm", "enterprise")
# The cache implementations and facades: they emit the service events, their callers declare the family.
_CACHE_LAYER_DIRS: Final = ("litellm/caching", "litellm/_v2/cache")
# Helpers that act on a cache handed in by the declaring caller, or forward to the response-cache facade.
_CACHE_PARAMETER_HELPERS: Final = frozenset(
    {
        "litellm/proxy/common_utils/cache_coordinator.py",
        "litellm/proxy/common_utils/user_api_key_cache.py",
        "litellm/utils.py",
    }
)
# Callers whose every cache call hits a process-local ``InMemoryCache`` (a ``DualCache`` built without
# ``redis_cache``, a ``local_only=True`` call, the client / logger / tool-name caches), so no Redis span exists.
_IN_MEMORY_ONLY_CALLERS: Final = frozenset(
    {
        "litellm/integrations/datadog/datadog_team_handler.py",
        "litellm/integrations/humanloop.py",
        "litellm/integrations/langfuse/langfuse_handler.py",
        "litellm/integrations/langfuse/langfuse_prompt_management.py",
        "litellm/integrations/newrelic/newrelic_team_handler.py",
        "litellm/integrations/shadow_eval_logger.py",
        "litellm/litellm_core_utils/litellm_logging.py",
        "litellm/litellm_core_utils/oauth_token_exchange.py",
        "litellm/litellm_core_utils/prompt_templates/factory.py",
        "litellm/litellm_core_utils/prompt_templates/image_handling.py",
        "litellm/litellm_core_utils/specialty_caches/dynamic_logging_cache.py",
        "litellm/litellm_core_utils/specialty_caches/service_trace_id_cache.py",
        "litellm/llms/azure/common_utils.py",
        "litellm/llms/bedrock/base_aws_llm.py",
        "litellm/llms/custom_httpx/http_handler.py",
        "litellm/llms/gigachat/authenticator.py",
        "litellm/llms/litellm_proxy/skills/handler.py",
        "litellm/llms/openai/common_utils.py",
        "litellm/llms/openai_like/model_info.py",
        "litellm/llms/vertex_ai/vertex_ai_non_gemini.py",
        "litellm/llms/watsonx/common_utils.py",
        "litellm/proxy/_experimental/mcp_server/byok_credential_cache.py",
        "litellm/proxy/_experimental/mcp_server/catalog.py",
        "litellm/proxy/_experimental/mcp_server/discoverable_endpoints.py",
        "litellm/proxy/_experimental/mcp_server/oauth_identity_binding.py",
        "litellm/proxy/_experimental/mcp_server/operations.py",
        "litellm/proxy/_experimental/mcp_server/outbound_credentials/sso_assertion_store.py",
        "litellm/proxy/_experimental/mcp_server/outbound_credentials/token_endpoint.py",
        "litellm/proxy/agent_endpoints/databricks_oauth.py",
        "litellm/proxy/common_utils/registry_read_through.py",
        "litellm/proxy/container_endpoints/ownership.py",
        "litellm/proxy/discovery_endpoints/agent_skills_endpoints.py",
        "litellm/proxy/guardrails/guardrail_hooks/straiker/straiker.py",
        "litellm/proxy/spend_tracking/key_metadata_recovery.py",
        "litellm/proxy/ui_crud_endpoints/latest_release_endpoints.py",
        "litellm/responses/litellm_completion_transformation/transformation.py",
        "litellm/router_utils/client_initalization_utils.py",
        "litellm/router_utils/router_callbacks/track_deployment_metrics.py",
        "litellm/secret_managers/cyberark_secret_manager.py",
        "litellm/secret_managers/google_secret_manager.py",
        "litellm/secret_managers/hashicorp_secret_manager.py",
        "litellm/secret_managers/main.py",
    }
)

_CACHE_CALL: Final = re.compile(
    r"\.(?:async_)?(?:get_cache|set_cache|batch_get_cache|batch_get_cache_shared|increment_cache|increment"
    r"|set_cache_pipeline|set_cache_pipeline_with_ttls|set_cache_sadd|delete_cache|batch_set_cache|increment_pipeline"
    r"|rpush|lpop|scan_iter|get_ttl|mget)\("
    r"|\b(?:reserve_redis_batch_reads|declare_batch_get|_prepare_batch_get)\("
    r"|\bbatch\.(?:set|delete|script|increment)\("
)
_DECLARES_TARGET: Final = re.compile(r"\b(?:with_service_target|service_target|response_cache_phase)\(")
_BUILDS_A_REDIS_CACHE: Final = re.compile(r"\bRedisCache\(|\bredis_cache=(?!None\b)")


def _redis_producers() -> tuple[str, ...]:
    files: Final = tuple(
        path for root in _REDIS_PRODUCER_ROOTS for path in sorted((_REPO / root).rglob("*.py"))
    )  # comprehension-ok: flatten the producer roots
    relative: Final = tuple(
        path.relative_to(_REPO).as_posix() for path in files if _CACHE_CALL.search(path.read_text())
    )
    return tuple(name for name in relative if not name.startswith(_CACHE_LAYER_DIRS))


def test_every_redis_producer_declares_a_key_family() -> None:
    """A module that reads or writes a shared cache without a declared target renders as a
    bare ``redis.get`` / ``redis.mget`` (flat under the request span, or an unnamed INTERNAL root
    for a background job), which is exactly what the sensitive-data pin read, the rate-limiter
    MGET and the budget-reset job did in production. Only process-local callers are exempt."""
    exempt: Final = _CACHE_PARAMETER_HELPERS | _IN_MEMORY_ONLY_CALLERS
    undeclared: Final = tuple(
        name
        for name in _redis_producers()
        if name not in exempt and not _DECLARES_TARGET.search((_REPO / name).read_text())
    )
    assert undeclared == ()


def test_every_in_memory_exemption_still_only_touches_a_process_local_cache() -> None:
    """The exemption list is a claim about each file, so a file that is deleted or starts building
    or receiving a ``RedisCache`` has to leave the list (and declare a family) rather than stay exempt."""
    producers: Final = frozenset(_redis_producers())
    stale: Final = tuple(sorted(_IN_MEMORY_ONLY_CALLERS - producers))
    assert stale == ()
    redis_backed: Final = tuple(
        name for name in sorted(_IN_MEMORY_ONLY_CALLERS) if _BUILDS_A_REDIS_CACHE.search((_REPO / name).read_text())
    )
    assert redis_backed == ()


def test_with_service_target_sets_the_target_for_sync_and_async_calls_and_restores_it() -> None:
    @with_service_target("rate_limits")
    def read() -> str | None:
        return current_service_target()

    @with_service_target("rate_limits")
    async def read_async() -> str | None:
        await asyncio.sleep(0)
        return current_service_target()

    assert read() == "rate_limits"
    assert asyncio.run(read_async()) == "rate_limits"
    assert current_service_target() is None
    with service_target("auth_objects"):
        assert read() == "rate_limits"
        assert current_service_target() == "auth_objects"


def test_with_service_target_keeps_the_wrapped_signature_and_coroutine_ness() -> None:
    import inspect

    @with_service_target("rate_limits")
    async def hook(self: object, data: dict[str, str], call_type: str) -> None:
        return None

    assert inspect.iscoroutinefunction(hook)
    assert tuple(inspect.signature(hook).parameters) == ("self", "data", "call_type")
    assert hook.__name__ == "hook"


def test_service_caller_is_inherited_by_a_task_spawned_inside_it_and_cleared_after() -> None:
    async def spawned() -> str | None:
        return current_service_caller()

    async def main() -> tuple[str | None, str | None]:
        with service_caller("prefetch <- auth"):
            task = asyncio.create_task(spawned())
        return await task, current_service_caller()

    assert asyncio.run(main()) == ("prefetch <- auth", None)


@pytest.mark.parametrize("value", [None, "x"])
def test_service_caller_restores_the_outer_value(value: str | None) -> None:
    with service_caller(value):
        with service_caller("inner"):
            assert current_service_caller() == "inner"
        assert current_service_caller() == value
    assert current_service_caller() is None


class _Suspend:
    def __await__(self) -> Generator[None]:
        yield


def test_a_targeted_coroutine_closed_from_another_context_does_not_raise() -> None:
    @with_service_target("router_usage")
    async def sync_forever() -> None:
        await _Suspend()

    suspended: Final = sync_forever()
    contextvars.copy_context().run(suspended.send, None)
    contextvars.copy_context().run(suspended.close)
    assert current_service_target() is None


_DIRECT_REDIS_CALL: Final = re.compile(r"\b_?redis_cache\.(?!async_register_script\b)(?:async_)?\w+\(")


@dataclass(frozen=True, slots=True)
class _FunctionScan:
    name: str
    reaches_redis_directly: bool
    declares_a_family: bool
    referenced_names: frozenset[str]


def _scan_function(source: str, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> _FunctionScan:
    body: Final = ast.get_source_segment(source, fn) or ""
    decorators: Final = "\n".join(ast.get_source_segment(source, d) or "" for d in fn.decorator_list)
    nodes: Final = tuple(ast.walk(fn))
    names: Final = frozenset(n.id for n in nodes if isinstance(n, ast.Name))
    attrs: Final = frozenset(n.attr for n in nodes if isinstance(n, ast.Attribute))
    return _FunctionScan(
        name=fn.name,
        reaches_redis_directly=bool(_DIRECT_REDIS_CALL.search(body)),
        declares_a_family=bool(_DECLARES_TARGET.search(body + "\n" + decorators)),
        referenced_names=(names | attrs) - {fn.name},
    )


def _covered_by_callers(scans: tuple[_FunctionScan, ...], covered: frozenset[str]) -> frozenset[str]:
    """Close ``covered`` over functions whose every in-file caller already declares a family."""
    callers: Final = {
        scan.name: frozenset(
            other.name for other in scans if other.name != scan.name and scan.name in other.referenced_names
        )
        for scan in scans
    }
    grown: Final = covered | frozenset(
        name for name, callers_of in callers.items() if callers_of and callers_of <= covered
    )
    return grown if grown == covered else _covered_by_callers(scans, grown)


def _direct_redis_callers_without_a_family(name: str) -> tuple[str, ...]:
    source: Final = (_REPO / name).read_text()
    scans: Final = tuple(
        _scan_function(source, node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    declared: Final = frozenset(scan.name for scan in scans if scan.declares_a_family)
    covered: Final = _covered_by_callers(scans, declared)
    return tuple(f"{name}::{scan.name}" for scan in scans if scan.reaches_redis_directly and scan.name not in covered)


def test_every_function_that_reaches_redis_directly_declares_its_family() -> None:
    """A file-level declaration hides the producer that lacks one: the Claude Code session router
    binding read sat in ``router.py`` beside dozens of declared families and still shipped as a bare
    ``redis.get``. A function that bypasses the cache facades and calls ``redis_cache`` itself must
    carry the family on itself, its decorator, or every one of its in-file callers."""
    exempt_files: Final = _CACHE_PARAMETER_HELPERS | _IN_MEMORY_ONLY_CALLERS
    undeclared: Final = tuple(
        function
        for name in _redis_producers()
        if name not in exempt_files
        for function in _direct_redis_callers_without_a_family(name)
    )  # comprehension-ok: flatten per-file findings
    assert undeclared == ()
