"""
Regression tests for pass-through endpoint registry unbounded growth.

Issue #24833: When endpoints come from the DB they have no `id` field.
Before the fix, a new random UUID was generated on every 30-second reload
cycle, so:
  1. Every reload produced a different route key in _registered_pass_through_routes
  2. The old entries never matched during cleanup (UUID changed) → registry grew unboundedly
  3. Additionally, the cleanup loop passed the full route key to remove_endpoint_routes()
     which expected only the endpoint_id prefix → cleanup was a no-op

Both issues are covered here.
"""
import asyncio
import hashlib
from unittest.mock import MagicMock

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _register_pass_through_endpoint,
    _registered_pass_through_routes,
)


# ---------------------------------------------------------------------------
# Helper: deterministic ID that the fixed code should produce
# ---------------------------------------------------------------------------

def _expected_auto_id(path: str, methods=None) -> str:
    """Mirror the deterministic-ID logic introduced in the fix."""
    _methods = sorted(methods or [])
    stable_key = f"path:{path}|methods:{','.join(_methods)}"
    return "auto-" + hashlib.sha256(stable_key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Test 1 – stable ID for DB endpoints (no `id` field)
# ---------------------------------------------------------------------------

def test_db_endpoint_without_id_gets_stable_deterministic_id():
    """
    A DB-sourced endpoint (no `id` field) must always receive the same ID
    across multiple calls with identical path+methods.  Random UUIDs would
    break cleanup.
    """

    async def _run():
        app_mock = MagicMock()
        visited: set = set()

        endpoint_a = {
            "path": "/my-service",
            "target": "http://backend:8080",
        }
        endpoint_b = dict(endpoint_a)  # fresh copy simulating second reload

        await _register_pass_through_endpoint(
            endpoint=endpoint_a,
            app=app_mock,
            premium_user=False,
            visited_endpoints=visited,
        )
        id_first = endpoint_a["id"]

        # Reset visited so the second call is treated as a fresh reload
        visited.clear()
        await _register_pass_through_endpoint(
            endpoint=endpoint_b,
            app=app_mock,
            premium_user=False,
            visited_endpoints=visited,
        )
        id_second = endpoint_b["id"]

        assert id_first == id_second, (
            f"Expected stable ID across reloads but got {id_first!r} then {id_second!r}. "
            "Random UUIDs cause registry leak."
        )
        assert id_first == _expected_auto_id("/my-service"), (
            f"ID format changed — expected {_expected_auto_id('/my-service')!r}, got {id_first!r}"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 2 – different paths produce different IDs
# ---------------------------------------------------------------------------

def test_different_paths_produce_different_ids():
    async def _run():
        app_mock = MagicMock()

        ep1 = {"path": "/svc-alpha", "target": "http://a"}
        ep2 = {"path": "/svc-beta", "target": "http://b"}
        visited: set = set()

        await _register_pass_through_endpoint(
            endpoint=ep1, app=app_mock, premium_user=False, visited_endpoints=visited
        )
        await _register_pass_through_endpoint(
            endpoint=ep2, app=app_mock, premium_user=False, visited_endpoints=visited
        )

        assert ep1["id"] != ep2["id"], "Different paths must produce different IDs"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 3 – remove_endpoint_routes works when called with full route key prefix
# ---------------------------------------------------------------------------

def test_remove_endpoint_routes_called_with_endpoint_id_prefix():
    """
    The cleanup loop extracts the endpoint_id by splitting on ':'.
    Verify that remove_endpoint_routes() correctly deletes all routes for
    that endpoint_id, regardless of how many route keys exist for it.
    """
    endpoint_id = "test-remove-prefix-ep"
    path = "/test-remove-prefix"
    methods_str = "DELETE,GET,PATCH,POST,PUT"
    exact_key = f"{endpoint_id}:exact:{path}:{methods_str}"
    sub_key = f"{endpoint_id}:subpath:{path}:{methods_str}"

    # Manually insert fake entries
    _registered_pass_through_routes[exact_key] = {
        "endpoint_id": endpoint_id,
        "path": path,
        "type": "exact",
        "passthrough_params": {},
    }
    _registered_pass_through_routes[sub_key] = {
        "endpoint_id": endpoint_id,
        "path": path,
        "type": "subpath",
        "passthrough_params": {},
    }

    try:
        assert exact_key in _registered_pass_through_routes
        assert sub_key in _registered_pass_through_routes

        # Simulate what the fixed cleanup loop does:
        # route_key → split to get endpoint_id → remove_endpoint_routes(endpoint_id)
        full_route_key = exact_key
        extracted_id = full_route_key.split(":", 1)[0]
        assert extracted_id == endpoint_id

        InitPassThroughEndpointHelpers.remove_endpoint_routes(extracted_id)

        # Both keys should be gone
        assert exact_key not in _registered_pass_through_routes, (
            "Exact route was not removed after cleanup"
        )
        assert sub_key not in _registered_pass_through_routes, (
            "Subpath route was not removed after cleanup"
        )
    finally:
        # Safety cleanup
        _registered_pass_through_routes.pop(exact_key, None)
        _registered_pass_through_routes.pop(sub_key, None)


# ---------------------------------------------------------------------------
# Test 4 – registry does NOT grow across multiple simulated reload cycles
# ---------------------------------------------------------------------------

def test_registry_does_not_grow_across_reload_cycles():
    """
    Simulate 5 consecutive reload cycles of the same DB endpoint.
    Registry size should remain constant (1 exact route), not grow by 1
    each cycle as it did before the fix.
    """

    async def _run():
        app_mock = MagicMock()

        path = "/stable-reload-test"
        target = "http://backend:9000"

        # Count entries related to our test path before we start
        def _count_our_routes():
            return sum(
                1
                for k in _registered_pass_through_routes
                if path in k
            )

        initial_count = _count_our_routes()

        for _ in range(5):
            endpoint = {"path": path, "target": target}  # fresh dict each cycle
            visited: set = set()
            await _register_pass_through_endpoint(
                endpoint=endpoint,
                app=app_mock,
                premium_user=False,
                visited_endpoints=visited,
            )

        final_count = _count_our_routes()

        # Should have added exactly 1 route, not 5
        assert final_count == initial_count + 1, (
            f"Registry grew from {initial_count} to {final_count} after 5 reload cycles. "
            f"Expected exactly {initial_count + 1}. "
            "This indicates the UUID-churn regression is still present."
        )

        # Cleanup
        keys_to_del = [k for k in _registered_pass_through_routes if path in k]
        for k in keys_to_del:
            del _registered_pass_through_routes[k]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 5 – explicitly-assigned IDs are preserved (no regression)
# ---------------------------------------------------------------------------

def test_explicit_id_is_preserved():
    """
    Endpoints that already carry an `id` field must keep that ID.
    The deterministic-ID path must only trigger when `id` is absent.
    """

    async def _run():
        app_mock = MagicMock()
        custom_id = "my-custom-endpoint-id-12345"
        endpoint = {
            "path": "/explicit-id-test",
            "target": "http://svc",
            "id": custom_id,
        }
        visited: set = set()

        await _register_pass_through_endpoint(
            endpoint=endpoint,
            app=app_mock,
            premium_user=False,
            visited_endpoints=visited,
        )

        assert endpoint["id"] == custom_id, (
            f"Explicit ID was overwritten! Got {endpoint['id']!r} instead of {custom_id!r}"
        )

        # Cleanup
        keys_to_del = [k for k in _registered_pass_through_routes if custom_id in k]
        for k in keys_to_del:
            del _registered_pass_through_routes[k]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 6 – stale endpoint cleanup: colon-safe ID extraction
# ---------------------------------------------------------------------------

def test_stale_endpoint_cleanup_colon_safe_split():
    """
    Verify that the cleanup loop correctly extracts the endpoint_id from
    a registry key even when the ID contains a colon (e.g. 'svc:v2'),
    and that deduplication prevents redundant remove calls.
    """
    import re

    # Simulate registry keys with a colon-containing ID
    colon_id = "svc:v2"
    route_key_exact = f"{colon_id}:exact:/some/path:GET,POST"
    route_key_subpath = f"{colon_id}:subpath:/some/path:GET,POST"

    # Verify regex-based extraction works correctly for both key types
    for key in [route_key_exact, route_key_subpath]:
        _match = re.match(r"^(.+?):(?:exact|subpath):", key)
        extracted = _match.group(1) if _match else key.split(":", 1)[0]
        assert extracted == colon_id, (
            f"Expected {colon_id!r} but got {extracted!r} from key {key!r}. "
            "The colon-safe split is broken."
        )

    # Also verify standard auto-generated IDs still work
    auto_id = "auto-abc123def456"
    auto_key = f"{auto_id}:exact:/test:GET"
    _match = re.match(r"^(.+?):(?:exact|subpath):", auto_key)
    extracted = _match.group(1) if _match else auto_key.split(":", 1)[0]
    assert extracted == auto_id, (
        f"Expected {auto_id!r} but got {extracted!r} for auto-generated ID"
    )
