from unittest.mock import MagicMock
import asyncio

# Import the specific components we need to test
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _registered_pass_through_routes,
)


def test_update_pass_through_route_updates_registry():
    """
    REGRESSION TEST: Verify that calling add_exact_path_route (or add_subpath_route)
    on an EXISTING route correctly updates the in-memory registry.
    """

    async def _async_test():
        # Setup - Unique IDs to avoid collision with other tests
        endpoint_id = "regression-test-endpoint"
        path = "/regression-test-path"
        route_key = f"{endpoint_id}:exact:{path}"
        target = "http://example.com"

        # Cleanup: Ensure clean state before test
        if route_key in _registered_pass_through_routes:
            del _registered_pass_through_routes[route_key]

        try:
            # 1. First Registration (Initial State)
            InitPassThroughEndpointHelpers.add_exact_path_route(
                app=MagicMock(),
                path=path,
                target=target,
                custom_headers={"Authorization": "Bearer INITIAL_TOKEN"},
                forward_headers=False,
                merge_query_params=False,
                dependencies=[],
                cost_per_request=0,
                endpoint_id=endpoint_id,
            )

            # Verify Initial State
            assert route_key in _registered_pass_through_routes
            initial_headers = _registered_pass_through_routes[route_key][
                "passthrough_params"
            ]["custom_headers"]
            assert initial_headers["Authorization"] == "Bearer INITIAL_TOKEN"

            # 2. Perform Update (Simulate API Update)
            # This call should overwrite the existing entry
            InitPassThroughEndpointHelpers.add_exact_path_route(
                app=MagicMock(),
                path=path,
                target=target,
                custom_headers={
                    "Authorization": "Bearer NEW_UPDATED_TOKEN"
                },  # Changed Header
                forward_headers=False,
                merge_query_params=False,
                dependencies=[],
                cost_per_request=0,
                endpoint_id=endpoint_id,
            )

            # 3. Verify Update Occurred
            updated_headers = _registered_pass_through_routes[route_key][
                "passthrough_params"
            ]["custom_headers"]

            # This assertion protects against the regression
            assert (
                updated_headers["Authorization"] == "Bearer NEW_UPDATED_TOKEN"
            ), "Registry failed to update! Old headers persisted despite update call."

        finally:
            # Cleanup: Remove test entry
            if route_key in _registered_pass_through_routes:
                del _registered_pass_through_routes[route_key]

    asyncio.run(_async_test())


def test_update_subpath_route_updates_registry():
    """
    REGRESSION TEST: Verify that calling add_subpath_route
    on an EXISTING route correctly updates the in-memory registry.
    """

    async def _async_test():
        # Setup
        endpoint_id = "regression-test-subpath"
        path = "/regression-test-wildcard"
        route_key = f"{endpoint_id}:subpath:{path}"
        target = "http://example.com"

        if route_key in _registered_pass_through_routes:
            del _registered_pass_through_routes[route_key]

        try:
            # 1. First Registration
            InitPassThroughEndpointHelpers.add_subpath_route(
                app=MagicMock(),
                path=path,
                target=target,
                custom_headers={"Authorization": "Bearer INITIAL_SUBPATH_TOKEN"},
                forward_headers=False,
                merge_query_params=False,
                dependencies=[],
                cost_per_request=0,
                endpoint_id=endpoint_id,
            )

            assert (
                _registered_pass_through_routes[route_key]["passthrough_params"][
                    "custom_headers"
                ]["Authorization"]
                == "Bearer INITIAL_SUBPATH_TOKEN"
            )

            # 2. Update
            InitPassThroughEndpointHelpers.add_subpath_route(
                app=MagicMock(),
                path=path,
                target=target,
                custom_headers={"Authorization": "Bearer NEW_SUBPATH_TOKEN"},
                forward_headers=False,
                merge_query_params=False,
                dependencies=[],
                cost_per_request=0,
                endpoint_id=endpoint_id,
            )

            # 3. Verify
            updated_headers = _registered_pass_through_routes[route_key][
                "passthrough_params"
            ]["custom_headers"]
            assert (
                updated_headers["Authorization"] == "Bearer NEW_SUBPATH_TOKEN"
            ), "Subpath registry failed to update!"

        finally:
            if route_key in _registered_pass_through_routes:
                del _registered_pass_through_routes[route_key]

    asyncio.run(_async_test())
