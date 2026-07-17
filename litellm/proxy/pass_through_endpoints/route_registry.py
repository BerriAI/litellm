"""In-memory registry of user-defined pass-through routes (config + DB).

This module is a dependency leaf: it must not import anything from
``litellm.proxy``. ``pass_through_endpoints.py`` imports ``user_api_key_auth``,
which imports ``auth_utils``, so the auth layer can only consult the registry
through a module that sits outside that cycle.

Registry keys are ``"{endpoint_id}:{exact|subpath}:{path}:{methods}"`` (the
methods segment is absent in older entries). Lookups expect a root-stripped
route, as returned by ``get_request_route`` /
``InitPassThroughEndpointHelpers._route_for_registry_lookup``.
"""

RegisteredRouteEntry = dict[str, object]

registered_pass_through_routes: dict[str, RegisteredRouteEntry] = {}  # mutable-ok: mutated on config and DB reload


def _key_matches_route(key: str, route: str) -> bool:
    parts = key.split(":", 3)
    if len(parts) < 3:
        return False
    route_type = parts[1]
    registered_path = parts[2]
    if route_type == "exact":
        return route == registered_path
    if route_type == "subpath":
        return route == registered_path or route.startswith(registered_path + "/")
    return False


def is_registered_custom_pass_through_route(route: str) -> bool:
    """Whether ``route`` matches a user-defined pass-through endpoint.

    Only covers endpoints registered via ``general_settings.pass_through_endpoints``
    or the DB overlay; built-in provider passthrough routes
    (``LiteLLMRoutes.mapped_pass_through_routes``) are deliberately excluded.
    A ``model`` field in the body of these requests belongs to the upstream's
    namespace, not LiteLLM's, so auth must not treat it as a managed model.
    """
    return any(_key_matches_route(key, route) for key in registered_pass_through_routes)
