"""Path allowlist for the UI backend (control plane) component.

The backend exposes management/admin endpoints consumed by the UI: keys, users,
teams, orgs, customers, budgets, tags, workflows, model management, spend &
analytics, settings (router/cache/cost-tracking/fallbacks), SSO/onboarding,
audit logs, debug, enterprise admin, and UI bootstrap helpers (logo, favicon,
.well-known config).

Anything LLM data-plane is dropped — those run on the gateway component.
"""

BACKEND_PATH_PREFIXES: tuple[str, ...] = (
    # Identity / access
    "/key/",
    "/v2/key/",
    "/user/",
    "/v2/user/",
    "/team/",
    "/v2/team/",
    "/organization/",
    "/customer/",
    "/end_user/",
    "/sso/",
    "/login",
    "/v2/login",
    "/v3/login",
    "/logout",
    "/token",
    "/onboarding/",
    "/audit",
    "/oauth/",
    "/invitation/",
    "/jwt/",
    "/secure_share/",
    # Models & routing config
    "/model/",
    "/v1/model/info",
    "/v2/model/",
    "/model_group",
    "/model_access_group/",
    "/model_hub/",
    "/v1/access_group",
    "/access_group/",
    "/router/",
    "/router_settings",
    "/adaptive_router/",
    "/fallback",
    "/fallbacks",
    "/cache_settings",
    "/coordination_redis/",
    "/cost_tracking",
    "/cost/",
    "/credentials",
    "/credential",
    "/provider/budgets",
    # Tools / agents (registry & policy admin)
    "/v1/tool/",
    "/v1/agents",
    # Guardrails admin
    "/v2/guardrails/",
    # MCP server admin + BYOK OAuth flow (UI-initiated) + dynamic per-server endpoints
    "/v1/mcp/",
    "/test/",
    "/{mcp_server_name}/",
    # Budgets / tags / workflows / memory mgmt
    "/budget/",
    "/tag/",
    "/workflow/",
    "/v1/workflows/",
    "/project/",
    "/memory/",
    "/mcp/",
    # Spend / analytics
    "/spend/",
    "/analytics/",
    "/global/",
    "/user_agent",
    "/usage/",
    "/daily/",
    # CloudZero cost-export admin (init / settings / export / dry-run / delete)
    "/cloudzero/",
    # Caching admin
    "/cache/",
    "/caching/",
    # Callbacks / hooks
    "/active/callbacks",
    "/callbacks",
    "/team_callback",
    # Rust data-plane gateway → proxy control-plane API (logging today, auth later)
    "/v1/rust_control_plane/",
    # Alerting / email / IP allowlist
    "/alerting/",
    "/email/",
    "/add/allowed_ip",
    "/delete/allowed_ip",
    "/get/",
    # Enterprise admin
    "/enterprise/",
    # Debug / config / profiling
    "/debug/",
    "/config/",
    "/memory-usage-in-mem-cache",
    "/otel-spans",
    "/lazy/",
    "/in_product_nudges",
    # Admin reload / schedule
    "/reload/",
    "/schedule/",
    "/settings",
    "/update/",
    "/upload/",
    # Dev / admin utilities
    "/utils/",
    # UI bootstrap helpers (assets the dashboard fetches)
    "/get_logo_url",
    "/get_image",
    "/get_favicon",
    "/.well-known/",
    "/litellm/.well-known/",
    "/ui_discovery/",
    "/ui-config",
    "/sso_settings",
    "/public/",
    "/robots.txt",
    # Health (k8s probes)
    "/health",
    # Plugin system
    "/api/plugins",
    "/plugin-proxy/",
)

BACKEND_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/routes",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/fallback/login",
    }
)

BACKEND_MOUNT_PATHS: frozenset[str] = frozenset(
    {
        "/swagger",  # API documentation static assets belong to the backend
    }
)
