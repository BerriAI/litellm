import { routeSegmentForPathname } from "@/utils/uiHref";

export interface DocTopic {
  readonly title: string;
  readonly href: string;
}

const DOCS_BASE = "https://docs.litellm.ai";

const topics = (...entries: readonly (readonly [string, string])[]): readonly DocTopic[] =>
  entries.map(([title, path]) => ({ title, href: `${DOCS_BASE}${path}` }));

export const PAGE_DOCS: ReadonlyMap<string, readonly DocTopic[]> = new Map<string, readonly DocTopic[]>([
  ["home", topics(["AI Gateway quickstart", "/docs/proxy/docker_quick_start"], ["Admin UI", "/docs/proxy/ui"])],
  [
    "api-keys",
    topics(
      ["Virtual keys", "/docs/proxy/virtual_keys"],
      ["Restrict model access", "/docs/proxy/model_access"],
      ["Budgets and rate limits", "/docs/proxy/users"],
    ),
  ],
  [
    "models-and-endpoints",
    topics(
      ["Model management", "/docs/proxy/model_management"],
      ["Load balancing", "/docs/proxy/load_balancing"],
      ["Custom pricing", "/docs/proxy/custom_pricing"],
    ),
  ],
  [
    "api-reference",
    topics(["Use your virtual key", "/docs/proxy/user_keys"], ["Connect clients", "/docs/proxy/client_setup/overview"]),
  ],
  ["playground", topics(["Chat UI", "/docs/proxy/ui/chat_ui"], ["Compare models", "/docs/proxy/model_compare_ui"])],
  ["projects", topics(["Projects", "/docs/proxy/project_management"])],
  [
    "access-groups",
    topics(["Access groups", "/docs/proxy/access_groups"], ["Model access groups", "/docs/proxy/model_access_groups"]),
  ],
  [
    "budgets",
    topics(
      ["Budgets and rate limits", "/docs/proxy/users"],
      ["Rate limit tiers", "/docs/proxy/rate_limit_tiers"],
      ["Budget reset times", "/docs/proxy/budget_reset_and_tz"],
    ),
  ],
  [
    "guardrails",
    topics(
      ["Guardrails quick start", "/docs/proxy/guardrails/quick_start"],
      ["Team guardrails", "/docs/proxy/guardrails/team_based_guardrails"],
      ["Guardrail playground", "/docs/proxy/guardrails/test_playground"],
    ),
  ],
  [
    "guardrails-monitor",
    topics(
      ["Guardrails quick start", "/docs/proxy/guardrails/quick_start"],
      ["Team guardrails", "/docs/proxy/guardrails/team_based_guardrails"],
      ["Guardrail playground", "/docs/proxy/guardrails/test_playground"],
    ),
  ],
  [
    "policies",
    topics(
      ["Guardrail policies", "/docs/proxy/guardrails/guardrail_policies"],
      ["Policy templates", "/docs/proxy/guardrails/policy_templates"],
    ),
  ],
  [
    "mcp-servers",
    topics(
      ["MCP gateway", "/docs/mcp"],
      ["MCP authentication", "/docs/mcp_authentication"],
      ["MCP access control", "/docs/mcp_control"],
    ),
  ],
  ["search-tools", topics(["Search tools in the UI", "/docs/proxy/ui_search_tools"], ["Search API", "/docs/search"])],
  [
    "tag-management",
    topics(
      ["Tag routing", "/docs/proxy/tag_routing"],
      ["Tag budgets", "/docs/proxy/tag_budgets"],
      ["Request tags", "/docs/proxy/request_tags"],
    ),
  ],
  [
    "vector-stores",
    topics(["Vector stores", "/docs/vector_stores"], ["Create a vector store", "/docs/vector_stores/create"]),
  ],
  ["memory", topics(["Memory", "/docs/proxy/memory"])],
  [
    "prompts",
    topics(
      ["Prompt management", "/docs/proxy/prompt_management"],
      ["LiteLLM prompt management", "/docs/proxy/litellm_prompt_management"],
    ),
  ],
  ["tool-policies", topics(["Tool policies", "/docs/proxy/tool_policies"])],
  ["skills", topics(["Skills gateway", "/docs/skills_gateway"])],
  [
    "caching",
    topics(
      ["Caching", "/docs/proxy/caching"],
      ["Cache controls", "/docs/proxy/caching_controls"],
      ["Prompt caching", "/docs/completion/prompt_caching"],
    ),
  ],
  [
    "cost-tracking",
    topics(
      ["Spend tracking", "/docs/proxy/cost_tracking"],
      ["Custom pricing", "/docs/proxy/custom_pricing"],
      ["Provider discounts", "/docs/proxy/provider_discounts"],
    ),
  ],
  ["usage", topics(["Spend tracking", "/docs/proxy/cost_tracking"], ["Billing", "/docs/proxy/billing"])],
  ["old-usage", topics(["Spend tracking", "/docs/proxy/cost_tracking"], ["Billing", "/docs/proxy/billing"])],
  [
    "cost-optimization",
    topics(["Prompt compression", "/docs/proxy/headroom"], ["Prompt caching", "/docs/completion/prompt_caching"]),
  ],
  ["logs", topics(["Request logs", "/docs/proxy/ui_logs"], ["Sessions", "/docs/proxy/ui_logs_sessions"])],
  ["lens", topics(["Lens", "/docs/proxy/lens"], ["Deploy Lens", "/docs/proxy/lens/deployment"])],
  [
    "admin-panel",
    topics(["SSO for Admin UI", "/docs/proxy/admin_ui_sso"], ["Role-based access", "/docs/proxy/access_control"]),
  ],
  [
    "logging-and-alerts",
    topics(
      ["Logging", "/docs/proxy/logging"],
      ["Alerting", "/docs/proxy/alerting"],
      ["Team logging", "/docs/proxy/team_logging"],
    ),
  ],
  [
    "telemetry",
    topics(
      ["Prometheus", "/docs/proxy/prometheus"],
      ["OpenTelemetry", "/docs/observability/opentelemetry_integration"],
    ),
  ],
  ["agents", topics(["A2A agents", "/docs/a2a"], ["Agent permissions", "/docs/a2a_agent_permissions"])],
  [
    "router-settings",
    topics(
      ["Routing", "/docs/routing"],
      ["Fallbacks and retries", "/docs/proxy/reliability"],
      ["Load balancing", "/docs/proxy/load_balancing"],
    ),
  ],
  [
    "users",
    topics(["Budgets and rate limits", "/docs/proxy/users"], ["Role-based access", "/docs/proxy/access_control"]),
  ],
  [
    "teams",
    topics(
      ["Team budgets", "/docs/proxy/team_budgets"],
      ["Let teams add models", "/docs/proxy/team_model_add"],
      ["Team logging", "/docs/proxy/team_logging"],
    ),
  ],
  [
    "organizations",
    topics(
      ["Role-based access", "/docs/proxy/access_control"],
      ["User management hierarchy", "/docs/proxy/user_management_heirarchy"],
    ),
  ],
]);

export function pageDocsFor(pathname: string): readonly DocTopic[] {
  return PAGE_DOCS.get(routeSegmentForPathname(pathname) || "home") ?? [];
}
