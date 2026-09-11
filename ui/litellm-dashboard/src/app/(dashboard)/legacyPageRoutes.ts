import { uiHref } from "@/utils/uiHref";

const LEGACY_PAGE_ROUTES: ReadonlyMap<string, string> = new Map(
  Object.entries({
    "api-keys": "api-keys",
    models: "models-and-endpoints",
    api_ref: "api-reference",
    "api-reference": "api-reference",
    "llm-playground": "playground",
    projects: "projects",
    chat: "chat",
    "access-groups": "access-groups",
    budgets: "budgets",
    workflows: "workflows",
    "guardrails-monitor": "guardrails-monitor",
    "mcp-servers": "mcp-servers",
    "search-tools": "search-tools",
    "tag-management": "tag-management",
    "vector-stores": "vector-stores",
    memory: "memory",
    policies: "policies",
    guardrails: "guardrails",
    prompts: "prompts",
    "tool-policies": "tool-policies",
    skills: "skills",
    "claude-code-plugins": "skills",
    caching: "caching",
    "cost-tracking": "cost-tracking",
    "transform-request": "transform-request",
    "ui-theme": "ui-theme",
    logs: "logs",
    "admin-panel": "admin-panel",
    "logging-and-alerts": "logging-and-alerts",
    "model-hub-table": "model-hub-table",
    new_usage: "usage",
    usage: "old-usage",
    "cost-optimization": "cost-optimization",
    agents: "agents",
    "router-settings": "router-settings",
    users: "users",
    teams: "teams",
    organizations: "organizations",
  }),
);

export function legacyPageRedirectHref(searchParams: URLSearchParams): string | null {
  const page = searchParams.get("page");
  const route = page === null ? undefined : LEGACY_PAGE_ROUTES.get(page);
  if (route === undefined) return null;
  const rest = new URLSearchParams(searchParams);
  rest.delete("page");
  const query = rest.toString();
  return query ? `${uiHref(route)}?${query}` : uiHref(route);
}
