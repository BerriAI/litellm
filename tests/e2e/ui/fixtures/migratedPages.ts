/**
 * Source of truth for the App Router migration E2E suites.
 *
 * Add an entry (legacy sidebar page id -> route segment) once a page's migration
 * has MERGED to the branch under test. Consumers pick it up automatically:
 *   - migration smoke (tests/migration/migratedPages.spec.ts), via MIGRATED_E2E_SEGMENTS:
 *       default mount:           npm run e2e:migration
 *       server-root-path mount:  SERVER_ROOT_PATH=/<root> npm run e2e:migration:root
 *   - navigation specs that assert per-page URLs (tests/navigation/sidebar.spec.ts)
 *
 * Keep this in lockstep with MIGRATED_PAGES in src/utils/migratedPages.ts.
 */
export const MIGRATED_E2E_PAGES: Record<string, string> = {
  "api-keys": "api-keys",
  models: "models-and-endpoints",
  api_ref: "api-reference",
  "llm-playground": "playground",
  projects: "projects",
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
  agents: "agents",
  "router-settings": "router-settings",
  users: "users",
  teams: "teams",
  organizations: "organizations",
};

export const MIGRATED_E2E_SEGMENTS: string[] = [...new Set(Object.values(MIGRATED_E2E_PAGES))];
