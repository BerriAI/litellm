export type MigratedPage = Readonly<{
  segment: string;
  linkName: string | RegExp;
  group?: string;
  content: Readonly<{ role: "heading" | "tab" | "button"; name: string }> | Readonly<{ text: string }>;
  unlicensedText?: string;
}>;

export const MIGRATED_E2E_PAGES: Readonly<Record<string, MigratedPage>> = {
  "api-keys": { segment: "api-keys", linkName: "Virtual Keys", content: { role: "heading", name: "Virtual Keys" } },
  models: {
    segment: "models-and-endpoints",
    linkName: "Models + Endpoints",
    content: { role: "heading", name: "Model Management" },
  },
  api_ref: {
    segment: "api-reference",
    linkName: "API Reference",
    content: { role: "heading", name: "OpenAI Compatible Proxy: API Reference" },
  },
  "llm-playground": { segment: "playground", linkName: "Playground", content: { role: "tab", name: "Chat" } },
  projects: {
    segment: "projects",
    linkName: /^Projects(?: Beta)?$/,
    content: { role: "heading", name: "Projects" },
  },
  "access-groups": {
    segment: "access-groups",
    linkName: "Access Groups",
    content: { role: "heading", name: "Access Groups" },
  },
  budgets: { segment: "budgets", linkName: "Budgets", content: { role: "heading", name: "Budgets" } },
  workflows: {
    segment: "workflows",
    linkName: "Workflow Runs",
    group: "Agentic",
    content: { text: "Workflow Runs" },
  },
  "guardrails-monitor": {
    segment: "guardrails-monitor",
    linkName: "Guardrails Monitor",
    content: { role: "heading", name: "Guardrails Monitor" },
  },
  "mcp-servers": {
    segment: "mcp-servers",
    linkName: "MCP Servers",
    content: { role: "heading", name: "MCP Servers" },
  },
  "search-tools": {
    segment: "search-tools",
    linkName: "Search Tools",
    group: "Tools",
    content: { role: "heading", name: "Search Tools" },
  },
  "tag-management": {
    segment: "tag-management",
    linkName: "Tag Management",
    group: "Experimental",
    content: { role: "heading", name: "Tag Management" },
  },
  "vector-stores": {
    segment: "vector-stores",
    linkName: "Vector Stores",
    group: "Tools",
    content: { role: "heading", name: "Vector Store Management" },
  },
  memory: { segment: "memory", linkName: "Memory", group: "Agentic", content: { role: "heading", name: "Memory" } },
  policies: { segment: "policies", linkName: "Policies", content: { role: "tab", name: "Policy Simulator" } },
  guardrails: { segment: "guardrails", linkName: "Guardrails", content: { role: "tab", name: "Guardrails" } },
  prompts: {
    segment: "prompts",
    linkName: "Prompts",
    group: "Experimental",
    content: { role: "button", name: "Add New Prompt" },
  },
  "tool-policies": {
    segment: "tool-policies",
    linkName: "Tool Policies",
    group: "Tools",
    content: { role: "heading", name: "Tool Policies" },
  },
  skills: { segment: "skills", linkName: "Skills", content: { role: "heading", name: "Skills" } },
  caching: { segment: "caching", linkName: "Response Cache", content: { role: "tab", name: "Cache Settings" } },
  "cost-tracking": {
    segment: "cost-tracking",
    linkName: "Cost Tracking",
    group: "Settings",
    content: { text: "Cost Tracking Settings" },
  },
  "transform-request": {
    segment: "transform-request",
    linkName: "API Playground",
    group: "Experimental",
    content: { role: "heading", name: "Playground" },
  },
  "ui-theme": {
    segment: "ui-theme",
    linkName: "UI Theme",
    group: "Settings",
    content: { role: "heading", name: "UI Theme Customization" },
  },
  logs: { segment: "logs", linkName: "Logs", content: { role: "heading", name: "Request Logs" } },
  "admin-panel": {
    segment: "admin-panel",
    linkName: "Admin Settings",
    group: "Settings",
    content: { role: "heading", name: "Admin Access" },
  },
  "logging-and-alerts": {
    segment: "logging-and-alerts",
    linkName: "Logging & Alerts",
    group: "Settings",
    content: { role: "tab", name: "Logging Callbacks" },
  },
  "model-hub-table": {
    segment: "model-hub-table",
    linkName: "AI Hub",
    content: { role: "heading", name: "AI Hub" },
  },
  new_usage: { segment: "usage", linkName: "Usage", content: { role: "heading", name: "Usage View" } },
  usage: {
    segment: "old-usage",
    linkName: "Old Usage",
    group: "Experimental",
    content: { role: "tab", name: "All Up" },
  },
  agents: { segment: "agents", linkName: "Agents", group: "Agentic", content: { role: "heading", name: "Agents" } },
  "router-settings": {
    segment: "router-settings",
    linkName: "Router Settings",
    group: "Settings",
    content: { role: "heading", name: "Routing Settings" },
  },
  users: { segment: "users", linkName: "Internal Users", content: { role: "tab", name: "Users" } },
  teams: { segment: "teams", linkName: "Teams", content: { role: "heading", name: "Teams" } },
  organizations: {
    segment: "organizations",
    linkName: "Organizations",
    content: { text: "Click on an organization ID to view its details." },
    unlicensedText: "This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key here.",
  },
};
