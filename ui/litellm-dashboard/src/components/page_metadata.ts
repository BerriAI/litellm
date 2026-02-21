/**
 * Page metadata for UI Settings configuration
 * This file contains descriptions and metadata for all navigation pages
 */

// Page descriptions for UI Settings configuration
export const pageDescriptions: Record<string, string> = {
  "api-keys": "Manage virtual keys for API access and authentication",
  "llm-playground": "Interactive playground for testing LLM requests",
  models: "Configure and manage LLM models and endpoints",
  agents: "Create and manage AI agents",
  "mcp-servers": "Configure Model Context Protocol servers",
  guardrails: "Set up content moderation and safety guardrails",
  policies: "Define access control and usage policies",
  "search-tools": "Configure RAG search and retrieval tools",
  "vector-stores": "Manage vector databases for embeddings",
  new_usage: "View usage analytics and metrics",
  logs: "Access request and response logs",
  users: "Manage internal user accounts and permissions",
  teams: "Create and manage teams for access control",
  organizations: "Manage organizations and their members",
  "access-groups": "Manage access groups for role-based permissions",
  budgets: "Set and monitor spending budgets",
  api_ref: "Browse API documentation and endpoints",
  "model-hub-table": "Explore available AI models and providers",
  "learning-resources": "Access tutorials and documentation",
  caching: "Configure response caching settings",
  "transform-request": "Set up request transformation rules",
  "cost-tracking": "Track and analyze API costs",
  "ui-theme": "Customize dashboard appearance",
  "tag-management": "Organize resources with tags",
  prompts: "Manage and version prompt templates",
  "claude-code-plugins": "Configure Claude Code plugins",
  usage: "View legacy usage dashboard",
  "router-settings": "Configure routing and load balancing settings",
  "logging-and-alerts": "Set up logging and alert configurations",
  "admin-panel": "Access admin panel and settings",
};

export interface PageMetadata {
  page: string;
  label: string;
  group: string;
  description: string;
}
