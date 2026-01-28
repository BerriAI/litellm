/**
 * Page metadata for UI Settings configuration
 * This file contains descriptions and metadata for all navigation pages
 */

// Page descriptions for UI Settings configuration
export const pageDescriptions: Record<string, string> = {
  "api-keys": "Manage virtual keys for API access and authentication",
  playground: "Interactive playground for testing LLM requests",
  "models-and-endpoints": "Configure and manage LLM models and endpoints",
  agents: "Create and manage AI agents",
  "mcp-servers": "Configure Model Context Protocol servers",
  guardrails: "Set up content moderation and safety guardrails",
  policies: "Define access control and usage policies",
  "search-tools": "Configure RAG search and retrieval tools",
  "vector-stores": "Manage vector databases for embeddings",
  usage: "View usage analytics and metrics",
  logs: "Access request and response logs",
  "internal-users": "Manage internal user accounts and permissions",
  teams: "Create and manage teams for access control",
  organizations: "Manage organizations and their members",
  budgets: "Set and monitor spending budgets",
  "api-reference": "Browse API documentation and endpoints",
  "ai-hub": "Explore available AI models and providers",
  "learning-resources": "Access tutorials and documentation",
  caching: "Configure response caching settings",
  "transform-request": "Set up request transformation rules",
  "pass-through-endpoints": "Configure pass-through API endpoints",
  "cost-tracking": "Track and analyze API costs",
  "ui-themes": "Customize dashboard appearance",
  "tag-management": "Organize resources with tags",
  prompts: "Manage and version prompt templates",
  "claude-code-plugins": "Configure Claude Code plugins",
};

export interface PageMetadata {
  page: string;
  label: string;
  group: string;
  description: string;
}
