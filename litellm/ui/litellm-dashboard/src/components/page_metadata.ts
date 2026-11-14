/**
 * Page metadata for UI Settings configuration
 * This file contains descriptions and metadata for all navigation pages
 */

import type { ParseKeys } from "i18next";

// Page descriptions for UI Settings configuration
export const pageDescriptions: Record<string, ParseKeys> = {
  home: "shell.pageDescriptions.home",
  "api-keys": "shell.pageDescriptions.apiKeys",
  "llm-playground": "shell.pageDescriptions.llmPlayground",
  models: "shell.pageDescriptions.models",
  agents: "shell.pageDescriptions.agents",
  agentic: "shell.pageDescriptions.agentic",
  workflows: "shell.pageDescriptions.workflows",
  "mcp-servers": "shell.pageDescriptions.mcpServers",
  memory: "shell.pageDescriptions.memory",
  guardrails: "shell.pageDescriptions.guardrails",
  policies: "shell.pageDescriptions.policies",
  "search-tools": "shell.pageDescriptions.searchTools",
  "tool-policies": "shell.pageDescriptions.toolPolicies",
  "vector-stores": "shell.pageDescriptions.vectorStores",
  new_usage: "shell.pageDescriptions.newUsage",
  alerts: "shell.pageDescriptions.alerts",
  "cost-optimization": "shell.pageDescriptions.costOptimization",
  logs: "shell.pageDescriptions.logs",
  lens: "shell.pageDescriptions.lens",
  "guardrails-monitor": "shell.pageDescriptions.guardrailsMonitor",
  users: "shell.pageDescriptions.users",
  teams: "shell.pageDescriptions.teams",
  organizations: "shell.pageDescriptions.organizations",
  projects: "shell.pageDescriptions.projects",
  "access-groups": "shell.pageDescriptions.accessGroups",
  budgets: "shell.pageDescriptions.budgets",
  "model-hub-table": "shell.pageDescriptions.modelHubTable",
  caching: "shell.pageDescriptions.caching",
  "transform-request": "shell.pageDescriptions.transformRequest",
  "cost-tracking": "shell.pageDescriptions.costTracking",
  "ui-theme": "shell.pageDescriptions.uiTheme",
  "tag-management": "shell.pageDescriptions.tagManagement",
  prompts: "shell.pageDescriptions.prompts",
  skills: "shell.pageDescriptions.skills",
  "router-settings": "shell.pageDescriptions.routerSettings",
  telemetry: "shell.pageDescriptions.telemetry",
  "logging-and-alerts": "shell.pageDescriptions.loggingAndAlerts",
  "admin-panel": "shell.pageDescriptions.adminPanel",
};

export interface PageMetadata {
  page: string;
  label: string;
  group: string;
  description: string;
}
