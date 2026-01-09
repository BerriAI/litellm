import { Page } from "./pages";

/**
 * Maps sidebar menu item labels to their corresponding page enum values.
 * This mapping is for the admin role.
 */
export const menuLabelToPage: Record<string, Page> = {
  "Virtual Keys": Page.ApiKeys,
  Playground: Page.LlmPlayground,
  Models: Page.Models,
  "Models + Endpoints": Page.Models,
  Usage: Page.NewUsage,
  Teams: Page.Teams,
  "Internal Users": Page.Users,
  "Internal User": Page.Users, // Legacy label support
  Organizations: Page.Organizations,
  "API Reference": Page.ApiRef,
  "AI Hub": Page.ModelHubTable,
  "Model Hub": Page.ModelHubTable,
  Logs: Page.Logs,
  Guardrails: Page.Guardrails,
  // Settings submenu items
  "Router Settings": Page.RouterSettings,
  "Logging & Alerts": Page.LoggingAndAlerts,
  "Admin Settings": Page.AdminPanel,
  "Cost Tracking": Page.CostTracking,
  "UI Theme": Page.UiTheme,
  // Experimental submenu items
  Caching: Page.Caching,
  Prompts: Page.Prompts,
  Budgets: Page.Budgets,
  "API Playground": Page.TransformRequest,
  "Tag Management": Page.TagManagement,
  "Old Usage": Page.Usage,
  // Tools submenu items
  "MCP Servers": Page.McpServers,
  "Vector Stores": Page.VectorStores,
};
