import { describe, expect, it } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { customerMatchesFilter, customerMatchesSearch, formatCustomerFilterValue } from "./customerFilters";

const customer: EndUser = { user_id: "customer-1", alias: "Research", blocked: false, spend: 0 };
const permission: NonNullable<EndUser["object_permission"]> = {
  object_permission_id: "permission-1",
  mcp_servers: [],
  mcp_access_groups: [],
  mcp_toolsets: [],
  mcp_tool_permissions: {},
  agent_access_groups: [],
  agents: [],
  models: [],
  blocked_tools: [],
  search_tools: [],
  vector_stores: [],
};

const MCP_SCOPES = ["limited", "denied", "unrestricted"] as const;

const scopesMatching = (row: EndUser): string[] =>
  MCP_SCOPES.filter((scope) => customerMatchesFilter(row, "mcp_access", scope));

describe("customer filters", () => {
  it("searches ID and alias case-insensitively and trims the search", () => {
    expect(customerMatchesSearch(customer, " CUSTOMER-1 ")).toBe(true);
    expect(customerMatchesSearch(customer, "research")).toBe(true);
    expect(customerMatchesSearch({ ...customer, alias: null }, "research")).toBe(false);
    expect(customerMatchesSearch(customer, " ")).toBe(true);
  });

  it.each([false, true])("filters both blocked and active for blocked=%s", (blocked) => {
    const row = { ...customer, blocked };
    expect(customerMatchesFilter(row, "blocked", "blocked")).toBe(blocked);
    expect(customerMatchesFilter(row, "blocked", "active")).toBe(!blocked);
  });

  it.each([0, 25, null, undefined])("uses a monetary limit for max_budget=%s", (max_budget) => {
    const row = { ...customer, litellm_budget_table: { created_at: "2026-01-01T00:00:00Z", max_budget } };
    expect(customerMatchesFilter(row, "max_budget", "yes")).toBe(max_budget != null);
    expect(customerMatchesFilter(row, "max_budget", "no")).toBe(max_budget == null);
  });

  it("treats absent budgets and an unlimited budget link as no monetary budget", () => {
    expect(customerMatchesFilter(customer, "max_budget", "no")).toBe(true);
    expect(customerMatchesFilter({ ...customer, budget_id: "budget-id" }, "max_budget", "yes")).toBe(false);
  });

  it.each([
    { mcp_servers: ["server-1"] },
    { mcp_access_groups: ["group-1"] },
    { mcp_toolsets: ["toolset-1"] },
    { mcp_tool_permissions: { "server-1": ["read"] } },
  ])("detects configured MCP grants %j", (grant) => {
    const row = { ...customer, object_permission: { ...permission, ...grant } };
    expect(scopesMatching(row)).toEqual(["limited"]);
  });

  it.each([
    { mcp_servers: [NO_MCP_SERVERS_SENTINEL], mcp_access_groups: ["group-1"] },
    { mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL] },
  ])("matches denied MCP grants only as No MCP access %j", (grant) => {
    expect(scopesMatching({ ...customer, object_permission: { ...permission, ...grant } })).toEqual(["denied"]);
  });

  it.each([undefined, null, permission])(
    "matches customers without a grant only as No MCP restriction (%j)",
    (object_permission) => {
      expect(scopesMatching({ ...customer, object_permission })).toEqual(["unrestricted"]);
    },
  );

  it("ignores invalid URL filter values and unrelated columns", () => {
    expect(customerMatchesFilter(customer, "blocked", "unexpected")).toBe(true);
    expect(customerMatchesFilter(customer, "max_budget", false)).toBe(true);
    expect(customerMatchesFilter(customer, "mcp_access", "unexpected")).toBe(true);
    expect(customerMatchesFilter(customer, "mcp_access", "yes")).toBe(true);
    expect(customerMatchesFilter(customer, "unrelated", "yes")).toBe(true);
  });

  it("formats filter chips with the labels shown in the drawer", () => {
    expect(formatCustomerFilterValue("blocked", "active")).toBe("Active");
    expect(formatCustomerFilterValue("max_budget", "yes")).toBe("Has budget");
    expect(formatCustomerFilterValue("mcp_access", "unrestricted")).toBe("No MCP restriction");
  });
});
