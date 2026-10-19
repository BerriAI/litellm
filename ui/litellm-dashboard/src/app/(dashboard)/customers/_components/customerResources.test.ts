import { describe, expect, it } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { customerMcpScope, getCustomerResourceCounts, type CustomerResourceCounts } from "./customerResources";

const emptyPermission: NonNullable<EndUser["object_permission"]> = {
  object_permission_id: "permission-id",
  agent_access_groups: [],
  agents: [],
  blocked_tools: [],
  mcp_servers: [],
  mcp_access_groups: [],
  mcp_toolsets: [],
  mcp_tool_permissions: {},
  models: [],
  search_tools: [],
  vector_stores: [],
};

const customer: EndUser = { user_id: "customer-id", spend: 0, blocked: false };
const noMcpCounts: CustomerResourceCounts = { models: "all", servers: 0, accessGroups: 0, toolsets: 0 };

describe("getCustomerResourceCounts", () => {
  it("counts distinct model and MCP grants", () => {
    const configured: EndUser = {
      ...customer,
      models: ["model-a", "model-b", "model-a"],
      object_permission: {
        ...emptyPermission,
        mcp_servers: ["server-a", "server-b", "server-a"],
        mcp_access_groups: ["group-a", "group-b", "group-a"],
        mcp_toolsets: ["toolset-a", "toolset-a"],
      },
    };
    const expectedCounts = { models: 2, servers: 2, accessGroups: 2, toolsets: 1 };
    expect(getCustomerResourceCounts(configured)).toEqual(expectedCounts);
  });

  it.each([undefined, [], ["all-proxy-models"], ["model-a", "all-proxy-models"], ["*"]])(
    "labels unrestricted models as all (%j)",
    (models) => {
      expect(getCustomerResourceCounts({ ...customer, models }).models).toBe("all");
    },
  );

  it("counts tool-permission servers without counting a server twice", () => {
    const configured = {
      ...customer,
      object_permission: {
        ...emptyPermission,
        mcp_servers: ["server-a"],
        mcp_tool_permissions: { "server-a": ["read", "write"], "server-b": ["read"] },
      },
    };
    expect(getCustomerResourceCounts(configured).servers).toBe(2);
    expect(customerMcpScope(configured.object_permission)).toBe("limited");
  });

  it("excludes the unsupported all-MCP sentinel from customer server counts", () => {
    const configured = {
      ...customer,
      object_permission: { ...emptyPermission, mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL, "server-a"] },
    };
    expect(getCustomerResourceCounts(configured).servers).toBe(1);
    expect(customerMcpScope(configured.object_permission)).toBe("limited");
  });

  it("does not claim MCP access from unsupported all-MCP sentinel-only grants", () => {
    const configured = {
      ...customer,
      object_permission: {
        ...emptyPermission,
        mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL],
        mcp_tool_permissions: { [ALL_PROXY_MCP_SERVERS_SENTINEL]: ["read"] },
      },
    };
    expect(getCustomerResourceCounts(configured).servers).toBe(0);
    expect(customerMcpScope(configured.object_permission)).toBe("denied");
  });

  it("zeroes every MCP grant when no-MCP overrides other grants", () => {
    const configured = {
      ...customer,
      object_permission: {
        ...emptyPermission,
        mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL, "server-a"],
        mcp_access_groups: ["group-a"],
        mcp_toolsets: ["toolset-a"],
        mcp_tool_permissions: { "server-a": ["read"] },
      },
    };
    expect(getCustomerResourceCounts(configured)).toEqual(noMcpCounts);
    expect(customerMcpScope(configured.object_permission)).toBe("denied");
  });

  it("excludes permission-table models and resources not enforced for customers", () => {
    const configured = {
      ...customer,
      models: ["customer-model"],
      object_permission: {
        ...emptyPermission,
        models: ["permission-model-a", "permission-model-b"],
        vector_stores: ["store-id"],
        agents: ["agent-id"],
        agent_access_groups: ["agent-group-id"],
        search_tools: ["search-tool-id"],
        skills: ["skill-id"],
      },
    };
    expect(getCustomerResourceCounts(configured)).toEqual({ ...noMcpCounts, models: 1 });
    expect(customerMcpScope(configured.object_permission)).toBe("unrestricted");
  });
});

describe("customerMcpScope", () => {
  it.each([
    undefined,
    null,
    emptyPermission,
    { ...emptyPermission, mcp_servers: null, mcp_access_groups: null, mcp_toolsets: null },
  ])("treats an empty customer-level grant as unrestricted (%j)", (object_permission) => {
    const unconfigured = { ...customer, object_permission };
    expect(customerMcpScope(object_permission)).toBe("unrestricted");
    expect(getCustomerResourceCounts(unconfigured)).toEqual(noMcpCounts);
  });

  it.each([
    { mcp_servers: ["server-id"] },
    { mcp_access_groups: ["group-id"] },
    { mcp_toolsets: ["toolset-id"] },
    { mcp_tool_permissions: { "server-id": ["read"] } },
  ])("recognizes a grant from each MCP source (%j)", (grant) => {
    expect(customerMcpScope({ ...emptyPermission, ...grant })).toBe("limited");
  });
});
