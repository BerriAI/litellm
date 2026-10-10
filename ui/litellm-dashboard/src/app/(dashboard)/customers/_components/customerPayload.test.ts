import { describe, expect, it } from "vitest";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { buildCustomerPayload, toCustomerFormValues } from "./customerPayload";

const customer: EndUser = { user_id: "customer-id", blocked: false, spend: 0 };
const permission = {
  object_permission_id: "permission-id",
  mcp_servers: ["server-id"],
  mcp_access_groups: ["group-id"],
  mcp_toolsets: ["toolset-id"],
  mcp_tool_permissions: { "server-id": ["read", "write"] },
  agent_access_groups: [],
  agents: [],
  blocked_tools: [],
  models: [],
  search_tools: [],
  skills: [],
  vector_stores: [],
};

const server: MCPServer = {
  server_id: "server-id",
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
};

describe("customer form values", () => {
  it("starts with empty optional values and grants", () => {
    const expected = {
      user_id: "",
      alias: "",
      budget_id: "",
      max_budget: "",
      models: [],
      allowed_model_region: "",
      default_model: "",
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
      mcp_tool_permissions: {},
    };
    expect(toCustomerFormValues(null)).toEqual(expected);
  });

  it("loads the stored identity, models and MCP grant", () => {
    const stored: EndUser = {
      ...customer,
      alias: "Alias",
      budget_id: "budget-id",
      models: ["model-id"],
      allowed_model_region: "eu",
      default_model: "default-model",
      object_permission: permission,
      litellm_budget_table: { max_budget: 50, created_at: "2026-01-01T00:00:00Z" },
    };
    const expected = {
      user_id: stored.user_id,
      alias: stored.alias,
      budget_id: stored.budget_id,
      max_budget: "",
      models: stored.models,
      allowed_model_region: stored.allowed_model_region,
      default_model: stored.default_model,
      mcp_servers_and_groups: {
        servers: permission.mcp_servers,
        accessGroups: permission.mcp_access_groups,
        toolsets: permission.mcp_toolsets,
      },
      mcp_tool_permissions: permission.mcp_tool_permissions,
    };
    expect(toCustomerFormValues(stored)).toEqual(expected);
  });
});

describe("customer payload", () => {
  it("creates a customer with only the fields that are set", () => {
    const values = { ...toCustomerFormValues(null), user_id: " customer-id " };
    expect(buildCustomerPayload(values, null, [], [])).toEqual({ user_id: customer.user_id });
  });

  it("keeps zero budgets and trims optional text fields", () => {
    const values = {
      ...toCustomerFormValues(null),
      user_id: customer.user_id,
      alias: " Alias ",
      max_budget: "0",
      models: ["model-id"],
      allowed_model_region: "us" as const,
      default_model: " default-model ",
    };
    const expected = {
      user_id: customer.user_id,
      alias: "Alias",
      max_budget: 0,
      models: values.models,
      allowed_model_region: values.allowed_model_region,
      default_model: "default-model",
    };
    expect(buildCustomerPayload(values, null, [], [])).toEqual(expected);
  });

  it("sends a selected existing budget", () => {
    const values = { ...toCustomerFormValues(null), user_id: customer.user_id, budget_id: "budget-id" };
    expect(buildCustomerPayload(values, null, [], [])).toEqual({
      user_id: customer.user_id,
      budget_id: values.budget_id,
    });
  });

  it.each(["", "   "])("clears a stored alias when the form contains %j", (alias) => {
    const stored = { ...customer, alias: "Alias" };
    const values = { ...toCustomerFormValues(stored), alias };

    expect(buildCustomerPayload(values, stored, [], [])).toEqual({
      user_id: stored.user_id,
      alias: "",
      models: [],
    });
  });

  it("includes the grant when a toolset is chosen on create", () => {
    const values = {
      ...toCustomerFormValues(null),
      user_id: customer.user_id,
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: ["toolset-id"] },
    };
    const grant = {
      mcp_servers: [],
      mcp_access_groups: [],
      mcp_toolsets: values.mcp_servers_and_groups.toolsets,
      mcp_tool_permissions: {},
    };
    expect(buildCustomerPayload(values, null, [], [])).toEqual({ user_id: customer.user_id, object_permission: grant });
  });

  it("omits the stored MCP grant on an unrelated edit, regardless of ordering", () => {
    const stored = {
      ...customer,
      object_permission: {
        ...permission,
        mcp_servers: ["server-id", "another-server"],
        mcp_access_groups: ["group-id", "another-group"],
        mcp_toolsets: ["toolset-id", "another-toolset"],
        mcp_tool_permissions: { "server-id": ["read", "write"], "another-server": ["list"] },
      },
    };
    const values = {
      ...toCustomerFormValues(stored),
      alias: "New alias",
      mcp_servers_and_groups: {
        servers: ["another-server", "server-id"],
        accessGroups: ["another-group", "group-id"],
        toolsets: ["another-toolset", "toolset-id"],
      },
      mcp_tool_permissions: { "another-server": ["list"], "server-id": ["write", "read"] },
    };
    expect(buildCustomerPayload(values, stored, [server], [])).toEqual({
      user_id: customer.user_id,
      alias: values.alias,
      models: [],
    });
  });

  it("keeps stored tool restrictions untouched when only the alias changes", () => {
    const stored = {
      ...customer,
      object_permission: { ...permission, mcp_servers: [], mcp_access_groups: [], mcp_toolsets: [] },
    };
    const values = { ...toCustomerFormValues(stored), alias: "New alias" };
    expect(buildCustomerPayload(values, stored, [server], [])).toEqual({
      user_id: customer.user_id,
      alias: values.alias,
      models: [],
    });
  });

  it("saves edits to a server granted only through its tool permissions", () => {
    const stored = {
      ...customer,
      object_permission: {
        ...permission,
        mcp_servers: [],
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: { "server-id": ["read"] },
      },
    };
    const values = { ...toCustomerFormValues(stored), mcp_tool_permissions: { "server-id": ["write"] } };
    const grant = {
      mcp_servers: [],
      mcp_access_groups: [],
      mcp_toolsets: [],
      mcp_tool_permissions: values.mcp_tool_permissions,
    };

    expect(buildCustomerPayload(values, stored, [server], [])).toEqual({
      user_id: customer.user_id,
      models: [],
      object_permission: grant,
    });
  });

  it.each([
    { kind: "server", allowedTools: ["read"] },
    { kind: "toolset", allowedTools: ["write"] },
  ])("keeps an independent tool grant when adding a $kind", ({ kind, allowedTools }) => {
    const stored = {
      ...customer,
      object_permission: {
        ...permission,
        mcp_servers: [],
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: { "server-id": ["read"] },
      },
    };
    const anotherServer = { ...server, server_id: "another-server" };
    const toolsets: MCPToolset[] = [
      {
        toolset_id: "toolset-id",
        toolset_name: "Toolset",
        tools: [{ server_id: anotherServer.server_id, tool_name: "list" }],
      },
    ];
    const selection = {
      servers: kind === "server" ? [anotherServer.server_id] : [],
      accessGroups: [],
      toolsets: kind === "toolset" ? [toolsets[0].toolset_id] : [],
    };
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: selection,
      mcp_tool_permissions: { "server-id": allowedTools },
    };
    const grant = {
      mcp_servers: selection.servers,
      mcp_access_groups: [],
      mcp_toolsets: selection.toolsets,
      mcp_tool_permissions: values.mcp_tool_permissions,
    };

    expect(buildCustomerPayload(values, stored, [server, anotherServer], toolsets)).toEqual({
      user_id: customer.user_id,
      models: [],
      object_permission: grant,
    });
  });

  it("keeps a canonical server edit from an independent shared-alias grant when adding a server", () => {
    const stored = {
      ...customer,
      object_permission: {
        ...permission,
        mcp_servers: [],
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: { wiki: ["read"] },
      },
    };
    const sharedServer = { ...server, alias: "wiki" };
    const anotherSharedServer = { ...sharedServer, server_id: "another-shared-server" };
    const addedServer = { ...server, server_id: "added-server" };
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [addedServer.server_id], accessGroups: [], toolsets: [] },
      mcp_tool_permissions: { wiki: ["read"], [sharedServer.server_id]: ["write"] },
    };
    const grant = {
      mcp_servers: values.mcp_servers_and_groups.servers,
      mcp_access_groups: [],
      mcp_toolsets: [],
      mcp_tool_permissions: values.mcp_tool_permissions,
    };

    expect(buildCustomerPayload(values, stored, [sharedServer, anotherSharedServer, addedServer], [])).toEqual({
      user_id: customer.user_id,
      models: [],
      object_permission: grant,
    });
  });

  it("sends an empty models list when clearing the selection on update", () => {
    const stored = { ...customer, models: ["model-id"] };
    const values = { ...toCustomerFormValues(stored), models: [] };
    expect(buildCustomerPayload(values, stored, [], [])).toEqual({ user_id: customer.user_id, models: [] });
  });

  it("drops the tool restriction for a deselected server", () => {
    const stored = { ...customer, object_permission: { ...permission, mcp_access_groups: [], mcp_toolsets: [] } };
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    };
    const grant = { mcp_servers: [], mcp_access_groups: [], mcp_toolsets: [], mcp_tool_permissions: {} };
    expect(buildCustomerPayload(values, stored, [server], [])).toEqual({
      user_id: customer.user_id,
      models: [],
      object_permission: grant,
    });
  });
});
