import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { toCustomerFormValues } from "./customerPayload";
import { useSaveCustomer } from "./useSaveCustomer";

const mocks = vi.hoisted(() => ({
  servers: vi.fn(),
  toolsets: vi.fn(),
  accessGroups: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({ useMCPServers: mocks.servers }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({ useMCPToolsets: mocks.toolsets }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups", () => ({ useMCPAccessGroups: mocks.accessGroups }));
vi.mock("@/app/(dashboard)/hooks/customers/useCustomerMutations", () => ({
  useCreateCustomer: () => ({ mutateAsync: mocks.create, isPending: false }),
  useUpdateCustomer: () => ({ mutateAsync: mocks.update, isPending: false }),
}));

const permission = {
  object_permission_id: "permission-id",
  mcp_servers: ["server-id"],
  mcp_access_groups: [],
  mcp_toolsets: [],
  mcp_tool_permissions: { "server-id": ["read"] },
  agent_access_groups: [],
  agents: [],
  blocked_tools: [],
  models: [],
  search_tools: [],
  skills: [],
  vector_stores: [],
};
const customer: EndUser = {
  user_id: "customer-id",
  blocked: false,
  spend: 0,
  object_permission: permission,
};
const server: MCPServer = {
  server_id: "server-id",
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
};
const toolset: MCPToolset = {
  toolset_id: "toolset-id",
  toolset_name: "Toolset",
  tools: [{ server_id: server.server_id, tool_name: "read" }],
};
const clearedGrant = { mcp_servers: [], mcp_access_groups: [], mcp_toolsets: [], mcp_tool_permissions: {} };

beforeEach(() => {
  vi.clearAllMocks();
  mocks.servers.mockReturnValue({ data: undefined, isSuccess: false });
  mocks.toolsets.mockReturnValue({ data: undefined, isSuccess: false });
  mocks.accessGroups.mockReturnValue({ data: undefined, isSuccess: false });
  mocks.create.mockResolvedValue(customer);
  mocks.update.mockResolvedValue(customer);
});

describe("useSaveCustomer", () => {
  it.each([
    { state: "pending", data: undefined },
    { state: "failed", data: [server] },
  ])("refuses a server revocation while the catalog is $state", async ({ data }) => {
    mocks.servers.mockReturnValue({ data, isSuccess: false });
    const { result } = renderHook(() => useSaveCustomer(customer));
    const values = {
      ...toCustomerFormValues(customer),
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow("MCP catalogs must finish loading");
    expect(mocks.update).not.toHaveBeenCalled();
  });

  it("saves an ordinary edit without catalogs and leaves unknown stored grants untouched", async () => {
    const stored: EndUser = {
      ...customer,
      object_permission: {
        ...permission,
        mcp_servers: ["unknown-server"],
        mcp_tool_permissions: { "unknown-server": ["read"] },
      },
    };
    const { result } = renderHook(() => useSaveCustomer(stored));

    await result.current.saveCustomer({ ...toCustomerFormValues(stored), alias: "New alias" });

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith({ user_id: stored.user_id, alias: "New alias", models: [] });
  });

  it("revokes a direct server once its catalog loads, without requiring an unused toolset catalog", async () => {
    mocks.servers.mockReturnValue({ data: [server], isSuccess: true });
    const { result } = renderHook(() => useSaveCustomer(customer));
    const values = {
      ...toCustomerFormValues(customer),
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    };

    await result.current.saveCustomer(values);

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith({
      user_id: customer.user_id,
      models: [],
      object_permission: clearedGrant,
    });
  });

  it("refuses a revocation with an empty final permission map until the server catalog loads", async () => {
    const stored: EndUser = { ...customer, object_permission: { ...permission, mcp_tool_permissions: {} } };
    const { result } = renderHook(() => useSaveCustomer(stored));
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow("MCP catalogs must finish loading");
    expect(mocks.update).not.toHaveBeenCalled();
  });

  it("refuses a new customer's server grant until the server catalog loads", async () => {
    const { result } = renderHook(() => useSaveCustomer(null));
    const values = {
      ...toCustomerFormValues(null),
      user_id: customer.user_id,
      mcp_servers_and_groups: { servers: [server.server_id], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow("MCP catalogs must finish loading");
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it.each([
    { state: "pending", data: undefined },
    { state: "failed", data: [toolset] },
  ])("refuses a toolset revocation while its catalog is $state, then saves after it loads", async ({ data }) => {
    const stored: EndUser = {
      ...customer,
      object_permission: { ...permission, mcp_servers: [], mcp_toolsets: [toolset.toolset_id] },
    };
    mocks.servers.mockReturnValue({ data: [server], isSuccess: true });
    mocks.toolsets.mockReturnValue({ data, isSuccess: false });
    const { result, rerender } = renderHook(() => useSaveCustomer(stored));
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow("MCP catalogs must finish loading");
    expect(mocks.update).not.toHaveBeenCalled();
    mocks.toolsets.mockReturnValue({ data: [toolset], isSuccess: true });
    rerender();

    await result.current.saveCustomer(values);

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith({
      user_id: stored.user_id,
      models: [],
      object_permission: clearedGrant,
    });
  });

  it("refuses a new direct server until it has an allowlist, then saves the allowlist", async () => {
    const added: MCPServer = { ...server, server_id: "added-server" };
    mocks.servers.mockReturnValue({ data: [server, added], isSuccess: true });
    const { result } = renderHook(() => useSaveCustomer(customer));
    const values = {
      ...toCustomerFormValues(customer),
      mcp_servers_and_groups: { servers: [server.server_id, added.server_id], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow(
      "Choose allowed tools for each MCP server before saving",
    );
    expect(mocks.update).not.toHaveBeenCalled();

    await result.current.saveCustomer({
      ...values,
      mcp_tool_permissions: { ...values.mcp_tool_permissions, [added.server_id]: ["read"] },
    });

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith({
      user_id: customer.user_id,
      models: [],
      object_permission: {
        mcp_servers: [server.server_id, added.server_id],
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: { [server.server_id]: ["read"], [added.server_id]: ["read"] },
      },
    });
  });

  it("refuses removing the toolset that limits a stored direct server's tools until it has an allowlist", async () => {
    const stored: EndUser = {
      ...customer,
      object_permission: { ...permission, mcp_toolsets: [toolset.toolset_id], mcp_tool_permissions: {} },
    };
    mocks.servers.mockReturnValue({ data: [server], isSuccess: true });
    mocks.toolsets.mockReturnValue({ data: [toolset], isSuccess: true });
    const { result } = renderHook(() => useSaveCustomer(stored));
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [server.server_id], accessGroups: [], toolsets: [] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow(
      "Choose allowed tools for each MCP server before saving",
    );
    expect(mocks.update).not.toHaveBeenCalled();

    await result.current.saveCustomer({ ...values, mcp_tool_permissions: { [server.server_id]: ["read"] } });

    expect(mocks.update).toHaveBeenCalledOnce();
  });

  it("refuses a permission edit while a stored access group's catalog has failed, then saves after it loads", async () => {
    const added: MCPServer = { ...server, server_id: "added-server" };
    const stored: EndUser = { ...customer, object_permission: { ...permission, mcp_access_groups: ["group-a"] } };
    mocks.servers.mockReturnValue({ data: [server, added], isSuccess: true });
    mocks.accessGroups.mockReturnValue({ data: undefined, isSuccess: false });
    const { result, rerender } = renderHook(() => useSaveCustomer(stored));
    const values = {
      ...toCustomerFormValues(stored),
      mcp_servers_and_groups: { servers: [server.server_id, added.server_id], accessGroups: ["group-a"], toolsets: [] },
      mcp_tool_permissions: { [server.server_id]: ["read"], [added.server_id]: ["read"] },
    };

    await expect(result.current.saveCustomer(values)).rejects.toThrow("MCP catalogs must finish loading");
    expect(mocks.update).not.toHaveBeenCalled();
    mocks.accessGroups.mockReturnValue({ data: ["group-a"], isSuccess: true });
    rerender();

    await result.current.saveCustomer(values);

    expect(mocks.update).toHaveBeenCalledOnce();
  });

  it.each([
    { case: "an unrestricted stored server", stored: ["added-server"], toolsetServer: "server-id" },
    { case: "a server its selected toolset covers", stored: [], toolsetServer: "added-server" },
  ])("saves $case without an allowlist", async ({ stored, toolsetServer }) => {
    const added: MCPServer = { ...server, server_id: "added-server" };
    const addedToolset: MCPToolset = { ...toolset, tools: [{ server_id: toolsetServer, tool_name: "read" }] };
    const existing: EndUser = {
      ...customer,
      object_permission: { ...permission, mcp_servers: stored, mcp_tool_permissions: {} },
    };
    mocks.servers.mockReturnValue({ data: [server, added], isSuccess: true });
    mocks.toolsets.mockReturnValue({ data: [addedToolset], isSuccess: true });
    const { result } = renderHook(() => useSaveCustomer(existing));

    await result.current.saveCustomer({
      ...toCustomerFormValues(existing),
      mcp_servers_and_groups: { servers: [added.server_id], accessGroups: [], toolsets: [toolset.toolset_id] },
    });

    expect(mocks.update).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ object_permission: expect.objectContaining({ mcp_toolsets: [toolset.toolset_id] }) }),
    );
  });
});
