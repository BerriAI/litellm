import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { toCustomerFormValues } from "./customerPayload";
import { useSaveCustomer } from "./useSaveCustomer";

const mocks = vi.hoisted(() => ({ servers: vi.fn(), toolsets: vi.fn(), create: vi.fn(), update: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({ useMCPServers: mocks.servers }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({ useMCPToolsets: mocks.toolsets }));
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
});
