import { describe, expect, it } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { describeMcpGrant } from "./CustomersTableColumns";

const emptyGrant: NonNullable<EndUser["object_permission"]> = {
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

describe("describeMcpGrant", () => {
  it.each([null, undefined, emptyGrant])("describes an empty grant as unrestricted", (permission) => {
    expect(describeMcpGrant(permission)).toBe("No restriction");
  });

  it("summarizes servers, groups, toolsets and tool limits", () => {
    const permission = {
      ...emptyGrant,
      mcp_servers: ["server-a", "server-b"],
      mcp_access_groups: ["group-id"],
      mcp_toolsets: ["toolset-id"],
      mcp_tool_permissions: { "server-a": ["read"] },
    };
    expect(describeMcpGrant(permission)).toBe("2 servers, 1 group, 1 toolset, 1 tool limit");
  });

  it("does not describe a tool-only grant as unrestricted", () => {
    const permission = { ...emptyGrant, mcp_tool_permissions: { "server-id": ["read"] } };
    expect(describeMcpGrant(permission)).toBe("1 tool limit");
  });

  it("labels the all-servers sentinel instead of counting it as a server", () => {
    const permission = { ...emptyGrant, mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL] };
    expect(describeMcpGrant(permission)).toBe("All MCP servers");
  });

  it("shows the no-servers sentinel as blocked even if other grants exist", () => {
    const permission = {
      ...emptyGrant,
      mcp_servers: [NO_MCP_SERVERS_SENTINEL],
      mcp_access_groups: ["group-id"],
    };
    expect(describeMcpGrant(permission)).toBe("No MCP access");
  });
});
