import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { CustomerResourcesCell, describeMcpGrant } from "./CustomersTableColumns";

const customer: EndUser = {
  user_id: "customer-id",
  spend: 0,
  blocked: false,
  models: ["model-a", "model-b"],
  object_permission: {
    object_permission_id: "permission-id",
    agent_access_groups: [],
    agents: [],
    blocked_tools: [],
    mcp_servers: ["server-a"],
    mcp_access_groups: ["group-a", "group-b"],
    mcp_toolsets: ["toolset-a", "toolset-b", "toolset-c"],
    mcp_tool_permissions: { "server-b": ["read"] },
    models: [],
    search_tools: [],
    vector_stores: [],
  },
};

describe("CustomerResourcesCell", () => {
  it("shows separately labeled counts for models and each MCP grant type", () => {
    render(<CustomerResourcesCell customer={customer} />);
    expect(screen.getByLabelText("2 models")).toHaveTextContent("2");
    expect(screen.getByLabelText("2 MCP servers")).toHaveTextContent("2");
    expect(screen.getByLabelText("2 MCP access groups")).toHaveTextContent("2");
    expect(screen.getByLabelText("3 MCP toolsets")).toHaveTextContent("3");
  });

  it("shows All for unrestricted models and excludes unsupported MCP sentinel counts", () => {
    render(
      <CustomerResourcesCell
        customer={{
          ...customer,
          models: [],
          object_permission: {
            ...customer.object_permission!,
            mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL],
            mcp_tool_permissions: {},
          },
        }}
      />,
    );
    expect(screen.getByLabelText("All models")).toHaveTextContent("All");
    expect(screen.getByLabelText("0 MCP servers")).toHaveTextContent("0");
  });
});

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
    expect(describeMcpGrant(permission)).toBe("1 server, 1 tool limit");
  });

  it("does not describe the unsupported customer all-servers sentinel as granting access", () => {
    const permission = { ...emptyGrant, mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL] };
    expect(describeMcpGrant(permission)).toBe("No MCP access");
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
