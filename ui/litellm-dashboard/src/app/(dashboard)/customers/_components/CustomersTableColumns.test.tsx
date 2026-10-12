import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { CustomerResourcesCell } from "./CustomersTableColumns";

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
