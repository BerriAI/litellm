import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import React from "react";

vi.mock("./vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("./mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: () => null,
}));

import OrganizationsTable from "./organizations";

describe("OrganizationsTable", () => {
  it("should render the OrganizationsTable component", () => {
    const setOrganizations = vi.fn();

    const { getByText } = render(
      <OrganizationsTable
        organizations={[]}
        userRole="Admin"
        userModels={[]}
        accessToken={null}
        setOrganizations={setOrganizations}
        premiumUser={true}
      />,
    );

    expect(getByText("+ Create New Organization")).toBeInTheDocument();
  });
});
