import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("./mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: null,
    userId: null,
    userRole: null,
  }),
}));

import OrganizationsTable from "./organizations";

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe("OrganizationsTable", () => {
  it("should render the OrganizationsTable component", () => {
    const { getByText } = renderWithQueryClient(
      <OrganizationsTable userRole="Admin" accessToken={null} premiumUser={true} />,
    );

    expect(getByText("+ Create New Organization")).toBeInTheDocument();
  });
});
