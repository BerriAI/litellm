import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockOrganizationInfoView = vi.fn();
let mockOrganizationsTableProps: any = null;

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
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
vi.mock("@/components/organization/organization_view", () => ({
  __esModule: true,
  default: (props: any) => {
    mockOrganizationInfoView(props);
    return <div data-testid="organization-info-view" />;
  },
}));
vi.mock("./OrganizationsTable", () => ({
  __esModule: true,
  default: (props: { isLoading: boolean }) => {
    mockOrganizationsTableProps = props;
    return <div data-testid="organizations-table">isLoading:{String(props.isLoading)}</div>;
  },
}));

import OrganizationsPanel from "./OrganizationsPanel";

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe("OrganizationsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockOrganizationsTableProps = null;
    window.history.pushState(null, "", "/");
  });

  it("gates non-premium users behind the enterprise notice", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={false} />);

    expect(screen.getByText(/LiteLLM Enterprise feature/i)).toBeInTheDocument();
    expect(screen.queryByText("+ Create New Organization")).not.toBeInTheDocument();
  });

  it("shows the create button for a premium admin", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    expect(screen.getByText("+ Create New Organization")).toBeInTheDocument();
  });

  it("resolves the loading skeleton to false when the query is disabled (no token)", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    // A disabled React Query keeps isPending true forever; feeding isLoading avoids a stuck skeleton.
    expect(screen.getByTestId("organizations-table")).toHaveTextContent("isLoading:false");
  });

  it("clicking an organization deep-links via ?org=", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => mockOrganizationsTableProps.onOrganizationClick("org-42"));

    expect(window.location.search).toContain("org=org-42");
  });

  it("renders OrganizationInfoView from a ?org= URL on load", () => {
    window.history.pushState(null, "", "/?org=org-42");

    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    expect(screen.getByTestId("organization-info-view")).toBeInTheDocument();
    expect(mockOrganizationInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-42" }));
    expect(screen.queryByTestId("organizations-table")).not.toBeInTheDocument();
  });

  it("closing the detail view clears ?org=", () => {
    window.history.pushState(null, "", "/?org=org-42");

    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => mockOrganizationInfoView.mock.calls.at(-1)?.[0].onClose());

    expect(window.location.search).not.toContain("org=");
  });
});
