import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type OrganizationsTableComponent from "./OrganizationsTable";
import type OrganizationInfoViewComponent from "@/components/organization/organization_view";

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
type OrganizationsTableProps = React.ComponentProps<typeof OrganizationsTableComponent>;
type OrganizationInfoViewProps = React.ComponentProps<typeof OrganizationInfoViewComponent>;

let capturedTableProps: OrganizationsTableProps | null = null;
vi.mock("./OrganizationsTable", () => ({
  __esModule: true,
  default: (props: OrganizationsTableProps) => {
    capturedTableProps = props;
    return <div data-testid="organizations-table">isLoading:{String(props.isLoading)}</div>;
  },
}));
const mockOrgInfoView = vi.fn<(props: OrganizationInfoViewProps) => void>();
vi.mock("@/components/organization/organization_view", () => ({
  __esModule: true,
  default: (props: OrganizationInfoViewProps) => {
    mockOrgInfoView(props);
    return <div data-testid="organization-info-view" />;
  },
}));

// The selected org is URL-derived (?org=) via useOrgDetailRouting. Next's real useSearchParams
// re-renders subscribers on history.pushState/replaceState; mirror that so URL changes propagate.
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  const LOCATION_CHANGE_EVENT = "test-locationchange";
  for (const method of ["pushState", "replaceState"] as const) {
    const original = window.history[method].bind(window.history);
    window.history[method] = (...args: Parameters<History["pushState"]>) => {
      original(...args);
      window.dispatchEvent(new Event(LOCATION_CHANGE_EVENT));
    };
  }
  const subscribe = (onChange: () => void) => {
    window.addEventListener(LOCATION_CHANGE_EVENT, onChange);
    window.addEventListener("popstate", onChange);
    return () => {
      window.removeEventListener(LOCATION_CHANGE_EVENT, onChange);
      window.removeEventListener("popstate", onChange);
    };
  };
  return {
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(subscribe, () => window.location.search)),
  };
});

import OrganizationsPanel from "./OrganizationsPanel";

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

beforeEach(() => {
  capturedTableProps = null;
  mockOrgInfoView.mockClear();
  window.history.replaceState(null, "", "/organizations/");
});

describe("OrganizationsPanel", () => {
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
});

describe("OrganizationsPanel - org detail deep link (?org=)", () => {
  it("clicking an organization pushes ?org= and opens the detail view", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => capturedTableProps?.onOrganizationClick("org-deep-link"));

    expect(window.location.search).toContain("org=org-deep-link");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-deep-link" }));
  });

  it("opens the org detail directly from a ?org= deep link", () => {
    window.history.replaceState(null, "", "/organizations/?org=org-from-url");
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-from-url", editOrg: false }),
    );
    expect(screen.queryByTestId("organizations-table")).not.toBeInTheDocument();
  });

  it("closing the org detail removes ?org= and returns to the list", () => {
    window.history.replaceState(null, "", "/organizations/?org=org-from-url");
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => mockOrgInfoView.mock.calls.at(-1)?.[0].onClose());

    expect(window.location.search).not.toContain("org=");
    expect(screen.queryByTestId("organization-info-view")).not.toBeInTheDocument();
    expect(screen.getByTestId("organizations-table")).toBeInTheDocument();
  });

  it("the edit action opens the detail in edit mode with ?org= set", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => capturedTableProps?.onEditClick("org-edit"));

    expect(window.location.search).toContain("org=org-edit");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-edit", editOrg: true }),
    );
  });

  it("a plain row click after leaving an edit view via browser history does not reopen in edit mode", () => {
    renderWithQueryClient(<OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={true} />);

    act(() => capturedTableProps?.onEditClick("org-edit"));
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ editOrg: true }));

    act(() => window.history.pushState(null, "", "/organizations/"));
    act(() => capturedTableProps?.onOrganizationClick("org-plain"));

    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-plain", editOrg: false }),
    );
  });
});
