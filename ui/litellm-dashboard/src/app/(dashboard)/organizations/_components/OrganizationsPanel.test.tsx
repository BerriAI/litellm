import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
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

import OrganizationsPanel from "./OrganizationsPanel";

const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();

interface RenderPanelOptions {
  premiumUser?: boolean;
  searchParams?: string;
}

const renderPanel = ({ premiumUser = true, searchParams = "" }: RenderPanelOptions = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const url = { current: searchParams };
  const handleUrlUpdate = (event: UrlUpdateEvent) => {
    onUrlUpdate(event);
    url.current = event.queryString;
  };
  const tree = (currentSearchParams: string) => (
    <NuqsTestingAdapter searchParams={currentSearchParams} onUrlUpdate={handleUrlUpdate} hasMemory>
      <QueryClientProvider client={queryClient}>
        <OrganizationsPanel userRole="Admin" accessToken={null} premiumUser={premiumUser} />
      </QueryClientProvider>
    </NuqsTestingAdapter>
  );
  const { rerender } = render(tree(searchParams));
  return {
    navigate: (nextSearchParams: string) => {
      rerender(tree(url.current));
      rerender(tree(nextSearchParams));
      url.current = nextSearchParams;
    },
  };
};

const expectQueryString = (queryString: string) =>
  waitFor(() => expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString })));

beforeEach(() => {
  capturedTableProps = null;
  mockOrgInfoView.mockClear();
  onUrlUpdate.mockClear();
});

describe("OrganizationsPanel", () => {
  it("gates non-premium users behind the enterprise notice", () => {
    renderPanel({ premiumUser: false });

    expect(screen.getByText(/LiteLLM Enterprise feature/i)).toBeInTheDocument();
    expect(screen.queryByText("+ Create New Organization")).not.toBeInTheDocument();
  });

  it("shows the create button for a premium admin", () => {
    renderPanel();

    expect(screen.getByText("+ Create New Organization")).toBeInTheDocument();
  });

  it("resolves the loading skeleton to false when the query is disabled (no token)", () => {
    renderPanel();

    // A disabled React Query keeps isPending true forever; feeding isLoading avoids a stuck skeleton.
    expect(screen.getByTestId("organizations-table")).toHaveTextContent("isLoading:false");
  });
});

describe("OrganizationsPanel - org detail deep link (?org=)", () => {
  it("clicking an organization pushes ?org= and opens the detail view", async () => {
    renderPanel();

    act(() => capturedTableProps?.onOrganizationClick("org-deep-link"));

    await expectQueryString("?org=org-deep-link");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ options: expect.objectContaining({ history: "push" }) }),
    );
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-deep-link" }));
  });

  it("opens the org detail directly from a ?org= deep link", () => {
    renderPanel({ searchParams: "?org=org-from-url" });

    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-from-url", editOrg: false }),
    );
    expect(screen.queryByTestId("organizations-table")).not.toBeInTheDocument();
  });

  it("closing the org detail removes ?org= and returns to the list", async () => {
    renderPanel({ searchParams: "?org=org-from-url" });

    act(() => mockOrgInfoView.mock.calls.at(-1)?.[0].onClose());

    await expectQueryString("");
    expect(screen.queryByTestId("organization-info-view")).not.toBeInTheDocument();
    expect(screen.getByTestId("organizations-table")).toBeInTheDocument();
  });

  it("the edit action opens the detail in edit mode with ?org= set", async () => {
    renderPanel();

    act(() => capturedTableProps?.onEditClick("org-edit"));

    await expectQueryString("?org=org-edit");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-edit", editOrg: true }),
    );
  });

  it("a plain row click after leaving an edit view via browser history does not reopen in edit mode", async () => {
    const { navigate } = renderPanel();

    act(() => capturedTableProps?.onEditClick("org-edit"));
    await expectQueryString("?org=org-edit");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ editOrg: true }));

    navigate("");
    expect(screen.getByTestId("organizations-table")).toBeInTheDocument();

    act(() => capturedTableProps?.onOrganizationClick("org-plain"));

    await expectQueryString("?org=org-plain");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(
      expect.objectContaining({ organizationId: "org-plain", editOrg: false }),
    );
  });
});
