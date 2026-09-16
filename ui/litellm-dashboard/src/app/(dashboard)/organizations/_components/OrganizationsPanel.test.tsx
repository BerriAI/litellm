import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type OrganizationsTableComponent from "./OrganizationsTable";
import type OrganizationInfoViewComponent from "@/components/organization/organization_view";
import type { OrganizationListFilters } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

const useOrganizationsSpy = vi.hoisted(() => vi.fn<(filters?: OrganizationListFilters) => void>());
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/app/(dashboard)/hooks/organizations/useOrganizations")>();
  return {
    ...actual,
    useOrganizations: (filters?: OrganizationListFilters) => {
      useOrganizationsSpy(filters);
      return actual.useOrganizations(filters);
    },
  };
});

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

const lastSearchParams = () => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

beforeEach(() => {
  capturedTableProps = null;
  mockOrgInfoView.mockClear();
  onUrlUpdate.mockClear();
  useOrganizationsSpy.mockClear();
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

    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-from-url" }));
    expect(screen.queryByTestId("organizations-table")).not.toBeInTheDocument();
  });

  it("closing the org detail removes ?org= and returns to the list", async () => {
    renderPanel({ searchParams: "?org=org-from-url" });

    act(() => mockOrgInfoView.mock.calls.at(-1)?.[0].onClose());

    await expectQueryString("");
    expect(screen.queryByTestId("organization-info-view")).not.toBeInTheDocument();
    expect(screen.getByTestId("organizations-table")).toBeInTheDocument();
  });

  it("the edit action pushes ?org= with ?org_tab=settings in one history entry", async () => {
    renderPanel();

    act(() => capturedTableProps?.onEditClick("org-edit"));

    await expectQueryString("?org=org-edit&org_tab=settings");
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ options: expect.objectContaining({ history: "push" }) }),
    );
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-edit" }));
  });

  it("a plain row click after leaving an edit view via browser history opens the detail without the settings tab", async () => {
    const { navigate } = renderPanel();

    act(() => capturedTableProps?.onEditClick("org-edit"));
    await expectQueryString("?org=org-edit&org_tab=settings");

    navigate("");
    expect(screen.getByTestId("organizations-table")).toBeInTheDocument();

    act(() => capturedTableProps?.onOrganizationClick("org-plain"));

    await expectQueryString("?org=org-plain");
    expect(mockOrgInfoView).toHaveBeenLastCalledWith(expect.objectContaining({ organizationId: "org-plain" }));
  });

  it("a row click drops a leftover ?org_tab= so the detail opens on its default tab", async () => {
    renderPanel({ searchParams: "?org_tab=settings" });

    act(() => capturedTableProps?.onOrganizationClick("org-plain"));

    await expectQueryString("?org=org-plain");
  });

  it("closing the org detail drops ?org_tab= together with ?org=", async () => {
    renderPanel({ searchParams: "?org=org-from-url&org_tab=members" });

    act(() => mockOrgInfoView.mock.calls.at(-1)?.[0].onClose());

    await expectQueryString("");
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ options: expect.objectContaining({ history: "push" }) }),
    );
  });
});

describe("OrganizationsPanel - list filters in the URL", () => {
  it("restores the name search and org ID filter from the URL and fetches with both", () => {
    renderPanel({ searchParams: "?org_search=Acme&filter_org_id=org-7" });

    expect(screen.getByPlaceholderText("Search by Organization Name")).toHaveValue("Acme");
    expect(screen.getByPlaceholderText("Search by Organization ID")).toHaveValue("org-7");
    expect(useOrganizationsSpy).toHaveBeenLastCalledWith({ org_id: "org-7", org_alias: "Acme" });
    expect(capturedTableProps?.searchActive).toBe(true);
  });

  it("keeps the org ID filter panel collapsed when the URL has no org ID filter", () => {
    renderPanel({ searchParams: "?org_search=Acme" });

    expect(screen.queryByPlaceholderText("Search by Organization ID")).not.toBeInTheDocument();
    expect(useOrganizationsSpy).toHaveBeenLastCalledWith({ org_id: "", org_alias: "Acme" });
  });

  it("writes the name search to ?org_search= and returns the list to the first page", async () => {
    renderPanel({ searchParams: "?page=3" });

    fireEvent.change(screen.getByPlaceholderText("Search by Organization Name"), { target: { value: "Acme" } });

    await waitFor(() => expect(lastSearchParams()?.get("org_search")).toBe("Acme"));
    expect(lastSearchParams()?.has("page")).toBe(false);
    expect(useOrganizationsSpy).toHaveBeenLastCalledWith({ org_id: "", org_alias: "Acme" });
  });

  it("writes the org ID filter to ?filter_org_id= and returns the list to the first page", async () => {
    renderPanel({ searchParams: "?page=3" });

    fireEvent.click(screen.getByRole("button", { name: "Filters" }));
    fireEvent.change(screen.getByPlaceholderText("Search by Organization ID"), { target: { value: "org-9" } });

    await waitFor(() => expect(lastSearchParams()?.get("filter_org_id")).toBe("org-9"));
    expect(lastSearchParams()?.has("page")).toBe(false);
    expect(useOrganizationsSpy).toHaveBeenLastCalledWith({ org_id: "org-9", org_alias: "" });
  });

  it("clears the search, the org ID filter and the page in one update on reset", async () => {
    renderPanel({ searchParams: "?org_search=Acme&filter_org_id=org-7&page=2" });

    fireEvent.click(screen.getByRole("button", { name: "Reset Filters" }));

    await expectQueryString("");
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(useOrganizationsSpy).toHaveBeenLastCalledWith({ org_id: "", org_alias: "" });
    expect(capturedTableProps?.searchActive).toBe(false);
  });
});
