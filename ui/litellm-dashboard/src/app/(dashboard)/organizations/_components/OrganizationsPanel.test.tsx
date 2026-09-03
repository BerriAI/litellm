import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type OrganizationsTableComponent from "./OrganizationsTable";
import type { Organization } from "@/components/networking";
import type OrganizationInfoViewComponent from "@/components/organization/organization_view";

vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: () => null,
}));
type AuthState = { accessToken: string | null; userId: string | null; userRole: string | null };
const SIGNED_OUT: AuthState = { accessToken: null, userId: null, userRole: null };
const SIGNED_IN: AuthState = { accessToken: "sk-test", userId: "user-1", userRole: "Admin" };
let authState: AuthState = SIGNED_OUT;
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => authState,
}));

const makeOrganization = (organization_id: string, organization_alias: string): Organization => ({
  organization_id,
  organization_alias,
  budget_id: "",
  metadata: {},
  models: [],
  spend: 0,
  model_spend: {},
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "user-1",
  litellm_budget_table: null,
  teams: null,
  users: null,
  members: null,
});
const ALPHA_ORG_ID = "ff9eb074-3f07-4e13-91fb-5337fb10d760";
const BETA_ORG_ID = "0a1b2c3d-1111-4222-8333-444455556666";
const SERVER_ORGANIZATIONS: Organization[] = [
  makeOrganization(ALPHA_ORG_ID, "Alpha Org"),
  makeOrganization(BETA_ORG_ID, "Beta Org"),
  makeOrganization("9f8e7d6c-9999-4888-8777-666655554444", "Gamma Org"),
];
const matchesServerFilters = (organization: Organization, orgId: string | null, orgAlias: string | null) => {
  const idMatches = !orgId || organization.organization_id === orgId;
  const aliasMatches = !orgAlias || organization.organization_alias.toLowerCase().includes(orgAlias.toLowerCase());
  return idMatches && aliasMatches;
};
vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    organizationListCall: (_accessToken: string, orgId: string | null = null, orgAlias: string | null = null) =>
      Promise.resolve(
        SERVER_ORGANIZATIONS.filter((organization) => matchesServerFilters(organization, orgId, orgAlias)),
      ),
    modelAvailableCall: () => Promise.resolve({ data: [] }),
  };
});
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
  authState = SIGNED_OUT;
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

describe("OrganizationsPanel - search by organization name or ID", () => {
  const renderSignedInPanel = async () => {
    authState = SIGNED_IN;
    renderPanel();
    await waitFor(() => expect(capturedTableProps?.organizations).toHaveLength(SERVER_ORGANIZATIONS.length));
  };

  const search = (value: string) =>
    fireEvent.change(screen.getByPlaceholderText("Search by organization name or ID"), { target: { value } });

  const visibleOrganizationIds = () =>
    capturedTableProps?.organizations.map((organization) => organization.organization_id);

  it.each([
    ["a full organization_id", ALPHA_ORG_ID, ALPHA_ORG_ID],
    ["a substring of an organization_id", "5337fb10", ALPHA_ORG_ID],
    ["an alias substring, ignoring case", "BETA", BETA_ORG_ID],
  ])("keeps only the organization matching %s and drops the others", async (_label, query, expectedId) => {
    await renderSignedInPanel();

    search(query);

    await waitFor(() => expect(visibleOrganizationIds()).toEqual([expectedId]));
  });

  it("shows the search empty state when neither an alias nor an id matches", async () => {
    await renderSignedInPanel();

    search("no-such-org");

    await waitFor(() => expect(capturedTableProps?.organizations).toEqual([]));
    expect(capturedTableProps?.searchActive).toBe(true);
  });
});
