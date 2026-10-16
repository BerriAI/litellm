import React from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { vi, test, expect, beforeEach, describe, type Mock } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import OrganizationInfoView from "./organization_view";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/organizations",
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

// Mock networking calls used by the component's mutation handlers.
vi.mock("../networking", () => {
  return {
    __esModule: true,
    organizationMemberAddCall: vi.fn(),
    organizationMemberUpdateCall: vi.fn(),
    organizationMemberDeleteCall: vi.fn(),
    serverRootPath: "",
  };
});

// Mock the React Query hook the component now reads org data from. The component
// also imports organizationKeys (used inside mutation handlers for invalidation),
// so provide a stub shape here too.
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganization: vi.fn(),
  organizationKeys: {
    all: ["organizations"],
    list: () => ["organizations", "list", { params: {} }],
    detail: (id: string) => ["organizations", "detail", id],
  },
}));

const mockUseOrganization = vi.mocked(useOrganization);

// Mock noisy/heavy child components to keep this test focused on render
vi.mock("../object_permissions_view", () => ({
  __esModule: true,
  default: () => <div data-testid="object-permissions-view" />,
}));
vi.mock("../team/edit_membership", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("../common_components/user_search_modal", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("../vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: () => null,
}));
vi.mock("../mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: () => null,
}));
const mockUseTeamsData = {
  data: [
    {
      team_id: "team_123",
      team_alias: "Engineering Team",
    },
    {
      team_id: "team_456",
      team_alias: "Marketing Team",
    },
  ],
};

const mockUseTeams = vi.fn(() => mockUseTeamsData);

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => mockUseTeams(),
  useTeam: () => ({ data: undefined }),
}));

const mockOrg = {
  organization_alias: "Acme Corp",
  organization_id: "org_123",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  created_by: "admin@example.com",
  spend: 0,
  models: ["gpt-4o-mini"],
  litellm_budget_table: {
    tpm_limit: null,
    rpm_limit: null,
    max_budget: 1000,
    budget_duration: "30d",
    max_parallel_requests: null,
  },
  object_permission: {},
  members: [],
  teams: [],
  metadata: null,
};

beforeEach(() => {
  mockUseOrganization.mockReset();
});

test("renders organization view after loading data", async () => {
  mockUseOrganization.mockReturnValue({ data: mockOrg, isLoading: false } as any);

  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  const [orgName] = await screen.findAllByText("Acme Corp");
  expect(orgName).toBeInTheDocument();
});

test("should display empty state when organization has no members", async () => {
  mockUseOrganization.mockReturnValue({ data: mockOrg, isLoading: false } as any);

  const user = userEvent.setup();
  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  await waitFor(() => {
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  await user.click(screen.getByRole("tab", { name: "Members" }));

  await waitFor(() => {
    expect(screen.getByText("No members found")).toBeInTheDocument();
  });
});

test("should display team aliases when teams are available", async () => {
  const orgWithTeams = {
    ...mockOrg,
    teams: [{ team_id: "team_123" }, { team_id: "team_456" }],
  };
  mockUseOrganization.mockReturnValue({ data: orgWithTeams, isLoading: false } as any);

  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  await waitFor(() => {
    expect(screen.getByText("Engineering Team")).toBeInTheDocument();
    expect(screen.getByText("Marketing Team")).toBeInTheDocument();
  });
});

test("should display team ID as fallback when alias is not found", async () => {
  mockUseTeams.mockReturnValueOnce({
    data: [
      {
        team_id: "team_123",
        team_alias: "Engineering Team",
      },
    ],
  });

  const orgWithUnknownTeam = {
    ...mockOrg,
    teams: [{ team_id: "team_999" }],
  };
  mockUseOrganization.mockReturnValue({ data: orgWithUnknownTeam, isLoading: false } as any);

  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  await waitFor(() => {
    expect(screen.getByText("team_999")).toBeInTheDocument();
  });
});

test("links each team badge to that team's detail page", async () => {
  const orgWithTeams = {
    ...mockOrg,
    teams: [{ team_id: "team_123" }, { team_id: "team_456" }],
  };
  mockUseOrganization.mockReturnValue({ data: orgWithTeams, isLoading: false } as any);

  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  await waitFor(() => {
    expect(screen.getByRole("link", { name: "Engineering Team" })).toHaveAttribute(
      "href",
      expect.stringContaining("/teams?team=team_123"),
    );
    expect(screen.getByRole("link", { name: "Marketing Team" })).toHaveAttribute(
      "href",
      expect.stringContaining("/teams?team=team_456"),
    );
  });
});

test("model badges stay non-clickable", async () => {
  mockUseOrganization.mockReturnValue({ data: mockOrg, isLoading: false } as any);

  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
    />,
  );

  await waitFor(() => {
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
  });
  expect(screen.queryByRole("link", { name: "gpt-4o-mini" })).not.toBeInTheDocument();
});

test("should keep unsaved settings edits when switching tabs and back", async () => {
  mockUseOrganization.mockReturnValue({ data: mockOrg, isLoading: false } as any);

  const user = userEvent.setup();
  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={true}
      userModels={[]}
    />,
  );

  await user.click(screen.getByRole("tab", { name: "Settings" }));
  await user.click(await screen.findByRole("button", { name: /Edit Settings/i }));

  const alias = await screen.findByLabelText(/Organization Name/i);
  await user.clear(alias);
  fireEvent.change(alias, { target: { value: "Renamed Org" } });
  expect(alias).toHaveValue("Renamed Org");

  await user.click(screen.getByRole("tab", { name: "Overview" }));
  await user.click(screen.getByRole("tab", { name: "Settings" }));

  expect(screen.getByLabelText(/Organization Name/i)).toHaveValue("Renamed Org");
});

test("renders a tpm/rpm limit of 0 as 0 in the overview and settings tabs, never as Unlimited", async () => {
  const zeroLimitOrg = {
    ...mockOrg,
    litellm_budget_table: { ...mockOrg.litellm_budget_table, tpm_limit: 0, rpm_limit: 0 },
  };
  mockUseOrganization.mockReturnValue({ data: zeroLimitOrg, isLoading: false } as unknown as ReturnType<
    typeof useOrganization
  >);

  const user = userEvent.setup();
  renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={true}
      userModels={[]}
    />,
  );

  const overview = await screen.findByRole("tabpanel", { name: "Overview" });
  expect(within(overview).getByText("TPM: 0")).toBeInTheDocument();
  expect(within(overview).getByText("RPM: 0")).toBeInTheDocument();

  await user.click(screen.getByRole("tab", { name: "Settings" }));
  const settings = await screen.findByRole("tabpanel", { name: "Settings" });
  expect(within(settings).getByText("TPM: 0")).toBeInTheDocument();
  expect(within(settings).getByText("RPM: 0")).toBeInTheDocument();
  expect(screen.queryByText("TPM: Unlimited")).not.toBeInTheDocument();
  expect(screen.queryByText("RPM: Unlimited")).not.toBeInTheDocument();
});

const renderOrgView = (props: { is_proxy_admin?: boolean } = {}) => (
  <OrganizationInfoView
    organizationId="org_123"
    onClose={() => {}}
    accessToken="test-token"
    is_org_admin={false}
    is_proxy_admin={props.is_proxy_admin ?? false}
    userModels={[]}
  />
);

const lastSearchParams = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

describe("organization detail tab in the URL (?org_tab=)", () => {
  beforeEach(() => {
    mockUseOrganization.mockReturnValue({ data: mockOrg, isLoading: false } as unknown as ReturnType<
      typeof useOrganization
    >);
  });

  test("opens on the tab named by ?org_tab=", () => {
    renderWithProviders(renderOrgView(), { searchParams: "?org=org_123&org_tab=members" });

    expect(screen.getByRole("tab", { name: "Members" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("No members found")).toBeInTheDocument();
  });

  test("the settings deep link used by the list's Edit action opens the Settings tab", () => {
    renderWithProviders(renderOrgView({ is_proxy_admin: true }), { searchParams: "?org=org_123&org_tab=settings" });

    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
  });

  test("opens on Overview when the URL names no tab", () => {
    renderWithProviders(renderOrgView(), { searchParams: "?org=org_123" });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  });

  test("writes the selected tab to ?org_tab= and drops it again for Overview", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(renderOrgView(), { searchParams: "?org=org_123", onUrlUpdate });

    await user.click(screen.getByRole("tab", { name: "Settings" }));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("org_tab")).toBe("settings"));
    expect(lastSearchParams(onUrlUpdate)?.get("org")).toBe("org_123");
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");

    await user.click(screen.getByRole("tab", { name: "Overview" }));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("org_tab")).toBe(false));
    expect(lastSearchParams(onUrlUpdate)?.get("org")).toBe("org_123");
  });

  test("falls back to Overview for an unknown ?org_tab= and removes it from the URL", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(renderOrgView(), {
      wrapper: ({ children }) => (
        <NuqsTestingAdapter
          searchParams="?org=org_123&org_tab=billing"
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
        </NuqsTestingAdapter>
      ),
    });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastSearchParams(onUrlUpdate)?.has("org_tab")).toBe(false);
    expect(lastSearchParams(onUrlUpdate)?.get("org")).toBe("org_123");
  });

  test("follows back and forward navigation between tabs while the detail view stays open", () => {
    const atUrl = (searchParams: string) => (
      <NuqsTestingAdapter searchParams={searchParams} hasMemory>
        <QueryClientProvider client={testQueryClient}>{renderOrgView()}</QueryClientProvider>
      </NuqsTestingAdapter>
    );
    const { rerender } = render(atUrl("?org=org_123&org_tab=members"));
    expect(screen.getByRole("tab", { name: "Members" })).toHaveAttribute("aria-selected", "true");

    rerender(atUrl("?org=org_123"));
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");

    rerender(atUrl("?org=org_123&org_tab=members"));
    expect(screen.getByRole("tab", { name: "Members" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("No members found")).toBeInTheDocument();
  });
});
