import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, test, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import OrganizationInfoView from "./organization_view";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

// Mock networking calls used by the component's mutation handlers
vi.mock("../networking", () => {
  return {
    __esModule: true,
    organizationMemberAddCall: vi.fn(),
    organizationMemberUpdateCall: vi.fn(),
    organizationMemberDeleteCall: vi.fn(),
    organizationUpdateCall: vi.fn(),
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
vi.mock("../molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
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

  const { findAllByText } = renderWithProviders(
    <OrganizationInfoView
      organizationId="org_123"
      onClose={() => {}}
      accessToken="test-token"
      is_org_admin={false}
      is_proxy_admin={false}
      userModels={[]}
      editOrg={false}
    />,
  );

  await waitFor(() => {
    expect(findAllByText("Acme Corp")).toBeTruthy();
  });
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
      editOrg={false}
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
      editOrg={false}
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
      editOrg={false}
    />,
  );

  await waitFor(() => {
    expect(screen.getByText("team_999")).toBeInTheDocument();
  });
});
