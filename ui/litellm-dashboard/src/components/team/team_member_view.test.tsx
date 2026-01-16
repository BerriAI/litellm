import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamData } from "./team_info";
import TeamMembersComponent from "./team_member_view";

// Mock the hooks
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("@/utils/roles", () => ({
  isUserTeamAdminForSingleTeam: vi.fn(() => false),
  isProxyAdminRole: vi.fn(() => false),
}));

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";

describe("TeamMembersComponent", () => {
  const mockHandleMemberDelete = vi.fn();
  const mockSetSelectedEditMember = vi.fn();
  const mockSetIsEditMemberModalVisible = vi.fn();
  const mockSetIsAddMemberModalVisible = vi.fn();

  const mockTeamData: TeamData = {
    team_id: "team-123",
    team_info: {
      team_alias: "Test Team",
      team_id: "team-123",
      organization_id: null,
      admins: ["admin@test.com"],
      members: ["user1@test.com"],
      members_with_roles: [
        {
          user_id: "user1@test.com",
          user_email: "user1@test.com",
          role: "member",
        },
        {
          user_id: "user2@test.com",
          user_email: "user2@test.com",
          role: "admin",
        },
      ],
      metadata: {},
      tpm_limit: null,
      rpm_limit: null,
      max_budget: null,
      budget_duration: null,
      models: [],
      blocked: false,
      spend: 0,
      max_parallel_requests: null,
      budget_reset_at: null,
      model_id: null,
      litellm_model_table: null,
      created_at: "2024-01-01T00:00:00Z",
      team_member_budget_table: null,
    },
    keys: [],
    team_memberships: [
      {
        user_id: "user1@test.com",
        team_id: "team-123",
        budget_id: "budget1",
        spend: 100.5,
        litellm_budget_table: {
          budget_id: "budget1",
          soft_budget: null,
          max_budget: 1000,
          max_parallel_requests: null,
          tpm_limit: 10000,
          rpm_limit: 100,
          model_max_budget: null,
          budget_duration: null,
        },
      },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useUISettings).mockReturnValue({
      data: { values: { disable_team_admin_delete_team_user: false } },
      isLoading: false,
      isError: false,
      error: null,
      isSuccess: true,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    vi.mocked(useAuthorized).mockReturnValue({
      userId: "test-user-id",
      userRole: "Admin",
      accessToken: "test-token",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });
  });

  it("should render team members table with headers", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={mockTeamData}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("User ID")).toBeInTheDocument();
    expect(screen.getByText("User Email")).toBeInTheDocument();
    expect(screen.getByText("Role")).toBeInTheDocument();
    expect(screen.getByText("Team Member Spend (USD)")).toBeInTheDocument();
    expect(screen.getByText("Team Member Budget (USD)")).toBeInTheDocument();
    expect(screen.getByText("Team Member Rate Limits")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should render team members data", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={mockTeamData}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    // user1@test.com appears twice (User ID and User Email columns)
    expect(screen.getAllByText("user1@test.com")).toHaveLength(2);
    // user2@test.com appears twice (User ID and User Email columns)
    expect(screen.getAllByText("user2@test.com")).toHaveLength(2);
    expect(screen.getByText("member")).toBeInTheDocument();
    expect(screen.getByText("admin")).toBeInTheDocument();
  });

  it("should render Add Member button", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={mockTeamData}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("Add Member")).toBeInTheDocument();
  });

  it("should show delete button for proxy admin when canEditTeam is true", () => {
    vi.mocked(isProxyAdminRole).mockReturnValue(true);
    vi.mocked(isUserTeamAdminForSingleTeam).mockReturnValue(false);

    const { container } = renderWithProviders(
      <TeamMembersComponent
        teamData={mockTeamData}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    // Verify that action buttons are rendered when canEditTeam is true
    // For proxy admin, both edit and delete buttons should be visible
    // Check for clickable icon elements (Tremor Icon components with cursor-pointer class)
    const clickableIcons = container.querySelectorAll('[class*="cursor-pointer"]');
    // Should have at least 4 icons: 2 edit buttons + 2 delete buttons for 2 members
    expect(clickableIcons.length).toBeGreaterThanOrEqual(4);

    // Verify members are rendered
    expect(screen.getAllByText("user1@test.com").length).toBeGreaterThan(0);
    expect(screen.getAllByText("user2@test.com").length).toBeGreaterThan(0);
  });
});
