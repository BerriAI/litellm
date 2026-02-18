import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamData } from "./TeamInfo";
import TeamMembersComponent from "./TeamMemberTab";

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

const mockHandleMemberDelete = vi.fn();
const mockSetSelectedEditMember = vi.fn();
const mockSetIsEditMemberModalVisible = vi.fn();
const mockSetIsAddMemberModalVisible = vi.fn();

const createMockTeamData = (overrides: Partial<TeamData> = {}): TeamData => ({
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
  ...overrides,
});

describe("TeamMembersComponent", () => {
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
      isLoading: false,
      isAuthorized: true,
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

  it("should render", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should render team members table with headers", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByRole("columnheader", { name: /user email/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /user id/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /team role/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /actions/i })).toBeInTheDocument();
  });

  it("should render team members data", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    // user1@test.com appears twice (User ID and User Email columns)
    expect(screen.getAllByText("user1@test.com").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("user2@test.com").length).toBeGreaterThanOrEqual(1);
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("member");
    expect(table).toHaveTextContent("admin");
  });

  it("should render Add Member button", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("Add Member")).toBeInTheDocument();
  });

  it("should display dash when user email is null", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData({
          team_info: {
            ...createMockTeamData().team_info,
            members_with_roles: [
              {
                user_id: "user-without-email",
                user_email: null,
                role: "user",
              },
            ],
          },
        })}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should display Default Proxy Admin tag for default_user_id", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData({
          team_info: {
            ...createMockTeamData().team_info,
            members_with_roles: [
              {
                user_id: "default_user_id",
                user_email: "admin@proxy.com",
                role: "admin",
              },
            ],
          },
        })}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });

  it("should display spend and rate limits for member with membership", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText(/\$100\.5/)).toBeInTheDocument();
    expect(screen.getByText(/100 RPM/)).toBeInTheDocument();
    expect(screen.getByText(/10000 TPM/)).toBeInTheDocument();
  });

  it("should display No Limit for budget when member has no budget", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("No Limit")).toBeInTheDocument();
  });

  it("should display No Limits for rate limits when member has no limits", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("No Limits")).toBeInTheDocument();
  });

  it("should call setIsEditMemberModalVisible and setSelectedEditMember when edit button is clicked", async () => {
    const user = userEvent.setup();
    vi.mocked(isProxyAdminRole).mockReturnValue(true);
    vi.mocked(isUserTeamAdminForSingleTeam).mockReturnValue(false);

    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    const editButtons = screen.getAllByTestId("edit-member");
    await user.click(editButtons[0]);

    expect(mockSetIsEditMemberModalVisible).toHaveBeenCalledWith(true);
    expect(mockSetSelectedEditMember).toHaveBeenCalled();
  });

  it("should call setIsAddMemberModalVisible when Add Member button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    const addButton = screen.getByRole("button", { name: /add member/i });
    await user.click(addButton);

    expect(mockSetIsAddMemberModalVisible).toHaveBeenCalledWith(true);
  });

  it("should hide delete button when disable_team_admin_delete_team_user is true and user is team admin", () => {
    vi.mocked(isProxyAdminRole).mockReturnValue(false);
    vi.mocked(isUserTeamAdminForSingleTeam).mockReturnValue(true);
    vi.mocked(useUISettings).mockReturnValue({
      data: { values: { disable_team_admin_delete_team_user: true } },
      isLoading: false,
      isError: false,
      error: null,
      isSuccess: true,
      isFetching: false,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.queryByTestId("delete-member")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("edit-member")).toHaveLength(2);
  });

  it("should show delete button for proxy admin when canEditTeam is true", () => {
    vi.mocked(isProxyAdminRole).mockReturnValue(true);
    vi.mocked(isUserTeamAdminForSingleTeam).mockReturnValue(false);

    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getAllByTestId("delete-member")).toHaveLength(2);
    expect(screen.getAllByTestId("edit-member")).toHaveLength(2);
  });

  it("should hide action buttons when canEditTeam is false", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.queryByTestId("edit-member")).not.toBeInTheDocument();
    expect(screen.queryByTestId("delete-member")).not.toBeInTheDocument();
  });
});
