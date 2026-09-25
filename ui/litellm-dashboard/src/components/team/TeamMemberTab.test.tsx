import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamData } from "./TeamInfo";
import TeamMembersComponent, { seedMemberBudgetFields } from "./TeamMemberTab";

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

const { POST } = vi.hoisted(() => ({ POST: vi.fn() }));
vi.mock("@/lib/http/api", () => ({ fetchClient: { POST } }));

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
const mockOnMemberSpendReset = vi.fn();
const mockOnMemberBudgetReset = vi.fn();

const budgetResetIso = new Date(2026, 6, 15, 12, 0, 0).toISOString();

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
      budget_source: "custom",
      spend: 100.5,
      total_spend: 1538.2608,
      litellm_budget_table: {
        budget_id: "budget1",
        soft_budget: null,
        max_budget: 1000,
        max_parallel_requests: null,
        tpm_limit: 10000,
        rpm_limit: 100,
        model_max_budget: null,
        budget_duration: null,
        budget_reset_at: budgetResetIso,
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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

  it("clears the member search when a different team is shown", () => {
    const props = {
      canEditTeam: false,
      handleMemberDelete: mockHandleMemberDelete,
      onMemberSpendReset: mockOnMemberSpendReset,
      onMemberBudgetReset: mockOnMemberBudgetReset,
      setSelectedEditMember: mockSetSelectedEditMember,
      setIsEditMemberModalVisible: mockSetIsEditMemberModalVisible,
      setIsAddMemberModalVisible: mockSetIsAddMemberModalVisible,
    };
    const { rerender } = renderWithProviders(<TeamMembersComponent teamData={createMockTeamData()} {...props} />);

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "user2" } });
    expect(screen.queryByText("user1@test.com")).not.toBeInTheDocument();

    const otherTeam = createMockTeamData({ team_id: "team-456" });
    rerender(<TeamMembersComponent teamData={otherTeam} {...props} />);

    expect(screen.getByTestId("datatable-search")).toHaveValue("");
    expect(screen.getAllByText("user1@test.com").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("user2@test.com").length).toBeGreaterThanOrEqual(1);
  });

  it("should render Add Member button", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getAllByText("-").length).toBeGreaterThanOrEqual(1);
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("$100.50")).toBeInTheDocument();
    expect(screen.getByText("$1,538.26")).toBeInTheDocument();
    expect(screen.getByText(/100 RPM/)).toBeInTheDocument();
    expect(screen.getByText(/10000 TPM/)).toBeInTheDocument();
  });

  it("should display the budget reset date for member with a budget reset", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("Jul 15, 2026")).toBeInTheDocument();
  });

  it("should display formatted budget and Unlimited for member with no budget", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.getByText("$1,000.00")).toBeInTheDocument();
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
  });

  it("should display No Limits for rate limits when member has no limits", () => {
    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={false}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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

  it("keeps a member's stored 0 limits as 0 in the table and in the edit payload, never unlimited", async () => {
    const user = userEvent.setup();
    vi.mocked(isProxyAdminRole).mockReturnValue(true);
    const baseTeamData = createMockTeamData();
    const teamData = {
      ...baseTeamData,
      team_memberships: baseTeamData.team_memberships.map((membership, index) =>
        index === 0
          ? {
              ...membership,
              litellm_budget_table: { ...membership.litellm_budget_table, max_budget: 0, tpm_limit: 0, rpm_limit: 0 },
            }
          : membership,
      ),
    };

    renderWithProviders(
      <TeamMembersComponent
        teamData={teamData}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    const memberRow = screen.getByRole("row", { name: /user1@test\.com/ });
    expect(within(memberRow).getByText("0 RPM / 0 TPM")).toBeInTheDocument();
    expect(within(memberRow).queryByText("No Limits")).not.toBeInTheDocument();

    await user.click(within(memberRow).getByTestId("edit-member"));

    const zeroLimitsMember = { user_id: "user1@test.com", max_budget_in_team: 0, tpm_limit: 0, rpm_limit: 0 };
    expect(mockSetSelectedEditMember).toHaveBeenCalledWith(expect.objectContaining(zeroLimitsMember));
  });

  it("seeds the edit payload with the stored temporary budget increase and expiry, keeping a 0 increase as 0", () => {
    const budget = {
      ...createMockTeamData().team_memberships[0].litellm_budget_table,
      temp_budget_increase: 0,
      temp_budget_expiry: "2030-01-02T03:04:00Z",
    };

    const seeded = {
      user_id: "user1@test.com",
      role: "member",
      max_budget_in_team: 1000,
      tpm_limit: 10000,
      rpm_limit: 100,
      budget_duration: null,
      allowed_models: [],
      temp_budget_increase: 0,
      temp_budget_expiry: "2030-01-02T03:04:00Z",
    };
    expect(seedMemberBudgetFields({ user_id: "user1@test.com", role: "member" }, budget)).toStrictEqual(seeded);
  });

  it("seeds null temporary budget fields for a member without a budget row", () => {
    expect(seedMemberBudgetFields({ user_id: "user2@test.com", role: "admin" }, undefined)).toMatchObject({
      max_budget_in_team: null,
      temp_budget_increase: null,
      temp_budget_expiry: null,
    });
  });

  it("should call setIsAddMemberModalVisible when Add Member button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <TeamMembersComponent
        teamData={createMockTeamData()}
        canEditTeam={true}
        handleMemberDelete={mockHandleMemberDelete}
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
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
        onMemberSpendReset={mockOnMemberSpendReset}
        onMemberBudgetReset={mockOnMemberBudgetReset}
        setSelectedEditMember={mockSetSelectedEditMember}
        setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
        setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
      />,
    );

    expect(screen.queryByTestId("edit-member")).not.toBeInTheDocument();
    expect(screen.queryByTestId("delete-member")).not.toBeInTheDocument();
  });

  describe("reset spend", () => {
    const renderEditableTab = () =>
      renderWithProviders(
        <TeamMembersComponent
          teamData={createMockTeamData()}
          canEditTeam={true}
          handleMemberDelete={mockHandleMemberDelete}
          onMemberSpendReset={mockOnMemberSpendReset}
          onMemberBudgetReset={mockOnMemberBudgetReset}
          setSelectedEditMember={mockSetSelectedEditMember}
          setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
          setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
        />,
      );

    it("resets the member's current cycle spend to $0 after confirming, then refreshes the team", async () => {
      const user = userEvent.setup();
      POST.mockResolvedValue({ data: {} });
      renderEditableTab();

      const memberRow = screen.getByRole("row", { name: /user1@test\.com/ });
      await user.click(within(memberRow).getByTestId("reset-member-spend"));

      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Spend" });
      expect(dialog).toHaveTextContent("user1@test.com");
      expect(dialog).toHaveTextContent("$100.5000");
      expect(POST).not.toHaveBeenCalled();

      await user.click(within(dialog).getByRole("button", { name: "Reset" }));

      await waitFor(() => expect(mockOnMemberSpendReset).toHaveBeenCalledTimes(1));
      expect(POST).toHaveBeenCalledExactlyOnceWith("/team/{team_id}/member/{user_id}/reset_spend", {
        params: { path: { team_id: "team-123", user_id: "user1@test.com" } },
        body: { reset_to: 0 },
      });
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("keeps the dialog open and does not refresh the team when the reset fails", async () => {
      const user = userEvent.setup();
      POST.mockRejectedValue(new Error("Cannot reset your own spend. Ask a proxy admin."));
      renderEditableTab();

      await user.click(screen.getByTestId("reset-member-spend"));
      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Spend" });
      await user.click(within(dialog).getByRole("button", { name: "Reset" }));

      await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
      expect(mockOnMemberSpendReset).not.toHaveBeenCalled();
      expect(screen.getByRole("dialog", { name: "Reset Team Member Spend" })).toBeInTheDocument();
    });

    it("does not call the API when the dialog is cancelled", async () => {
      const user = userEvent.setup();
      renderEditableTab();

      await user.click(screen.getByTestId("reset-member-spend"));
      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Spend" });
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(POST).not.toHaveBeenCalled();
    });

    it("only offers the reset on members that have current cycle spend", () => {
      renderEditableTab();

      expect(
        within(screen.getByRole("row", { name: /user1@test\.com/ })).getByTestId("reset-member-spend"),
      ).toBeVisible();
      expect(
        within(screen.getByRole("row", { name: /user2@test\.com/ })).queryByTestId("reset-member-spend"),
      ).not.toBeInTheDocument();
    });

    it("hides the reset on the caller's own row for a team admin, since the backend rejects it", () => {
      vi.mocked(useAuthorized).mockReturnValue({ userId: "user1@test.com", userRole: "Internal User" } as never);
      vi.mocked(isProxyAdminRole).mockReturnValue(false);
      renderEditableTab();

      expect(screen.queryByTestId("reset-member-spend")).not.toBeInTheDocument();
    });

    it("shows the reset on the caller's own row for a proxy admin", () => {
      vi.mocked(useAuthorized).mockReturnValue({ userId: "user1@test.com", userRole: "Admin" } as never);
      vi.mocked(isProxyAdminRole).mockReturnValue(true);
      renderEditableTab();

      expect(screen.getByTestId("reset-member-spend")).toBeVisible();
    });
  });

  describe("budget source", () => {
    const teamDataWithDefault = () => {
      const base = createMockTeamData();
      return createMockTeamData({
        team_info: {
          ...base.team_info,
          team_member_budget_table: { max_budget: 25, budget_duration: null, tpm_limit: null, rpm_limit: null },
        },
        team_memberships: [
          base.team_memberships[0],
          {
            user_id: "user2@test.com",
            team_id: "team-123",
            budget_id: "team-default-budget",
            budget_source: "team_default",
            spend: 0,
            total_spend: null,
            litellm_budget_table: {
              budget_id: "team-default-budget",
              soft_budget: null,
              max_budget: 25,
              max_parallel_requests: null,
              tpm_limit: null,
              rpm_limit: null,
              model_max_budget: null,
              budget_duration: null,
              budget_reset_at: null,
            },
          },
        ],
      });
    };

    const renderTab = (teamData: TeamData, canEditTeam = true) =>
      renderWithProviders(
        <TeamMembersComponent
          teamData={teamData}
          canEditTeam={canEditTeam}
          handleMemberDelete={mockHandleMemberDelete}
          onMemberSpendReset={mockOnMemberSpendReset}
          onMemberBudgetReset={mockOnMemberBudgetReset}
          setSelectedEditMember={mockSetSelectedEditMember}
          setIsEditMemberModalVisible={mockSetIsEditMemberModalVisible}
          setIsAddMemberModalVisible={mockSetIsAddMemberModalVisible}
        />,
      );

    it("labels each member's budget as Custom or Team default and shows the team amount for inherited members", () => {
      renderTab(teamDataWithDefault());

      const customRow = screen.getByRole("row", { name: /user1@test\.com/ });
      const inheritedRow = screen.getByRole("row", { name: /user2@test\.com/ });
      expect(within(customRow).getByTestId("member-budget-source")).toHaveTextContent("Custom");
      expect(customRow).toHaveTextContent("$1,000.00");
      expect(within(inheritedRow).getByTestId("member-budget-source")).toHaveTextContent("Team default");
      expect(inheritedRow).toHaveTextContent("$25.00");
    });

    it("caps a Custom member at the team default when the private row has no budget limit", () => {
      const base = createMockTeamData();
      renderTab(
        createMockTeamData({
          team_info: {
            ...base.team_info,
            team_member_budget_table: { max_budget: 20, budget_duration: null, tpm_limit: null, rpm_limit: null },
          },
          team_memberships: [
            {
              user_id: "user2@test.com",
              team_id: "team-123",
              budget_id: "budget2",
              budget_source: "custom",
              spend: 0,
              total_spend: 0,
              litellm_budget_table: {
                budget_id: "budget3",
                soft_budget: null,
                max_budget: null,
                max_parallel_requests: null,
                tpm_limit: null,
                rpm_limit: 100,
                model_max_budget: null,
                budget_duration: null,
                budget_reset_at: null,
              },
            },
          ],
        }),
      );

      const row = screen.getByRole("row", { name: /user2@test\.com/ });
      expect(within(row).getByTestId("member-budget-source")).toHaveTextContent("Custom");
      expect(row).toHaveTextContent("$20.00");
      expect(row).not.toHaveTextContent("Unlimited");
    });

    it("shows no source label for a member with neither a custom nor a team budget", () => {
      renderTab(createMockTeamData({ team_memberships: [] }));

      expect(screen.queryByTestId("member-budget-source")).not.toBeInTheDocument();
      expect(screen.queryByTestId("reset-member-budget")).not.toBeInTheDocument();
    });

    it("only offers Use team default on customized members, and only to editors", () => {
      const { unmount } = renderTab(teamDataWithDefault());

      expect(
        within(screen.getByRole("row", { name: /user1@test\.com/ })).getByTestId("reset-member-budget"),
      ).toBeVisible();
      expect(
        within(screen.getByRole("row", { name: /user2@test\.com/ })).queryByTestId("reset-member-budget"),
      ).not.toBeInTheDocument();

      unmount();
      renderTab(teamDataWithDefault(), false);
      expect(screen.queryByTestId("reset-member-budget")).not.toBeInTheDocument();
    });

    it("puts the member back on the team default after confirming, then refreshes the team", async () => {
      const user = userEvent.setup();
      POST.mockResolvedValue({ data: {} });
      renderTab(teamDataWithDefault());

      await user.click(screen.getByTestId("reset-member-budget"));

      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Budget" });
      expect(dialog).toHaveTextContent("user1@test.com");
      expect(dialog).toHaveTextContent("team default of $25.00");
      expect(dialog).toHaveTextContent("Custom budget: $1,000.00");
      expect(POST).not.toHaveBeenCalled();

      await user.click(within(dialog).getByRole("button", { name: "Use team default" }));

      await waitFor(() => expect(mockOnMemberBudgetReset).toHaveBeenCalledTimes(1));
      expect(POST).toHaveBeenCalledExactlyOnceWith("/team/{team_id}/member/{user_id}/reset_budget", {
        params: { path: { team_id: "team-123", user_id: "user1@test.com" } },
      });
      expect(mockOnMemberSpendReset).not.toHaveBeenCalled();
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("keeps the dialog open and does not refresh the team when the reset fails", async () => {
      const user = userEvent.setup();
      POST.mockRejectedValue(new Error("Team admin cannot reset budgets"));
      renderTab(teamDataWithDefault());

      await user.click(screen.getByTestId("reset-member-budget"));
      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Budget" });
      await user.click(within(dialog).getByRole("button", { name: "Use team default" }));

      await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
      expect(mockOnMemberBudgetReset).not.toHaveBeenCalled();
      expect(screen.getByRole("dialog", { name: "Reset Team Member Budget" })).toBeInTheDocument();
    });

    it("does not call the API when the dialog is cancelled", async () => {
      const user = userEvent.setup();
      renderTab(teamDataWithDefault());

      await user.click(screen.getByTestId("reset-member-budget"));
      const dialog = await screen.findByRole("dialog", { name: "Reset Team Member Budget" });
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(POST).not.toHaveBeenCalled();
    });
  });
});
