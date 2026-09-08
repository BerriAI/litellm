import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import UserInfoView from "./user_info_view";

const mockTeamMemberAddCall = vi.fn();
const mockTeamMemberDeleteCall = vi.fn();
const mockTeamListCall = vi.fn();
const mockUserGetInfoV2 = vi.fn();
const mockTeamInfoCall = vi.fn();
const mockUserUpdateUserCall = vi.fn();
const mockFetchMCPServers = vi.fn();
const mockListMCPTools = vi.fn();

const MCP_SERVER = { server_id: "srv-1", server_name: "GitHub MCP", alias: "GitHub MCP" };

const MOCK_USER_DATA = {
  user_id: "user-123",
  user_email: "test@example.com",
  user_alias: "Test Alias",
  user_role: "admin",
  spend: 0,
  max_budget: 100,
  models: [],
  budget_duration: "30d",
  budget_reset_at: null,
  metadata: {},
  created_at: "2025-01-01T00:00:00.000Z",
  updated_at: "2025-01-02T00:00:00.000Z",
  sso_user_id: null,
  teams: ["team-1", "team-2"],
  object_permission: {
    mcp_servers: ["srv-1"],
    mcp_access_groups: ["dev-group"],
    mcp_tool_permissions: { "srv-1": ["list_issues"] },
  },
};

const MOCK_USER_DATA_NO_TEAMS = {
  ...MOCK_USER_DATA,
  teams: [],
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/users",
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

vi.mock("@/components/networking", () => {
  return {
    serverRootPath: "/",
    userGetInfoV2: (...args: unknown[]) => mockUserGetInfoV2(...args),
    userDeleteCall: vi.fn(),
    userUpdateUserCall: (...args: unknown[]) => mockUserUpdateUserCall(...args),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    invitationCreateCall: vi.fn(),
    teamInfoCall: (...args: unknown[]) => mockTeamInfoCall(...args),
    teamListCall: (...args: unknown[]) => mockTeamListCall(...args),
    teamMemberAddCall: (...args: unknown[]) => mockTeamMemberAddCall(...args),
    teamMemberDeleteCall: (...args: unknown[]) => mockTeamMemberDeleteCall(...args),
    getProxyBaseUrl: () => "https://litellm.test",
    fetchMCPServers: (...args: unknown[]) => mockFetchMCPServers(...args),
    fetchMCPToolsets: vi.fn().mockResolvedValue([]),
    listMCPTools: (...args: unknown[]) => mockListMCPTools(...args),
  };
});

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [MCP_SERVER], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups", () => ({
  useMCPAccessGroups: () => ({ data: ["dev-group"], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: () => ({ data: [], isLoading: false }),
}));
describe("UserInfoView add-to-team form", () => {
  const defaultProps = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "test-token",
    userRole: "proxy_admin" as string | null,
    possibleUIRoles: null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUserGetInfoV2.mockResolvedValue(MOCK_USER_DATA);
    mockTeamInfoCall.mockImplementation((_token: string, teamId: string) => {
      const teamMap: Record<string, any> = {
        "team-1": { team_id: "team-1", team_info: { team_alias: "Alpha Team" } },
        "team-2": { team_id: "team-2", team_info: { team_alias: "Beta Team" } },
        "team-3": { team_id: "team-3", team_info: { team_alias: "Gamma Team" } },
      };
      return Promise.resolve(teamMap[teamId] || { team_id: teamId, team_info: { team_alias: null } });
    });
    mockTeamListCall.mockResolvedValue([
      { team_id: "team-1", team_alias: "Alpha Team" },
      { team_id: "team-2", team_alias: "Beta Team" },
      { team_id: "team-3", team_alias: "Gamma Team" },
    ]);
    mockTeamMemberAddCall.mockResolvedValue({});
    mockFetchMCPServers.mockResolvedValue([MCP_SERVER]);
    mockListMCPTools.mockResolvedValue({ tools: [] });
  });

  const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

  const openAddTeam = async (user: ReturnType<typeof userEvent.setup>, anchorTeam = "Alpha Team") => {
    render(<UserInfoView {...defaultProps} />);
    await screen.findByText(anchorTeam);
    await user.click(screen.getByText("Add Team"));
    await screen.findByText("Add User to Team");
  };

  const teamField = () => screen.getAllByRole("combobox")[0];
  const roleField = () => screen.getAllByRole("combobox")[1];
  const submitButton = () => screen.getByRole("button", { name: /Add to Team|Adding\.\.\./i });

  const chooseTeam = async (user: ReturnType<typeof userEvent.setup>, alias: string) => {
    await user.click(teamField());
    await user.click(await screen.findByTitle(alias));
  };

  // handleUserUpdate refreshes the local copy field by field rather than refetching,
  // so a field it forgets reads back stale the next time the form is opened and the
  // operator sees the save they just made apparently undone.
  describe("per-model budgets survive a save", () => {
    // Edit Settings lives on the details tab and is gated on write access.
    const budgetProps = {
      ...defaultProps,
      userRole: "Admin",
      initialTab: 1,
    };

    const openEditor = async (user: ReturnType<typeof userEvent.setup>) => {
      await user.click(await screen.findByRole("button", { name: /edit settings/i }));
      return screen.findByPlaceholderText("Max spend ($)");
    };

    beforeEach(() => {
      mockUserGetInfoV2.mockResolvedValue({
        ...MOCK_USER_DATA,
        model_max_budget: { "gpt-4": { budget_limit: 5, time_period: "30d" } },
      });
      mockUserUpdateUserCall.mockResolvedValue({});
    });

    it("shows the saved cap, not the pre-save one, when the form is reopened", async () => {
      const user = setup();
      render(<UserInfoView {...budgetProps} />);

      fireEvent.change(await openEditor(user), { target: { value: "42" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalled();
      });
      expect(mockUserUpdateUserCall.mock.calls[0][1].model_max_budget).toEqual({
        "gpt-4": { budget_limit: 42, time_period: "30d" },
      });

      expect(await openEditor(user)).toHaveValue(42);
    });
  });

  it("offers only the teams the user is not already a member of", async () => {
    const user = setup();
    await openAddTeam(user);

    await user.click(teamField());

    expect(await screen.findByTitle("Gamma Team")).toBeInTheDocument();
    expect(screen.queryByTitle("Alpha Team")).not.toBeInTheDocument();
    expect(screen.queryByTitle("Beta Team")).not.toBeInTheDocument();
  });

  it("shows the default role name on the trigger rather than a blank or raw value", async () => {
    const user = userEvent.setup();
    await openAddTeam(user);

    expect(screen.getAllByRole("combobox")[1]).toHaveTextContent("user");
  });

  it("adds the user to the chosen team with the default role", async () => {
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(mockTeamMemberAddCall).toHaveBeenCalledWith("test-token", "team-3", {
      role: "user",
      user_id: "user-123",
    });
  });

  it("sends the admin role when it is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(roleField());
    await user.click(await screen.findByText("admin", { selector: "span.font-medium" }));
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(mockTeamMemberAddCall).toHaveBeenCalledWith("test-token", "team-3", {
      role: "admin",
      user_id: "user-123",
    });
  });

  it("keeps the submit disabled until a team is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    expect(submitButton()).toBeDisabled();

    await chooseTeam(user, "Gamma Team");

    expect(submitButton()).toBeEnabled();
  });

  it("sends nothing while no team is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    await user.click(submitButton());

    expect(mockTeamMemberAddCall).not.toHaveBeenCalled();
  });

  it("narrows the team list to the aliases matching what was typed", async () => {
    mockUserGetInfoV2.mockResolvedValue(MOCK_USER_DATA_NO_TEAMS);
    const user = setup();
    render(<UserInfoView {...defaultProps} />);
    await screen.findByText("Add Team");
    await user.click(screen.getByText("Add Team"));
    await screen.findByText("Add User to Team");

    await user.click(teamField());
    await screen.findByTitle("Alpha Team");

    await user.keyboard("Gam");

    expect(await screen.findByTitle("Gamma Team")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTitle("Alpha Team")).not.toBeInTheDocument());
  });

  it("keeps the dialog open when the member could not be added", async () => {
    mockTeamMemberAddCall.mockRejectedValue(new Error("nope"));
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(screen.getByText("Add User to Team")).toBeInTheDocument();
  });
});
