import * as networking from "@/components/networking";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { toast } from "@/lib/toast";
import { storeLoginToken, clearTokenCookies } from "@/utils/cookieUtils";
import type { Team } from "../key_team_helpers/key_list";
import TeamInfoView, { type TeamData } from "./TeamInfo";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  teamInfoCall: vi.fn(),
  teamUpdateCall: vi.fn(),
  getUiConfig: vi.fn().mockResolvedValue({ admin_ui_disabled: false }),
  getUiSettings: vi.fn().mockResolvedValue({ values: {} }),
  userGetInfoV2: vi.fn().mockResolvedValue({ models: [], user_info: {} }),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  organizationListCall: vi.fn().mockResolvedValue([]),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
  fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
  getTeamPermissionsCall: vi.fn().mockResolvedValue({ all_available_permissions: [], team_member_permissions: [] }),
  getRouterSettingsCall: vi.fn().mockResolvedValue({ fields: [] }),
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: [] }),
  fetchMCPServers: vi.fn().mockResolvedValue([]),
  fetchMCPToolsets: vi.fn().mockResolvedValue([]),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  getClaudeCodePluginsList: vi.fn().mockResolvedValue({ plugins: [], count: 0 }),
  keyListCall: vi.fn().mockResolvedValue({ keys: [], total_count: 0, total_pages: 1 }),
}));

const createMockTeamData = (overrides = {}) => ({
  team_id: "123",
  team_info: {
    team_alias: "Test Team",
    team_id: "123",
    organization_id: null,
    admins: ["admin@test.com"],
    members: ["user1@test.com"],
    members_with_roles: [
      {
        user_id: "user1@test.com",
        user_email: "user1@test.com",
        role: "member",
        spend: 0,
        budget_id: "budget1",
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
    guardrails: [],
    policies: [],
    object_permission: null,
    ...overrides,
  },
  keys: [],
  team_memberships: [],
});

describe("TeamInfoView - atomic member budget updates", () => {
  const props = {
    teamId: "123",
    onUpdate: vi.fn(),
    onClose: vi.fn(),
    accessToken: "test-token",
    is_team_admin: true,
    is_proxy_admin: true,
    userModels: ["gpt-4", "gpt-3.5-turbo"],
    editTeam: false,
    premiumUser: false,
  };

  const customBudgetMembership = (
    userId: string,
    maxBudget: number | null = 50,
  ): TeamData["team_memberships"][number] => ({
    user_id: userId,
    team_id: "123",
    budget_id: `budget-${userId}`,
    budget_source: "custom",
    spend: 0,
    total_spend: 0,
    litellm_budget_table: {
      budget_id: `budget-${userId}`,
      soft_budget: null,
      max_budget: maxBudget,
      max_parallel_requests: null,
      tpm_limit: null,
      rpm_limit: null,
      model_max_budget: null,
      budget_duration: null,
      budget_reset_at: null,
    },
  });

  const savedTeam: Team = {
    team_id: "123",
    team_alias: "Test Team",
    models: [],
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    organization_id: "org-1",
    created_at: "2024-01-01T00:00:00Z",
    keys: [],
    members_with_roles: [],
    spend: 0,
  };

  const openEditorWithCustomMembers = async (
    user: ReturnType<typeof userEvent.setup>,
    userIds: string[] = ["user-custom"],
    maxBudget: number | null = 50,
  ) => {
    const data = {
      ...createMockTeamData({
        team_member_budget_table: { max_budget: 10, budget_duration: null, tpm_limit: null, rpm_limit: null },
      }),
      team_memberships: userIds.map((id) => customBudgetMembership(id, maxBudget)),
    } as TeamData;
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    return await screen.findByLabelText("Default Budget (USD)");
  };

  beforeEach(() => {
    testQueryClient.clear();
    const payload = btoa(JSON.stringify({ user_id: "user-1", user_role: "proxy_admin", key: "test-token" }));
    storeLoginToken(`header.${payload}.signature`);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
        switch (url.pathname) {
          case "/v1/access_group":
            return Response.json([]);
          case "/team/metadata_schema":
            return Response.json({ fields: [] });
          case "/key/list":
            return Response.json({ keys: [], total_count: 0, total_pages: 1 });
          default:
            throw new Error(`Unexpected request: ${url.pathname}`);
        }
      }),
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    clearTokenCookies();
    testQueryClient.clear();
  });

  it("keeps overrides by default and saves without another dialog", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    fireEvent.change(input, { target: { value: "20" } });
    expect(screen.getByLabelText("Existing member budgets")).toHaveTextContent("Keep custom budgets");
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1]).toEqual(
      expect.objectContaining({
        team_member_budget: 20,
        team_member_budget_update_mode: "keep",
      }),
    );
  });

  it.each([
    ["Raise smaller budgets", "raise"],
    ["Lower larger budgets", "lower"],
    ["Raise and lower", "both"],
  ])("sends %s as one atomic save and reports the server count", async (label, mode) => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({
      data: savedTeam,
      team_id: "123",
      member_budgets_updated: 7,
    });
    fireEvent.change(input, { target: { value: "20" } });
    await user.click(screen.getByLabelText("Existing member budgets"));
    await user.click(await screen.findByRole("option", { name: label }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1]).toEqual(
      expect.objectContaining({
        team_member_budget: 20,
        team_member_budget_update_mode: mode,
      }),
    );
    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith(
        "Team settings updated. 7 member budgets now follow the team default amount",
      ),
    );
  });

  it.each(["10", "0", ""])("does not send a bulk action when the amount becomes %s", async (amount) => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    fireEvent.change(input, { target: { value: "20" } });
    await user.click(screen.getByLabelText("Existing member budgets"));
    await user.click(await screen.findByRole("option", { name: "Raise and lower" }));
    fireEvent.change(input, { target: { value: amount } });
    expect(screen.queryByLabelText("Existing member budgets")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget_update_mode).toBeUndefined();
  });

  it("preserves the chosen action after a failed save so the whole operation can be retried", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    vi.mocked(networking.teamUpdateCall).mockRejectedValueOnce(new Error("transaction rolled back"));
    fireEvent.change(input, { target: { value: "20" } });
    await user.click(screen.getByLabelText("Existing member budgets"));
    await user.click(await screen.findByRole("option", { name: "Lower larger budgets" }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /save changes/i })).toBeEnabled());
    expect(screen.getByLabelText("Existing member budgets")).toHaveTextContent("Lower larger budgets");
    expect(toast.success).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(2));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[1][1].team_member_budget_update_mode).toBe("lower");
  });
});
