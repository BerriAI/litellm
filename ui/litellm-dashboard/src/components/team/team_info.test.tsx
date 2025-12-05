import * as networking from "@/components/networking";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import TeamInfoView from "./team_info";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  teamInfoCall: vi.fn(),
  teamMemberDeleteCall: vi.fn(),
  teamMemberAddCall: vi.fn(),
  teamMemberUpdateCall: vi.fn(),
  teamUpdateCall: vi.fn(),
  getGuardrailsList: vi.fn(),
  fetchMCPAccessGroups: vi.fn(),
  getTeamPermissionsCall: vi.fn(),
  organizationInfoCall: vi.fn(),
}));

describe("TeamInfoView", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", async () => {
    // Mock the team info response
    vi.mocked(networking.teamInfoCall).mockResolvedValue({
      team_id: "123",
      team_info: {
        team_alias: "Test Team",
        team_id: "123",
        organization_id: null,
        admins: ["admin@test.com"],
        members: ["user1@test.com", "user2@test.com"],
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
      },
      keys: [],
      team_memberships: [],
    });

    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    render(
      <TeamInfoView
        teamId="123"
        onUpdate={() => {}}
        onClose={() => {}}
        accessToken="123"
        is_team_admin={true}
        is_proxy_admin={true}
        userModels={[]}
        editTeam={false}
        premiumUser={false}
      />,
    );
    await waitFor(
      () => {
        expect(screen.queryByText("User ID")).not.toBeNull();
      },
      // This is a workaround to fix the flaky test issue. TODO: Remove this once we have a better solution.
      { timeout: 10000 },
    );
  });

  it("should not show all-proxy-models option when user has no access to it", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue({
      team_id: "123",
      team_info: {
        team_alias: "Test Team",
        team_id: "123",
        organization_id: null,
        admins: ["admin@test.com"],
        members: ["user1@test.com", "user2@test.com"],
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
        models: ["gpt-4"],
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
      team_memberships: [],
    });

    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    render(
      <TeamInfoView
        teamId="123"
        onUpdate={() => {}}
        onClose={() => {}}
        accessToken="123"
        is_team_admin={true}
        is_proxy_admin={true}
        userModels={["gpt-4", "gpt-3.5-turbo"]}
        editTeam={false}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getAllByText("Test Team")).not.toBeNull();
    });

    const settingsTab = screen.getByRole("tab", { name: "Settings" });
    act(() => {
      fireEvent.click(settingsTab);
    });

    await waitFor(() => {
      expect(screen.getByText("Team Settings")).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    act(() => {
      fireEvent.click(editButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Models")).toBeInTheDocument();
    });

    const allProxyModelsOption = screen.queryByText("All Proxy Models");
    expect(allProxyModelsOption).not.toBeInTheDocument();
  }, 10000); // This is a workaround to fix the flaky test issue. TODO: Remove this once we have a better solution.

  it("should only show organization models in dropdown when team is in organization with limited models", async () => {
    const organizationId = "org-123";
    const organizationModels = ["gpt-4", "claude-3-opus"];
    const userModels = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-2"];

    vi.mocked(networking.teamInfoCall).mockResolvedValue({
      team_id: "123",
      team_info: {
        team_alias: "Test Team",
        team_id: "123",
        organization_id: organizationId,
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
        models: ["gpt-4"],
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
      team_memberships: [],
    });

    vi.mocked(networking.organizationInfoCall).mockResolvedValue({
      organization_id: organizationId,
      organization_name: "Test Organization",
      spend: 0,
      max_budget: null,
      models: organizationModels,
      tpm_limit: null,
      rpm_limit: null,
      members: null,
    });

    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    render(
      <TeamInfoView
        teamId="123"
        onUpdate={() => {}}
        onClose={() => {}}
        accessToken="123"
        is_team_admin={true}
        is_proxy_admin={true}
        userModels={userModels}
        editTeam={false}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getAllByText("Test Team")).not.toBeNull();
    });

    const settingsTab = screen.getByRole("tab", { name: "Settings" });
    act(() => {
      fireEvent.click(settingsTab);
    });

    await waitFor(() => {
      expect(screen.getByText("Team Settings")).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    act(() => {
      fireEvent.click(editButton);
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Models")).toBeInTheDocument();
    });

    const modelsSelect = screen.getByLabelText("Models");
    act(() => {
      fireEvent.mouseDown(modelsSelect);
    });

    await waitFor(() => {
      const dropdownOptions = screen.getAllByRole("option");
      const optionTexts = dropdownOptions.map((option) => option.textContent);

      organizationModels.forEach((model) => {
        expect(optionTexts).toContain(model);
      });

      const modelsNotInOrganization = userModels.filter((m) => !organizationModels.includes(m));
      modelsNotInOrganization.forEach((model) => {
        expect(optionTexts).not.toContain(model);
      });
    });
  }, 10000);
});
