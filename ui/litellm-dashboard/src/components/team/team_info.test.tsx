import { afterEach, describe, expect, it, vi } from "vitest";
import TeamInfoView from "./team_info";
import { render, waitFor } from "@testing-library/react";
import * as networking from "@/components/networking";

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

    vi.mocked(networking.getGuardrailsList).mockResolvedValue([]);
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    const { getByText } = render(
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
    await waitFor(() => {
      expect(getByText("User ID")).toBeInTheDocument();
    });
  });
});
