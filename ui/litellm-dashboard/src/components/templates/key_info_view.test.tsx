import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "./key_info_view";

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(),
}));

describe("KeyInfoView", () => {
  beforeEach(() => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });
  });
  const MOCK_KEY_DATA: KeyResponse = {
    token: "test-token-123",
    token_id: "test-token-123",
    key_name: "sk-...TUuw",
    key_alias: "asdasdas",
    spend: 0,
    max_budget: 0,
    expires: "null",
    models: [],
    aliases: {},
    config: {},
    user_id: "default_user_id",
    team_id: null,
    max_parallel_requests: 10,
    metadata: {
      logging: [],
      tags: ["test-tag"],
    },
    tpm_limit: 10,
    rpm_limit: 10,
    duration: "30d",
    budget_duration: "30d",
    budget_reset_at: "never",
    allowed_cache_controls: [],
    allowed_routes: [],
    permissions: {},
    model_spend: {},
    model_max_budget: {},
    soft_budget_cooldown: false,
    blocked: false,
    litellm_budget_table: {},
    organization_id: null,
    created_at: "2025-10-29T01:26:41.613000Z",
    updated_at: "2025-10-29T01:47:33.980000Z",
    team_spend: 100,
    team_alias: "",
    team_tpm_limit: 100,
    team_rpm_limit: 100,
    team_max_budget: 100,
    team_models: [],
    team_blocked: false,
    soft_budget: 200,
    team_model_aliases: {},
    team_member_spend: 0,
    team_metadata: {},
    end_user_id: "default_user_id",
    end_user_tpm_limit: 10,
    end_user_rpm_limit: 10,
    end_user_max_budget: 0,
    last_refreshed_at: Date.now(),
    api_key: "sk-...TUuw",
    user_role: "user",
    rpm_limit_per_model: {},
    tpm_limit_per_model: {},
    user_tpm_limit: 10,
    user_rpm_limit: 10,
    user_email: "test@example.com",
    object_permission: {
      object_permission_id: "067002ed-3b01-4bb3-b942-cefa400f0049",
      mcp_servers: [],
      mcp_access_groups: [],
      mcp_tool_permissions: {},
      vector_stores: [],
    },
    auto_rotate: false,
    rotation_interval: undefined,
    last_rotation_at: undefined,
    key_rotation_at: undefined,
  };

  it("should render tags", async () => {
    const { getByText } = render(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={true}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(getByText("test-tag")).toBeInTheDocument();
    });
  });

  it("should not render tags in metadata textarea", async () => {
    const { container, getByText } = render(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={true}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(getByText("Metadata")).toBeInTheDocument();
      const metadataBlock = container.querySelector("pre");
      expect(metadataBlock).toBeInTheDocument();
      expect(metadataBlock?.textContent?.trim()).toBe("{}");
    });
  });

  it("should allow proxy admin to modify key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const keyData = { ...MOCK_KEY_DATA, user_id: "other-user-id" };
    render(
      <KeyInfoView
        keyData={keyData}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"proxy-admin-user"}
        userRole={"proxy_admin"}
        premiumUser={true}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
      expect(screen.getByText("Delete Key")).toBeInTheDocument();
    });
  });

  it("should allow team admin to modify key", async () => {
    const teamId = "test-team-id";
    const teamAdminUserId = "team-admin-user";
    const mockTeam: Team = {
      team_id: teamId,
      team_alias: "Test Team",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "org-1",
      created_at: "2025-01-01T00:00:00Z",
      keys: [],
      members_with_roles: [
        {
          user_id: teamAdminUserId,
          role: "admin",
        },
      ],
    };

    vi.mocked(useTeams).mockReturnValue({
      teams: [mockTeam],
      setTeams: vi.fn(),
    });

    const keyData = { ...MOCK_KEY_DATA, team_id: teamId, user_id: "other-user-id" };
    render(
      <KeyInfoView
        keyData={keyData}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={teamAdminUserId}
        userRole={"user"}
        premiumUser={true}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
      expect(screen.getByText("Delete Key")).toBeInTheDocument();
    });
  });

  it("should allow owner to modify their own key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const ownerUserId = "owner-user-id";
    const keyData = { ...MOCK_KEY_DATA, user_id: ownerUserId };
    render(
      <KeyInfoView
        keyData={keyData}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={ownerUserId}
        userRole={"user"}
        premiumUser={true}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
      expect(screen.getByText("Delete Key")).toBeInTheDocument();
    });
  });

  it("should not allow other user to modify key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const keyData = { ...MOCK_KEY_DATA, user_id: "owner-user-id" };
    render(
      <KeyInfoView
        keyData={keyData}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"other-user-id"}
        userRole={"user"}
        premiumUser={true}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Regenerate Key")).not.toBeInTheDocument();
      expect(screen.queryByText("Delete Key")).not.toBeInTheDocument();
    });
  });

  it("should not allow Internal Viewer to modify key even if they own it", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    const ownerUserId = "internal-viewer-user-id";
    const keyData = { ...MOCK_KEY_DATA, user_id: ownerUserId };
    render(
      <KeyInfoView
        keyData={keyData}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={ownerUserId}
        userRole={"Internal Viewer"}
        premiumUser={true}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Regenerate Key")).not.toBeInTheDocument();
      expect(screen.queryByText("Delete Key")).not.toBeInTheDocument();
    });
  });
});
