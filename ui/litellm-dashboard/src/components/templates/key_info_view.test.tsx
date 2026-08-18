import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { renderWithProviders } from "../../../tests/test-utils";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { keyDeleteCall, keyUpdateCall } from "../networking";
import { QueryClient } from "@tanstack/react-query";
import KeyInfoView from "./key_info_view";

const editViewMocks = vi.hoisted(() => ({
  onSubmit: undefined as ((v: Record<string, any>) => Promise<void>) | undefined,
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => ({ data: [] }),
}));

vi.mock("./key_edit_view", () => ({
  KeyEditView: ({ onSubmit }: { onSubmit: (v: Record<string, any>) => Promise<void> }) => {
    editViewMocks.onSubmit = onSubmit;
    return <div data-testid="key-edit-view-stub" />;
  },
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

const MCP_CATALOG = [
  { server_id: "srv-1", server_name: "deploy_tracker", alias: "deploy" },
  { server_id: "srv-2", server_name: "incident_log", alias: "incidents", mcp_access_groups: ["ops_readonly"] },
];

const MCP_TOOLSETS = [
  { toolset_id: "ts-1", toolset_name: "incidents", tools: [{ server_id: "srv-2", tool_name: "write" }] },
];

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: vi.fn(() => ({ data: MCP_CATALOG })),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: vi.fn(() => ({ data: MCP_TOOLSETS })),
}));

import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";

vi.mock("../networking", () => ({
  serverRootPath: "",
  keyDeleteCall: vi.fn().mockResolvedValue({}),
  keyUpdateCall: vi.fn().mockResolvedValue({}),
  getPolicyInfoWithGuardrails: vi.fn().mockResolvedValue({
    resolved_guardrails: ["guardrail-1", "guardrail-2"],
  }),
}));

const mockResetKeySpendMutate = vi.fn();
vi.mock("@/app/(dashboard)/hooks/keys/useResetKeySpend", () => ({
  useResetKeySpend: vi.fn(() => ({
    mutate: mockResetKeySpendMutate,
    isPending: false,
  })),
}));

vi.mock("@/utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
  formatNumberWithCommas: vi.fn((value: number, decimals?: number) => {
    return value.toFixed(decimals ?? 2);
  }),
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
    project_id: null,
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

  // Base mock for useAuthorized hook
  const baseUseAuthorizedMock = {
    accessToken: "test-token",
    userId: "test-user",
    userRole: "admin",
    premiumUser: true,
    token: "test-token",
    userEmail: null,
    disabledPersonalKeyCreation: null,
    showSSOBanner: false,
  };

  const openMoreKeyActions = async () => {
    await userEvent.click(await screen.findByRole("button", { name: /more key actions/i }));
  };

  describe("last updated", () => {
    const renderWithTimestamps = (overrides: Partial<KeyResponse>) => {
      vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

      return renderWithProviders(
        <KeyInfoView
          keyData={{
            ...MOCK_KEY_DATA,
            created_at: "2021-06-15T12:00:00Z",
            updated_at: "2023-06-15T12:00:00Z",
            ...overrides,
          }}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );
    };

    const findLastUpdatedText = async () => {
      const label = await screen.findByText("Last Updated");
      return label.closest("div")?.parentElement?.parentElement?.textContent ?? "";
    };

    it("should show when the key was last configured, not when it last recorded spend", async () => {
      renderWithTimestamps({ settings_updated_at: "2022-06-15T12:00:00Z" });

      expect(await findLastUpdatedText()).toMatch(/Jun \d+, 2022/);
      expect(screen.queryByText(/Jun \d+, 2023/)).not.toBeInTheDocument();
    });

    it("should fall back to creation time for a key that was never reconfigured", async () => {
      renderWithTimestamps({ settings_updated_at: null });

      expect(await findLastUpdatedText()).toMatch(/Jun \d+, 2021/);
      expect(screen.queryByText(/Jun \d+, 2023/)).not.toBeInTheDocument();
    });
  });

  it("should render the key's saved router fallbacks", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, router_settings: { num_retries: 2, fallbacks: [{ "gpt-4": ["gpt-4o"] }] } }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    expect(await screen.findByText("Router Settings")).toBeInTheDocument();
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("Number of Retries: 2")).toBeInTheDocument();
  });

  it("should render tags", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    renderWithProviders(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText("test-tag")).toBeInTheDocument();
    });
  });

  it("should not render tags in metadata textarea", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    const { container } = renderWithProviders(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText("Metadata")).toBeInTheDocument();
      const metadataBlock = container.querySelector("pre");
      expect(metadataBlock).toBeInTheDocument();
      expect(metadataBlock?.textContent?.trim()).toBe("{}");
    });
  });

  it("should render the estimated output token settings from key metadata", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    const keyData = {
      ...MOCK_KEY_DATA,
      metadata: {
        ...MOCK_KEY_DATA.metadata,
        default_estimated_output_tokens: 512,
        default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
      },
    };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    expect(await screen.findByText("Estimated Output Tokens: 512")).toBeInTheDocument();
    expect(await screen.findByText('Estimated Output Tokens Per Model: {"gpt-4":4096}')).toBeInTheDocument();
  });

  it("should fall back to Default when no estimated output tokens are configured", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    renderWithProviders(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    expect(await screen.findByText("Estimated Output Tokens: Default")).toBeInTheDocument();
    expect(await screen.findByText("Estimated Output Tokens Per Model: Default")).toBeInTheDocument();
  });

  it("should allow proxy admin to modify key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: "proxy-admin-user",
      userRole: "proxy_admin",
    });

    const keyData = { ...MOCK_KEY_DATA, user_id: "other-user-id" };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
    });
    await openMoreKeyActions();
    expect(await screen.findByRole("menuitem", { name: /delete key/i })).toBeInTheDocument();
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
      spend: 0,
    };

    vi.mocked(useTeams).mockReturnValue({
      teams: [mockTeam],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: teamAdminUserId,
      userRole: "user",
    });

    const keyData = { ...MOCK_KEY_DATA, team_id: teamId, user_id: "other-user-id" };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
    });
    await openMoreKeyActions();
    expect(await screen.findByRole("menuitem", { name: /delete key/i })).toBeInTheDocument();
  });

  it("should allow owner to modify their own key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: "owner-user-id",
      userRole: "user",
    });

    const ownerUserId = "owner-user-id";
    const keyData = { ...MOCK_KEY_DATA, user_id: ownerUserId };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.getByText("Regenerate Key")).toBeInTheDocument();
    });
    await openMoreKeyActions();
    expect(await screen.findByRole("menuitem", { name: /delete key/i })).toBeInTheDocument();
  });

  it("should not allow other user to modify key", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: "other-user-id",
      userRole: "user",
    });

    const keyData = { ...MOCK_KEY_DATA, user_id: "owner-user-id" };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Regenerate Key")).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /more key actions/i })).not.toBeInTheDocument();
    });
  });

  it("should not allow Internal Viewer to modify key even if they own it", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: "internal-viewer-user-id",
      userRole: "Internal Viewer",
    });

    const ownerUserId = "internal-viewer-user-id";
    const keyData = { ...MOCK_KEY_DATA, user_id: ownerUserId };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Regenerate Key")).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /more key actions/i })).not.toBeInTheDocument();
    });
  });

  it("should handle case when teamsData exists but no team matches key team_id", async () => {
    const differentTeamId = "different-team-id";
    const mockTeam: Team = {
      team_id: differentTeamId,
      team_alias: "Different Team",
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
          user_id: "team-admin-user",
          role: "admin",
        },
      ],
      spend: 0,
    };

    vi.mocked(useTeams).mockReturnValue({
      teams: [mockTeam],
      setTeams: vi.fn(),
    });

    vi.mocked(useAuthorized).mockReturnValue({
      ...baseUseAuthorizedMock,
      userId: "team-admin-user",
      userRole: "user",
    });

    const keyData = { ...MOCK_KEY_DATA, team_id: "non-matching-team-id", user_id: "other-user-id" };
    renderWithProviders(
      <KeyInfoView keyData={keyData} onClose={() => {}} keyId={"test-key-id"} onKeyDataUpdate={() => {}} teams={[]} />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Regenerate Key")).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /more key actions/i })).not.toBeInTheDocument();
    });
  });

  describe("entity links in the header", () => {
    const mockTeam: Team = {
      team_id: "linked-team-id",
      team_alias: "Linked Team",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "org-1",
      created_at: "2025-01-01T00:00:00Z",
      keys: [],
      members_with_roles: [],
      spend: 0,
    };

    beforeEach(() => {
      vi.mocked(useTeams).mockReturnValue({ teams: [mockTeam], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);
    });

    it("links the key's team by alias, resolved from the teams list, to the team page", async () => {
      const keyData = { ...MOCK_KEY_DATA, team_id: "linked-team-id" };
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );

      expect(await screen.findByRole("link", { name: "Linked Team" })).toHaveAttribute(
        "href",
        expect.stringContaining("/teams?team=linked-team-id"),
      );
    });

    it("links the key's user and creator to their user pages by id", async () => {
      const keyData = {
        ...MOCK_KEY_DATA,
        user_id: "owner-user-id",
        user_email: "owner@example.com",
        created_by: "creator-user-id",
        created_by_user: { user_id: "creator-user-id", user_email: "creator@example.com", user_alias: null },
      };
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );

      expect(await screen.findByRole("link", { name: "owner@example.com" })).toHaveAttribute(
        "href",
        expect.stringContaining("/users?user=owner-user-id"),
      );
      expect(screen.getByRole("link", { name: "creator@example.com" })).toHaveAttribute(
        "href",
        expect.stringContaining("/users?user=creator-user-id"),
      );
    });

    it("links the key's organization by id, falling back to the team's organization", async () => {
      const keyData = { ...MOCK_KEY_DATA, team_id: "linked-team-id", organization_id: null };
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );

      expect(await screen.findByRole("link", { name: "org-1" })).toHaveAttribute(
        "href",
        expect.stringContaining("/organizations?org=org-1"),
      );
    });

    it("renders no team link when the key has no team", async () => {
      renderWithProviders(
        <KeyInfoView
          keyData={{ ...MOCK_KEY_DATA, team_id: null }}
          onClose={() => {}}
          keyId="test-key-id"
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await screen.findByText("Team");
      expect(screen.queryByRole("link", { name: /team/i })).not.toBeInTheDocument();
    });
  });

  it("should call onClose when back button is clicked", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);
    const onCloseMock = vi.fn();

    renderWithProviders(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={onCloseMock}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /back to keys/i })).toBeInTheDocument();
    });

    const backButton = screen.getByRole("button", { name: /back to keys/i });
    await userEvent.click(backButton);

    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  describe("'Edit Settings' button visibility in the Settings tab", () => {
    const renderAndOpenSettingsTab = async (keyData = MOCK_KEY_DATA) => {
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );
      await waitFor(() => {
        expect(screen.getByRole("tab", { name: /settings/i })).toBeInTheDocument();
      });
      await userEvent.click(screen.getByRole("tab", { name: /settings/i }));
    };

    it("should show the Edit Settings button when the user is a proxy admin for a key they do not own", async () => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user-id",
        userRole: "proxy_admin",
      });

      await renderAndOpenSettingsTab({ ...MOCK_KEY_DATA, user_id: "someone-else-id" });

      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    it("should show the Edit Settings button when the user is the key owner", async () => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "owner-user-id",
        userRole: "Internal User",
      });

      await renderAndOpenSettingsTab({ ...MOCK_KEY_DATA, user_id: "owner-user-id" });

      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    it("should not show the Edit Settings button when an Internal User does not own the key", async () => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "non-owner-user-id",
        userRole: "Internal User",
      });

      await renderAndOpenSettingsTab({ ...MOCK_KEY_DATA, user_id: "owner-user-id" });

      expect(screen.queryByRole("button", { name: /edit settings/i })).not.toBeInTheDocument();
    });

    it("should not show the Edit Settings button when the user is an Internal Viewer even if they own the key", async () => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "owner-user-id",
        userRole: "Internal Viewer",
      });

      await renderAndOpenSettingsTab({ ...MOCK_KEY_DATA, user_id: "owner-user-id" });

      expect(screen.queryByRole("button", { name: /edit settings/i })).not.toBeInTheDocument();
    });

    it("should show the Edit Settings button when the user is a team admin for the key's team", async () => {
      const teamId = "test-team-id";
      const teamAdminUserId = "team-admin-user";
      vi.mocked(useTeams).mockReturnValue({
        teams: [
          {
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
            members_with_roles: [{ user_id: teamAdminUserId, role: "admin" }],
            spend: 0,
          },
        ],
        setTeams: vi.fn(),
      });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: teamAdminUserId,
        userRole: "user",
      });

      await renderAndOpenSettingsTab({ ...MOCK_KEY_DATA, team_id: teamId, user_id: "other-user-id" });

      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });
  });

  it("should display guardrails when present", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    const keyDataWithGuardrails = {
      ...MOCK_KEY_DATA,
      metadata: {
        ...MOCK_KEY_DATA.metadata,
        guardrails: ["guardrail-1", "guardrail-2"],
      },
    };

    renderWithProviders(
      <KeyInfoView
        keyData={keyDataWithGuardrails}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Guardrails")).toBeInTheDocument();
    });
  });

  it("should display policies when present", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    const keyDataWithPolicies = {
      ...MOCK_KEY_DATA,
      metadata: {
        ...MOCK_KEY_DATA.metadata,
        policies: ["policy-1"],
      },
    };

    renderWithProviders(
      <KeyInfoView
        keyData={keyDataWithPolicies}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Policies")).toBeInTheDocument();
    });
  });

  it("should display no key found message when keyData is undefined", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseUseAuthorizedMock);

    renderWithProviders(
      <KeyInfoView
        keyData={undefined}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Key not found")).toBeInTheDocument();
    });
  });

  describe("Reset Spend button visibility", () => {
    it("should show Reset Spend button for proxy admin", async () => {
      vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });

      renderWithProviders(
        <KeyInfoView
          keyData={MOCK_KEY_DATA}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      expect(await screen.findByRole("menuitem", { name: /reset spend/i })).toBeInTheDocument();
    });

    it("should show Reset Spend button for team admin of key's team", async () => {
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
        members_with_roles: [{ user_id: teamAdminUserId, role: "admin" }],
        spend: 0,
      };

      vi.mocked(useTeams).mockReturnValue({ teams: [mockTeam], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: teamAdminUserId,
        userRole: "user",
      });

      const keyData = { ...MOCK_KEY_DATA, team_id: teamId, user_id: "other-user-id" };
      renderWithProviders(
        <KeyInfoView
          keyData={keyData}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      expect(await screen.findByRole("menuitem", { name: /reset spend/i })).toBeInTheDocument();
    });

    it("should not show Reset Spend button for regular key owner", async () => {
      vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "owner-user-id",
        userRole: "user",
      });

      const keyData = { ...MOCK_KEY_DATA, user_id: "owner-user-id" };
      renderWithProviders(
        <KeyInfoView
          keyData={keyData}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      expect(await screen.findByRole("menuitem", { name: /delete key/i })).toBeInTheDocument();
      expect(screen.queryByRole("menuitem", { name: /reset spend/i })).not.toBeInTheDocument();
    });
  });

  describe("Reset Spend modal flow", () => {
    it("should open confirmation modal when Reset Spend is clicked", async () => {
      vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });

      renderWithProviders(
        <KeyInfoView
          keyData={MOCK_KEY_DATA}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      await userEvent.click(await screen.findByRole("menuitem", { name: /reset spend/i }));

      await waitFor(() => {
        expect(screen.getByText("Reset Key Spend")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: /^reset$/i })).toBeInTheDocument();
      });
    });

    it("should call mutate with token on confirm", async () => {
      vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });

      const keyDataWithSpend = { ...MOCK_KEY_DATA, spend: 5.0 };
      renderWithProviders(
        <KeyInfoView
          keyData={keyDataWithSpend}
          onClose={() => {}}
          keyId={"test-key-id"}
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      await userEvent.click(await screen.findByRole("menuitem", { name: /reset spend/i }));

      await waitFor(() => {
        expect(screen.getByText("Reset Key Spend")).toBeInTheDocument();
      });

      // Click the confirm button in the modal
      await userEvent.click(screen.getByRole("button", { name: /^reset$/i }));

      await waitFor(() => {
        expect(mockResetKeySpendMutate).toHaveBeenCalledWith(
          MOCK_KEY_DATA.token,
          expect.objectContaining({ onSuccess: expect.any(Function), onError: expect.any(Function) }),
        );
      });
    });
  });

  describe("premium metadata payload normalization", () => {
    const enterEditMode = async (keyData: KeyResponse) => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );
      await userEvent.click(screen.getByRole("tab", { name: /settings/i }));
      await userEvent.click(screen.getByRole("button", { name: /edit settings/i }));
      await waitFor(() => expect(editViewMocks.onSubmit).toBeDefined());
    };

    beforeEach(() => {
      editViewMocks.onSubmit = undefined;
      vi.mocked(keyUpdateCall).mockClear();
      vi.mocked(keyUpdateCall).mockResolvedValue({});
    });

    it("should drop an empty policies field when the key previously had no policies", async () => {
      // Reproduces the real bug: after a successful /key/update, the response echoes
      // top-level `policies: []` into client state. Without stripping, the next save
      // resends `[]` and trips the premium gate in prepare_metadata_fields.
      const keyData: KeyResponse = {
        ...MOCK_KEY_DATA,
        user_id: "proxy-admin-user",
        metadata: {},
        policies: [],
      } as KeyResponse;

      await enterEditMode(keyData);
      await editViewMocks.onSubmit!({ key: keyData.token, token: keyData.token, policies: [] });

      expect(keyUpdateCall).toHaveBeenCalledWith(
        expect.anything(),
        expect.not.objectContaining({ policies: expect.anything() }),
      );
    });

    it("should keep an empty policies field when the key previously had policies set", async () => {
      // Premium users must still be able to clear existing policies by sending `[]`.
      const keyData: KeyResponse = {
        ...MOCK_KEY_DATA,
        user_id: "proxy-admin-user",
        metadata: { policies: ["existing-policy"] },
        policies: ["existing-policy"],
      } as KeyResponse;

      await enterEditMode(keyData);
      await editViewMocks.onSubmit!({ key: keyData.token, token: keyData.token, policies: [] });

      expect(keyUpdateCall).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ policies: [] }));
    });

    it("should keep an empty policies field when the previous value lives only at the top level of keyData", async () => {
      // Defensive: some premium fields may be present at the top level but not
      // mirrored into metadata. A genuine clear must still be forwarded.
      const keyData: KeyResponse = {
        ...MOCK_KEY_DATA,
        user_id: "proxy-admin-user",
        metadata: {},
        policies: ["existing-policy"],
      } as KeyResponse;

      await enterEditMode(keyData);
      await editViewMocks.onSubmit!({ key: keyData.token, token: keyData.token, policies: [] });

      expect(keyUpdateCall).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ policies: [] }));
    });
  });

  describe("MCP tool permissions on save", () => {
    const KEY_WITH_TOOL_PERMISSIONS: KeyResponse = {
      ...MOCK_KEY_DATA,
      user_id: "proxy-admin-user",
      object_permission: {
        ...MOCK_KEY_DATA.object_permission,
        mcp_servers: ["srv-1", "srv-2"],
        mcp_access_groups: [],
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      },
    } as KeyResponse;

    const enterEditMode = async (keyData: KeyResponse) => {
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });
      renderWithProviders(
        <KeyInfoView keyData={keyData} onClose={() => {}} keyId="test-key-id" onKeyDataUpdate={() => {}} teams={[]} />,
      );
      await userEvent.click(screen.getByRole("tab", { name: /settings/i }));
      await userEvent.click(screen.getByRole("button", { name: /edit settings/i }));
      await waitFor(() => expect(editViewMocks.onSubmit).toBeDefined());
    };

    const submittedToolPermissions = () => {
      const payload = vi.mocked(keyUpdateCall).mock.calls.at(-1)?.[1] as Record<string, any>;
      return payload.object_permission.mcp_tool_permissions;
    };

    beforeEach(() => {
      editViewMocks.onSubmit = undefined;
      vi.mocked(keyUpdateCall).mockClear();
      vi.mocked(keyUpdateCall).mockResolvedValue({});
      vi.mocked(useMCPServers).mockReturnValue({ data: MCP_CATALOG } as unknown as ReturnType<typeof useMCPServers>);
      vi.mocked(useMCPToolsets).mockReturnValue({ data: MCP_TOOLSETS } as unknown as ReturnType<typeof useMCPToolsets>);
    });

    it("drops the allowlist of every deselected server instead of leaving it entitled", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({});
    });

    it("drops only the deselected server and keeps the one still granted", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ "srv-1": ["read"] });
    });

    it("keeps an allowlist whose server is reachable through a retained access group", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: ["ops_readonly"], toolsets: [] },
        mcp_tool_permissions: { "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ "srv-2": ["write"] });
    });

    it("drops an allowlist the retained access group does not reach", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: ["ops_readonly"], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ "srv-2": ["write"] });
    });

    it("drops an allowlist the retained toolset does not cover", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: ["ts-1"] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ "srv-2": ["write"] });
    });

    it("refuses to save a permission change while a selected toolset is unresolvable", async () => {
      vi.mocked(useMCPToolsets).mockReturnValue({ data: undefined } as unknown as ReturnType<typeof useMCPToolsets>);
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: ["ts-1"] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(keyUpdateCall).not.toHaveBeenCalled();
    });

    it("resolves a name-keyed allowlist against the server catalog", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { deploy_tracker: ["read"], incident_log: ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ deploy_tracker: ["read"] });
    });

    it("clears every allowlist when the admin picks the no-MCP-servers sentinel", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: ["no-mcp-servers"], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({});
    });

    it("keeps every allowlist when the admin grants all proxy servers", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: ["all-proxy-mcpservers"], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(submittedToolPermissions()).toEqual({ "srv-1": ["read"], "srv-2": ["write"] });
    });

    it("refuses to save a permission change it cannot compute without the server catalog", async () => {
      vi.mocked(useMCPServers).mockReturnValue({ data: undefined } as ReturnType<typeof useMCPServers>);
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"], "srv-2": ["write"] },
      });

      expect(keyUpdateCall).not.toHaveBeenCalled();
    });

    it("preserves a vector-store edit made in the same save", async () => {
      await enterEditMode(KEY_WITH_TOOL_PERMISSIONS);
      await editViewMocks.onSubmit!({
        key: KEY_WITH_TOOL_PERMISSIONS.token,
        token: KEY_WITH_TOOL_PERMISSIONS.token,
        vector_stores: ["vs-1"],
        mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: [], toolsets: [] },
        mcp_tool_permissions: { "srv-1": ["read"] },
      });

      const payload = vi.mocked(keyUpdateCall).mock.calls.at(-1)?.[1] as Record<string, any>;
      expect(payload.object_permission.vector_stores).toEqual(["vs-1"]);
    });
  });

  describe("delete flow", () => {
    it("invalidates the keys list query after a successful delete so active filters survive (LIT-4080)", async () => {
      const invalidateSpy = vi.spyOn(QueryClient.prototype, "invalidateQueries");
      vi.mocked(useAuthorized).mockReturnValue({
        ...baseUseAuthorizedMock,
        userId: "proxy-admin-user",
        userRole: "proxy_admin",
      });

      renderWithProviders(
        <KeyInfoView
          keyData={MOCK_KEY_DATA}
          onClose={() => {}}
          keyId="test-key-id"
          onKeyDataUpdate={() => {}}
          teams={[]}
        />,
      );

      await openMoreKeyActions();
      await userEvent.click(await screen.findByRole("menuitem", { name: /delete key/i }));

      const confirmInput = await screen.findByPlaceholderText(MOCK_KEY_DATA.key_alias);
      await userEvent.type(confirmInput, MOCK_KEY_DATA.key_alias);
      await userEvent.click(screen.getByRole("button", { name: /^delete$/i }));

      await waitFor(() => {
        expect(keyDeleteCall).toHaveBeenCalledWith("test-token", MOCK_KEY_DATA.token);
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["keys", "list"] });
      });

      invalidateSpy.mockRestore();
    });
  });
});
