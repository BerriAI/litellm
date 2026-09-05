import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import UserInfoView from "./user_info_view";
import { extractMcpEntitlement } from "@/components/mcp_server_management/mcpEntitlement";

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

// entityLinks -> migratedPages imports serverRootPath from the same module, so the mock must export it too.
vi.mock("@/components/networking", () => {
  return {
    serverRootPath: "/",
    userGetInfoV2: (...args: any[]) => mockUserGetInfoV2(...args),
    userDeleteCall: vi.fn(),
    userUpdateUserCall: (...args: unknown[]) => mockUserUpdateUserCall(...args),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    invitationCreateCall: vi.fn(),
    teamInfoCall: (...args: any[]) => mockTeamInfoCall(...args),
    teamListCall: (...args: any[]) => mockTeamListCall(...args),
    teamMemberAddCall: (...args: any[]) => mockTeamMemberAddCall(...args),
    teamMemberDeleteCall: (...args: any[]) => mockTeamMemberDeleteCall(...args),
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

describe("UserInfoView", () => {
  const defaultProps = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "test-token",
    userRole: null as string | null,
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
    mockTeamMemberDeleteCall.mockResolvedValue({});
    mockUserUpdateUserCall.mockResolvedValue({});
    mockFetchMCPServers.mockResolvedValue([MCP_SERVER]);
    mockListMCPTools.mockResolvedValue({ tools: [{ name: "list_issues", description: "List issues" }] });
  });

  it("should render the loading state", () => {
    render(<UserInfoView {...defaultProps} />);

    expect(screen.getByText("Loading user data...")).toBeInTheDocument();
  });

  it("should render the user email after loading", async () => {
    render(<UserInfoView {...defaultProps} />);

    const emails = await screen.findAllByText("test@example.com");
    expect(emails.length).toBeGreaterThan(0);
    expect(screen.queryByText("Loading user data...")).not.toBeInTheDocument();
  });

  it("should render the user alias after loading", async () => {
    render(<UserInfoView {...defaultProps} />);

    const aliases = await screen.findAllByText("Test Alias");
    expect(aliases.length).toBeGreaterThan(0);
  });

  it("should render overview spend and budget with two decimal places", async () => {
    mockUserGetInfoV2.mockResolvedValue({
      ...MOCK_USER_DATA,
      spend: 98.854,
      max_budget: 3_000_000,
    });

    render(<UserInfoView {...defaultProps} />);

    expect(await screen.findByText("$98.85")).toBeInTheDocument();
    expect(screen.getByText(/of \$3,000,000\.00/)).toBeInTheDocument();
  });

  it("should render teams in a table with team names", async () => {
    render(<UserInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
      expect(screen.getByText("Beta Team")).toBeInTheDocument();
    });
  });

  it("should link each team name to its team page", async () => {
    render(<UserInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("link", { name: "Alpha Team" })).toHaveAttribute(
        "href",
        expect.stringContaining("/teams?team=team-1"),
      );
      expect(screen.getByRole("link", { name: "Beta Team" })).toHaveAttribute(
        "href",
        expect.stringContaining("/teams?team=team-2"),
      );
    });
  });

  it("should show 'No teams' when user has no teams", async () => {
    mockUserGetInfoV2.mockResolvedValue(MOCK_USER_DATA_NO_TEAMS);
    render(<UserInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("No teams")).toBeInTheDocument();
    });
  });

  it("should show Add Team button for proxy admins", async () => {
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" />);

    await waitFor(() => {
      expect(screen.getByText("Add Team")).toBeInTheDocument();
    });
  });

  it("should not show Add Team button for non-proxy-admins", async () => {
    render(<UserInfoView {...defaultProps} userRole="internal_user" />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    });
    expect(screen.queryByText("Add Team")).not.toBeInTheDocument();
  });

  it("should show delete buttons for proxy admins", async () => {
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    });
    // Should have the Actions column header
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should not show delete buttons for non-proxy-admins", async () => {
    render(<UserInfoView {...defaultProps} userRole="internal_user" />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    });
    expect(screen.queryByText("Actions")).not.toBeInTheDocument();
  });

  it("should open the add team modal when Add Team is clicked", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" />);

    await waitFor(() => {
      expect(screen.getByText("Add Team")).toBeInTheDocument();
    });

    await user.click(screen.getByText("Add Team"));

    await waitFor(() => {
      expect(screen.getByText("Add User to Team")).toBeInTheDocument();
    });
    expect(mockTeamListCall).toHaveBeenCalledWith("test-token", null);
  });

  it("should open remove confirmation modal when delete is clicked", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    });

    // Find the row with Alpha Team and click its delete button
    const alphaRow = screen.getByText("Alpha Team").closest("tr")!;
    const deleteButton = within(alphaRow).getByRole("button");
    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.getByText("Remove from Team")).toBeInTheDocument();
      expect(screen.getByText(/Removing this user from the team will also delete any keys/)).toBeInTheDocument();
    });
  });

  it("should call teamMemberDeleteCall when remove is confirmed", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" />);

    await waitFor(() => {
      expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    });

    // Click delete on Alpha Team
    const alphaRow = screen.getByText("Alpha Team").closest("tr")!;
    const deleteButton = within(alphaRow).getByRole("button");
    await user.click(deleteButton);

    // Confirm deletion
    await waitFor(() => {
      expect(screen.getByText("Remove from Team")).toBeInTheDocument();
    });

    // The DeleteResourceModal's OK button has text "Delete" - find it within the modal
    const modal = screen.getByRole("dialog", { name: "Remove from Team" });
    const deleteConfirmButton = within(modal).getByRole("button", { name: /delete/i });
    await user.click(deleteConfirmButton);

    await waitFor(() => {
      expect(mockTeamMemberDeleteCall).toHaveBeenCalledWith("test-token", "team-1", {
        role: "user",
        user_id: "user-123",
      });
    });
  });

  it("should keep the Details panel state while the Overview tab is shown", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...defaultProps} userRole="proxy_admin" initialTab={1} />);

    await user.click(await screen.findByText("GitHub MCP (srv-1)"));
    expect(await screen.findByText("list_issues")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Overview" }));
    expect(await screen.findByText(/of \$100\.00/)).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Details" }));

    expect(screen.getByText("list_issues")).toBeVisible();
  });

  describe("MCP permissions", () => {
    it("should render the user's MCP entitlements in read mode", async () => {
      const user = userEvent.setup();
      render(<UserInfoView {...defaultProps} userRole="proxy_admin" initialTab={1} />);

      await waitFor(() => {
        expect(screen.getByText("MCP Permissions")).toBeInTheDocument();
      });

      const grantedServer = await screen.findByText("GitHub MCP (srv-1)");
      expect(screen.getByText("dev-group")).toBeInTheDocument();
      expect(screen.queryByText("list_issues")).not.toBeInTheDocument();

      await user.click(grantedServer);

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
    });

    it("should nest MCP entitlements under object_permission when an admin saves", async () => {
      const user = userEvent.setup();
      render(<UserInfoView {...defaultProps} userRole="proxy_admin" initialTab={1} startInEditMode />);

      const saveButton = await screen.findByText("Save Changes");
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalledTimes(1);
      });

      const [token, payload, roleArg] = mockUserUpdateUserCall.mock.calls[0];
      expect(token).toBe("test-token");
      expect(roleArg).toBeNull();
      expect(payload.user_id).toBe("user-123");
      const expectedObjectPermission = {
        mcp_servers: ["srv-1"],
        mcp_access_groups: ["dev-group"],
        mcp_toolsets: [],
        mcp_tool_permissions: { "srv-1": ["list_issues"] },
      };
      expect(payload.object_permission).toEqual(expectedObjectPermission);
      expect(payload).not.toHaveProperty("mcp_servers_and_groups");
      expect(payload).not.toHaveProperty("mcp_tool_permissions");
      expect(payload).not.toHaveProperty("mcp_servers");
    });

    it("should send tool selections made in the edit form", async () => {
      const user = userEvent.setup();
      render(<UserInfoView {...defaultProps} userRole="proxy_admin" initialTab={1} startInEditMode />);

      await screen.findByText("Save Changes");
      await waitFor(() => {
        expect(mockListMCPTools).toHaveBeenCalledWith("test-token", "srv-1");
      });
      await waitFor(() => {
        expect(screen.queryByText("Loading tools...")).not.toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: "Deselect All" }));
      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalledTimes(1);
      });

      const [, payload] = mockUserUpdateUserCall.mock.calls[0];
      expect(payload.object_permission.mcp_tool_permissions).toEqual({ "srv-1": [] });
    });

    it("should preserve every tool allowlist when the granted servers are unchanged", async () => {
      const user = userEvent.setup();
      mockUserGetInfoV2.mockResolvedValue({
        ...MOCK_USER_DATA,
        object_permission: {
          mcp_servers: ["srv-1"],
          mcp_access_groups: ["group-a"],
          mcp_tool_permissions: { "srv-1": ["list_issues"], "srv-via-group": ["read_only"] },
        },
      });
      render(<UserInfoView {...defaultProps} userRole="proxy_admin" initialTab={1} startInEditMode />);

      const saveButton = await screen.findByText("Save Changes");
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalledTimes(1);
      });

      const [, payload] = mockUserUpdateUserCall.mock.calls[0];
      expect(payload.object_permission.mcp_tool_permissions).toEqual({
        "srv-1": ["list_issues"],
        "srv-via-group": ["read_only"],
      });
    });

    it("should not send object_permission for a non-admin editor", async () => {
      const user = userEvent.setup();
      render(<UserInfoView {...defaultProps} userRole="Internal User" initialTab={1} startInEditMode />);

      const saveButton = await screen.findByText("Save Changes");
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalledTimes(1);
      });

      const [, payload] = mockUserUpdateUserCall.mock.calls[0];
      expect(payload).not.toHaveProperty("object_permission");
      expect(screen.queryByText("MCP Servers / Access Groups")).not.toBeInTheDocument();
    });
  });
});

describe("extractMcpEntitlement", () => {
  const CATALOG = [
    { server_id: "srv-1", server_name: "deploy_tracker", alias: "deploy" },
    { server_id: "srv-2", server_name: "issue_tracker", alias: null },
    { server_id: "srv-via-group", server_name: "audit_log", alias: null, mcp_access_groups: ["ops_readonly"] },
  ] as any;

  const TOOLSETS = [
    { toolset_id: "ts-1", toolset_name: "audit", tools: [{ server_id: "srv-via-group", tool_name: "read" }] },
  ] as any;

  const form = (
    selection: { servers?: string[]; accessGroups?: string[]; toolsets?: string[] },
    toolPermissions: Record<string, string[]>,
  ) => ({
    mcp_servers_and_groups: {
      servers: selection.servers ?? [],
      accessGroups: selection.accessGroups ?? [],
      toolsets: selection.toolsets ?? [],
    },
    mcp_tool_permissions: toolPermissions,
  });

  it("drops the tool allowlist of a server the admin just deselected", () => {
    const result = extractMcpEntitlement(
      form({ servers: ["srv-1"] }, { "srv-1": ["read"], "srv-2": ["delete"] }),
      CATALOG,
    );
    expect(result?.mcp_tool_permissions).toEqual({ "srv-1": ["read"] });
  });

  it("drops the allowlist of a server reached only through an access group the admin removed", () => {
    const result = extractMcpEntitlement(form({}, { "srv-via-group": ["read"] }), CATALOG);
    expect(result?.mcp_tool_permissions).toEqual({});
  });

  it("keeps a name-keyed allowlist for a server that is still selected by id", () => {
    // The gateway resolves a tool-permission key by id, name OR alias, so an entry written by the
    // API or by config may be keyed by name. Comparing keys to the selector's ids alone drops it
    // while the server stays granted, which removes the restriction entirely.
    const result = extractMcpEntitlement(form({ servers: ["srv-1"] }, { deploy_tracker: ["create_issue"] }), CATALOG);
    expect(result?.mcp_tool_permissions).toEqual({ deploy_tracker: ["create_issue"] });
  });

  it("keeps an alias-keyed allowlist for a server that is still selected by id", () => {
    const result = extractMcpEntitlement(form({ servers: ["srv-1"] }, { deploy: ["create_issue"] }), CATALOG);
    expect(result?.mcp_tool_permissions).toEqual({ deploy: ["create_issue"] });
  });

  it("drops a name-keyed allowlist once its server is deselected", () => {
    const result = extractMcpEntitlement(form({ servers: ["srv-2"] }, { deploy_tracker: ["create_issue"] }), CATALOG);
    expect(result?.mcp_tool_permissions).toEqual({});
  });

  it("prunes nothing when the server catalog has not loaded", () => {
    // Every key is unresolvable without the catalog, and pruning while under-informed is the
    // direction that widens.
    const result = extractMcpEntitlement(form({ servers: ["srv-2"] }, { "srv-1": ["create_issue"] }), []);
    expect(result?.mcp_tool_permissions).toEqual({ "srv-1": ["create_issue"] });
  });

  it("keeps an entry whose key names no known server", () => {
    const result = extractMcpEntitlement(form({ servers: ["srv-1"] }, { "srv-deleted": ["read"] }), CATALOG);
    expect(result?.mcp_tool_permissions).toEqual({ "srv-deleted": ["read"] });
  });

  it.each([
    ["selected server first", ["srv-shared-a", "srv-shared-b"]],
    ["deselected server first", ["srv-shared-b", "srv-shared-a"]],
  ])("keeps a shared-name allowlist while any server it names is granted (%s)", (_label, order) => {
    // Names are not unique and the gateway unions a name key into EVERY server answering to it, so
    // this one entry restricts both. Resolving to the first match would drop it whenever the catalog
    // happened to return the deselected server first, stripping the restriction from the one still
    // granted, which is a widening that reproduces on one deployment and not another.
    const catalog = [
      { server_id: "srv-shared-a", server_name: "shared", alias: null },
      { server_id: "srv-shared-b", server_name: "shared", alias: null },
    ] as any;
    const ordered = order.map((id) => catalog.find((server: any) => server.server_id === id));

    const result = extractMcpEntitlement(form({ servers: ["srv-shared-a"] }, { shared: ["read"] }), ordered as any);
    expect(result?.mcp_tool_permissions).toEqual({ shared: ["read"] });
  });

  it("drops a shared-name allowlist once no server it names is granted", () => {
    const catalog = [
      { server_id: "srv-shared-a", server_name: "shared", alias: null },
      { server_id: "srv-shared-b", server_name: "shared", alias: null },
    ] as any;
    const result = extractMcpEntitlement(form({ servers: [] }, { shared: ["read"] }), catalog);
    expect(result?.mcp_tool_permissions).toEqual({});
  });

  it("keeps the allowlist of a server granted through a retained access group", () => {
    const result = extractMcpEntitlement(
      form({ servers: ["srv-1"], accessGroups: ["ops_readonly"] }, { "srv-1": ["read"], "srv-via-group": ["read"] }),
      CATALOG,
    );
    expect(result?.mcp_tool_permissions).toEqual({ "srv-1": ["read"], "srv-via-group": ["read"] });
  });

  it("keeps the allowlist of a deselected server that a retained access group still supplies", () => {
    const result = extractMcpEntitlement(
      form({ accessGroups: ["ops_readonly"] }, { "srv-via-group": ["read"] }),
      CATALOG,
      TOOLSETS,
    );
    expect(result?.mcp_tool_permissions).toEqual({ "srv-via-group": ["read"] });
  });

  it("drops the allowlist of a deselected server that the retained access group does not contain", () => {
    // The gateway grants each allowlist key as its own server, so retaining a group covering only
    // srv-via-group must not keep srv-1 callable after the admin removed it.
    const result = extractMcpEntitlement(
      form({ accessGroups: ["ops_readonly"] }, { "srv-1": ["read"] }),
      CATALOG,
      TOOLSETS,
    );
    expect(result?.mcp_tool_permissions).toEqual({});
  });

  it("keeps the allowlist of a deselected server that a retained toolset still supplies", () => {
    const result = extractMcpEntitlement(
      form({ toolsets: ["ts-1"] }, { "srv-via-group": ["read"] }),
      CATALOG,
      TOOLSETS,
    );
    expect(result?.mcp_tool_permissions).toEqual({ "srv-via-group": ["read"] });
  });

  it("drops the allowlist of a deselected server that the retained toolset does not cover", () => {
    const result = extractMcpEntitlement(form({ toolsets: ["ts-1"] }, { "srv-1": ["read"] }), CATALOG, TOOLSETS);
    expect(result?.mcp_tool_permissions).toEqual({});
  });

  it("prunes nothing when a selected toolset is missing from the toolset catalog", () => {
    // An unresolvable toolset could supply any server, so pruning against it would be a guess in
    // the widening direction.
    const result = extractMcpEntitlement(form({ toolsets: ["ts-unknown"] }, { "srv-1": ["read"] }), CATALOG, TOOLSETS);
    expect(result?.mcp_tool_permissions).toEqual({ "srv-1": ["read"] });
  });

  it("returns null when the MCP section was not rendered", () => {
    expect(extractMcpEntitlement({ user_email: "a@b.c" }, CATALOG)).toBeNull();
  });
});
