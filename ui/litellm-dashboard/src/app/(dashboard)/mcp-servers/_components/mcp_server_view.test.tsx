import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MCPServerView } from "./mcp_server_view";
import * as networking from "@/components/networking";
import { setSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";
import { copyToClipboard } from "@/utils/dataUtils";
import type { MCPServer } from "@/components/mcp_tools/types";

vi.mock(".", () => ({
  MCPToolsViewer: () => <div>tools viewer</div>,
}));

vi.mock("./mcp_server_edit", () => ({
  default: ({ onCancel, onSuccess }: { onCancel: () => void; onSuccess: () => void }) => (
    <div>
      edit form<button onClick={onCancel}>Cancel edit</button>
      <button onClick={onSuccess}>Save edit</button>
    </div>
  ),
  EDIT_OAUTH_UI_STATE_KEY: "litellm-mcp-oauth-edit-state",
}));

vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn() }));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  fetchMCPServerUserCredentials: vi.fn(),
  revokeMCPServerUserCredential: vi.fn(),
}));

const baseServer = {
  server_id: "srv-1",
  server_name: "demo server",
  alias: "demo_alias",
  description: "A demo MCP server",
  transport: "http",
  url: "https://example.com/mcp",
  auth_type: "api_key",
} as MCPServer;

const renderView = (overrides: Partial<MCPServer> = {}, props: Record<string, unknown> = {}) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <MCPServerView
        mcpServer={{ ...baseServer, ...overrides } as MCPServer}
        onBack={vi.fn()}
        isProxyAdmin
        isEditing={false}
        accessToken="tok"
        userRole="Admin"
        userID="u1"
        availableAccessGroups={[]}
        {...props}
      />
    </QueryClientProvider>,
  );

const openUserCredentials = async (props: Record<string, unknown>) => {
  vi.mocked(networking.fetchMCPServerUserCredentials).mockResolvedValue([
    {
      user_id: "alice",
      credential_type: "byok",
      expires_at: null,
      connected_at: null,
      updated_at: "2026-01-01T00:00:00+00:00",
    },
  ]);
  renderView({}, props);
  await userEvent.click(screen.getByRole("tab", { name: "User Credentials" }));
  return within(await screen.findByRole("region", { name: "Stored user credentials" })).getByRole("row", {
    name: /alice/,
  });
};

describe("MCPServerView", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
  });

  // Name, alias and description each label the header and a Settings row, so
  // only the server id is unique to the header.
  it("shows the server identity in the header", () => {
    renderView();

    expect(screen.getByText("srv-1")).toBeInTheDocument();
    expect(screen.getAllByText("demo server").length).toBeGreaterThan(0);
    expect(screen.getAllByText("A demo MCP server").length).toBeGreaterThan(0);
    expect(screen.getAllByText("demo_alias").length).toBeGreaterThan(0);
  });

  it("falls back to a placeholder name when the server has neither name nor alias", () => {
    renderView({ server_name: undefined, alias: undefined });

    expect(screen.getByText("Unnamed Server")).toBeInTheDocument();
  });

  // "Transport" and "Authentication" label both an Overview card and a Settings
  // row, so only Overview-exclusive labels identify the Overview panel.
  it("summarises the connection on the Overview tab", () => {
    renderView();

    expect(screen.getByText("Host URL")).toBeInTheDocument();
    expect(screen.getByText("Cost Configuration")).toBeInTheDocument();
    expect(screen.getAllByText("HTTP").length).toBeGreaterThan(0);
    expect(screen.getAllByText("https://example.com/mcp").length).toBeGreaterThan(0);
  });

  it("offers a Settings tab to proxy admins only", () => {
    renderView();
    expect(screen.getByRole("tab", { name: "Settings" })).toBeInTheDocument();
  });

  it("hides the Settings tab from non-admins", () => {
    renderView({}, { isProxyAdmin: false });
    expect(screen.queryByRole("tab", { name: "Settings" })).not.toBeInTheDocument();
  });

  it("opens the tools viewer on the MCP Tools tab", async () => {
    renderView();

    await userEvent.click(screen.getByRole("tab", { name: "MCP Tools" }));

    expect(await screen.findByText("tools viewer")).toBeInTheDocument();
  });

  it("shows the read-only settings summary before editing", async () => {
    renderView({ allow_all_keys: true, available_on_public_internet: false });

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("MCP Server Settings")).toBeInTheDocument();
    expect(screen.getByText("Allow All Keys")).toBeInTheDocument();
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    expect(screen.getByText("Internal only")).toBeInTheDocument();
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
  });

  it("swaps in the edit form when Edit Settings is pressed", async () => {
    renderView();

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Settings" }));

    expect(await screen.findByText("edit form")).toBeInTheDocument();
  });

  it("opens straight into the edit form when isEditing is set", async () => {
    renderView({}, { isEditing: true });

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("edit form")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
  });

  it.each([false, true])("keeps config settings read-only with isEditing=%s", async (isEditing) => {
    renderView({ is_config: true }, { isEditing });

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(screen.getByRole("button", { name: "Edit Settings" })).toBeDisabled();
    expect(screen.getByText("Defined in config. Edit your YAML configuration to make changes")).toBeVisible();
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
  });

  it.each([true, false])("honors config read-only state on OAuth return: %s", async (isConfig) => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "srv-1" }));
    renderView({ is_config: isConfig });

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    if (isConfig) {
      expect(screen.queryByText("edit form")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeDisabled();
    } else {
      expect(screen.getByText("edit form")).toBeVisible();
    }
  });

  it("does not open the editor for a view-only admin", async () => {
    renderView({}, { isViewOnly: true, isEditing: true });
    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    expect(screen.getByRole("button", { name: "Edit Settings" })).toBeDisabled();
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
  });

  it("opens on the tab named by initialTabIndex", async () => {
    renderView({}, { initialTabIndex: 1 });

    expect(await screen.findByText("tools viewer")).toBeInTheDocument();
  });

  it("returns to the server list when Back is pressed", async () => {
    const onBack = vi.fn();
    renderView({}, { onBack });

    await userEvent.click(screen.getByRole("button", { name: /Back to All Servers/ }));

    expect(onBack).toHaveBeenCalled();
  });

  it("lists the allowed tools, or says all tools are enabled", async () => {
    renderView({ allowed_tools: ["search", "fetch"] });
    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("search")).toBeInTheDocument();
    expect(screen.getByText("fetch")).toBeInTheDocument();
    expect(screen.queryByText("All tools enabled")).not.toBeInTheDocument();
  });

  it("says all tools are enabled when no allowlist is stored", async () => {
    renderView({ allowed_tools: [] });
    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("All tools enabled")).toBeInTheDocument();
  });

  it("lets a full admin revoke a stored user credential", async () => {
    const row = await openUserCredentials({});
    expect(within(row).getByRole("button", { name: "Revoke credential for user alice" })).toBeInTheDocument();
  });

  it("shows stored credentials to a view-only admin session without a revoke control", async () => {
    const row = await openUserCredentials({ isViewOnly: true });
    expect(row).toHaveTextContent("BYOK API key");
    expect(within(row).queryByRole("button", { name: /^Revoke credential/ })).not.toBeInTheDocument();
  });
  it("keeps URL reveal state shared across Overview and Settings", async () => {
    renderView({ url: "https://example.com/mcp/private-token" });
    const overview = within(screen.getByRole("tabpanel", { name: "Overview" }));
    expect(overview.getByText("https://example.com/mcp/...")).toBeVisible();
    await userEvent.click(overview.getByRole("button", { name: "Show full URL" }));
    expect(overview.getByText("https://example.com/mcp/private-token")).toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    const settings = within(screen.getByRole("tabpanel", { name: "Settings" }));
    expect(settings.getByText("https://example.com/mcp/private-token")).toBeVisible();
    await userEvent.click(settings.getByRole("button", { name: "Hide full URL" }));
    expect(settings.getByText("https://example.com/mcp/...")).toBeVisible();
    await userEvent.click(settings.getByRole("button", { name: "Show full URL" }));
    expect(settings.getByText("https://example.com/mcp/private-token")).toBeVisible();
  });

  it("does not offer URL reveal or user credentials to a non-admin", () => {
    renderView({ url: "https://example.com/mcp/private-token" }, { isProxyAdmin: false, userRole: null });
    expect(screen.queryByRole("button", { name: "Show full URL" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "User Credentials" })).not.toBeInTheDocument();
  });

  it("renders absent connection and settings values without an editor", () => {
    const server = {
      server_name: null,
      alias: null,
      description: null,
      url: null,
      transport: null,
      auth_type: null,
      mcp_access_groups: [],
    };
    renderView(server, { initialTabIndex: 2 });
    const settings = within(screen.getByRole("tabpanel", { name: "Settings" }));
    expect(settings.getAllByText("—")).toHaveLength(6);
    expect(settings.getByText("SSE")).toBeVisible();
    expect(settings.getByText("none")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Show full URL" })).not.toBeInTheDocument();
  });

  it("uses the alias as the header name when no server name is stored", () => {
    renderView({ server_name: null });
    expect(screen.getByRole("heading", { name: "demo_alias" })).toBeVisible();
  });

  it.each([true, false])("copies the displayed name and ID, clipboard success=%s", async (success) => {
    vi.useFakeTimers();
    vi.mocked(copyToClipboard).mockResolvedValue(success);
    renderView({ server_name: null });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy server name" }));
    });
    expect(copyToClipboard).toHaveBeenNthCalledWith(1, "demo_alias");
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy server id" }));
    });
    expect(copyToClipboard).toHaveBeenNthCalledWith(2, "srv-1");
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByRole("heading", { name: "demo_alias" })).toBeVisible();
  });

  it("returns to settings on cancel and to the list on successful save", async () => {
    const onBack = vi.fn();
    renderView({}, { initialTabIndex: 2, isEditing: true, onBack });
    await userEvent.click(screen.getByRole("button", { name: "Cancel edit" }));
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
    expect(onBack).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Edit Settings" }));
    await userEvent.click(screen.getByRole("button", { name: "Save edit" }));
    expect(onBack).toHaveBeenCalledOnce();
  });

  it.each([true, false])("shows OAuth delegation state %s without a pass-through row", (enabled) => {
    renderView(
      { auth_type: "oauth2", delegate_auth_to_upstream: enabled, extra_headers: ["Authorization"] },
      { initialTabIndex: 2 },
    );
    const settings = within(screen.getByRole("tabpanel", { name: "Settings" }));
    expect(settings.getByText("Delegate Auth to Upstream")).toBeVisible();
    expect(settings.queryByText("OAuth Pass-through")).not.toBeInTheDocument();
    if (enabled) expect(settings.getByText("Enabled (PKCE passthrough)")).toBeVisible();
    else expect(settings.getAllByText("Disabled")).toHaveLength(2);
  });

  it.each([true, false])("shows pass-through state %s and named access groups", (enabled) => {
    const server = {
      extra_headers: ["X-Custom", "aUtHoRiZaTiOn"],
      oauth_passthrough: enabled,
      available_on_public_internet: true,
      mcp_access_groups: ["developers", "operators"],
    };
    renderView(server, { initialTabIndex: 2 });
    const settings = within(screen.getByRole("tabpanel", { name: "Settings" }));
    expect(settings.getByText("OAuth Pass-through")).toBeVisible();
    expect(settings.getByText("X-Custom, aUtHoRiZaTiOn")).toBeVisible();
    expect(settings.getByText("Public")).toBeVisible();
    expect(settings.getByText("developers")).toBeVisible();
    expect(settings.getByText("operators")).toBeVisible();
    if (enabled) expect(settings.getByText("Enabled")).toBeVisible();
    else expect(settings.getAllByText("Disabled")).toHaveLength(2);
  });

  it("omits pass-through settings when there is no authorization header", () => {
    renderView({ extra_headers: ["X-Custom"], server_name: "demo_alias" }, { initialTabIndex: 2 });
    expect(screen.queryByText("OAuth Pass-through")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "demo_alias" })).toBeVisible();
  });

  it.each(["invalid json", JSON.stringify({ serverId: "different-server" })])(
    "ignores unrelated OAuth return state %s",
    (stored) => {
      setSecureItem(EDIT_OAUTH_UI_STATE_KEY, stored);
      renderView();
      expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByText("edit form")).not.toBeInTheDocument();
    },
  );
});
