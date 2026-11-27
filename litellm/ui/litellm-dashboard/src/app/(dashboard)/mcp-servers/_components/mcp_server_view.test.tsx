import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { MCPServerView } from "./mcp_server_view";
import * as networking from "@/components/networking";
import { setSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";
import type { MCPServer } from "@/components/mcp_tools/types";
import { mcpServersKeys } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";

vi.mock(".", () => ({
  MCPToolsViewer: () => <div>tools viewer</div>,
}));

vi.mock("./mcp_server_edit", () => ({
  default: ({ mcpServer, onSuccess }: { mcpServer: MCPServer; onSuccess: (server: MCPServer) => void }) => (
    <div>
      edit form
      <button type="button" onClick={() => onSuccess({ ...mcpServer, alias: "renamed" })}>
        save edit
      </button>
    </div>
  ),
  EDIT_OAUTH_UI_STATE_KEY: "litellm-mcp-oauth-edit-state",
}));

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

const newQueryClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
  keepMountWrites?: boolean;
}

const renderView = (
  overrides: Partial<MCPServer> = {},
  props: Record<string, unknown> = {},
  queryClient: QueryClient = newQueryClient(),
  { searchParams = "", onUrlUpdate, keepMountWrites = false }: UrlOptions = {},
) =>
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={!keepMountWrites}
    >
      <QueryClientProvider client={queryClient}>
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
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );

const renderViewAt = (searchParams: string, props: Record<string, unknown> = {}, onUrlUpdate?: OnUrlUpdateFunction) =>
  renderView({}, props, newQueryClient(), { searchParams, onUrlUpdate });

const activePanel = () => screen.getByRole("tabpanel");

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

    expect(within(activePanel()).getByText("tools viewer")).toBeInTheDocument();
  });

  it("shows the read-only settings summary before editing", async () => {
    renderView({
      allow_all_keys: true,
      available_on_public_internet: false,
      mcp_info: { server_name: "demo server", is_public: true, is_public_explicit: true },
    });

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("MCP Server Settings")).toBeInTheDocument();
    expect(screen.getByText("Allow All Keys")).toBeInTheDocument();
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    expect(screen.getByText("Network access")).toBeInTheDocument();
    expect(screen.getByText("All Networks")).toBeInTheDocument();
    expect(screen.queryByText("MCP Hub")).not.toBeInTheDocument();
    expect(screen.queryByText("Listed")).not.toBeInTheDocument();
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
  });

  it("swaps in the edit form when Edit Settings is pressed", async () => {
    renderView();

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Settings" }));

    expect(await screen.findByText("edit form")).toBeInTheDocument();
  });

  it("drops the cached server list and tool catalog once the edit form saves", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } });
    const serversKey = mcpServersKeys.list();
    const toolsKey = ["mcpTools", "srv-1", {}, null];
    const otherToolsKey = ["mcpTools", "srv-2", {}, null];
    queryClient.setQueryData(serversKey, [baseServer]);
    queryClient.setQueryData(toolsKey, { tools: [] });
    queryClient.setQueryData(otherToolsKey, { tools: [] });
    const onBack = vi.fn();
    renderView({}, { onBack }, queryClient);

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Settings" }));
    await userEvent.click(await screen.findByRole("button", { name: "save edit" }));

    expect(queryClient.getQueryState(serversKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(toolsKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(otherToolsKey)?.isInvalidated).toBe(false);
    expect(onBack).toHaveBeenCalledTimes(1);
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

  it.each([
    { transport: "stdio", stdioEnabled: false, shown: true },
    { transport: "stdio", stdioEnabled: true, shown: false },
    { transport: "http", stdioEnabled: false, shown: false },
  ])(
    "explains why a $transport server is inert when stdioEnabled=$stdioEnabled",
    ({ transport, stdioEnabled, shown }) => {
      renderView({ transport }, { stdioEnabled });

      expect(screen.getByText("srv-1")).toBeInTheDocument();
      expect(screen.queryByText("stdio is disabled on this proxy") !== null).toBe(shown);
    },
  );

  it("leaves the stdio warning to the edit form once editing starts", async () => {
    renderView({ transport: "stdio" }, { stdioEnabled: false });
    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));
    expect(screen.getByText("stdio is disabled on this proxy")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Edit Settings" }));

    expect(screen.getByText("edit form")).toBeInTheDocument();
    expect(screen.queryByText("stdio is disabled on this proxy")).not.toBeInTheDocument();
  });

  it("opens on the server tab named in the URL", () => {
    renderViewAt("?server=srv-1&server_tab=tools");

    expect(screen.getByRole("tab", { name: "MCP Tools" })).toHaveAttribute("aria-selected", "true");
    expect(within(activePanel()).getByText("tools viewer")).toBeInTheDocument();
  });

  it("opens the settings named in the URL for an admin", () => {
    renderViewAt("?server_tab=settings");

    expect(within(activePanel()).getByText("MCP Server Settings")).toBeInTheDocument();
  });

  it("writes the server tab the user picks and drops it for the default tab", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderViewAt("?server=srv-1", {}, onUrlUpdate);

    expect(within(activePanel()).getByText("Host URL")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Settings" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server_tab")).toBe("settings");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("srv-1");
    expect(within(activePanel()).getByText("MCP Server Settings")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Overview" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("server_tab")).toBe(false);
    expect(within(activePanel()).getByText("Host URL")).toBeInTheDocument();
  });

  it("drops a settings tab from the URL for a non-admin", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderView({}, { isProxyAdmin: false }, newQueryClient(), {
      searchParams: "?server=srv-1&server_tab=settings",
      onUrlUpdate,
      keepMountWrites: true,
    });

    expect(within(activePanel()).getByText("Host URL")).toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("server_tab")).toBe(false);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("srv-1");
  });

  it("reopens the edit form when returning from an edit OAuth redirect for this server", () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "srv-1" }));

    renderViewAt("?server_tab=settings");

    expect(within(activePanel()).getByText("edit form")).toBeInTheDocument();
  });

  it("stays read only when the edit OAuth redirect was for another server", () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "other" }));

    renderViewAt("?server_tab=settings");

    expect(within(activePanel()).getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.queryByText("edit form")).not.toBeInTheDocument();
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
});
