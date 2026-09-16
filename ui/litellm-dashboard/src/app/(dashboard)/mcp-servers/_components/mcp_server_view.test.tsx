import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { MCPServerView } from "./mcp_server_view";
import type { MCPServer } from "@/components/mcp_tools/types";
import { setSecureItem } from "@/utils/secureStorage";
import { render, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

vi.mock(".", () => ({
  MCPToolsViewer: () => <div>tools viewer</div>,
}));

vi.mock("./mcp_server_edit", () => ({
  default: () => <div>edit form</div>,
  EDIT_OAUTH_UI_STATE_KEY: "litellm-mcp-oauth-edit-state",
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

const viewElement = (overrides: Partial<MCPServer>, props: Record<string, unknown>) => (
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
);

const renderView = (
  overrides: Partial<MCPServer> = {},
  props: Record<string, unknown> = {},
  searchParams = "",
  onUrlUpdate?: OnUrlUpdateFunction,
) => renderWithProviders(viewElement(overrides, props), { searchParams, onUrlUpdate });

const renderViewKeepingMountWrites = (searchParams: string, props: Record<string, unknown>) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      {viewElement({}, props)}
    </NuqsTestingAdapter>,
  );
  return onUrlUpdate;
};

const activePanel = () => screen.getByRole("tabpanel");

describe("MCPServerView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
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

  it("opens on the server tab named in the URL", () => {
    renderView({}, {}, "?server=srv-1&server_tab=tools");

    expect(screen.getByRole("tab", { name: "MCP Tools" })).toHaveAttribute("aria-selected", "true");
    expect(within(activePanel()).getByText("tools viewer")).toBeInTheDocument();
  });

  it("opens the settings named in the URL for an admin", () => {
    renderView({}, {}, "?server_tab=settings");

    expect(within(activePanel()).getByText("MCP Server Settings")).toBeInTheDocument();
  });

  it("writes the server tab the user picks and drops it for the default tab", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderView({}, {}, "?server=srv-1", onUrlUpdate);

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
    const onUrlUpdate = renderViewKeepingMountWrites("?server=srv-1&server_tab=settings", { isProxyAdmin: false });

    expect(within(activePanel()).getByText("Host URL")).toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("server_tab")).toBe(false);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("srv-1");
  });

  it("reopens the edit form when returning from an edit OAuth redirect for this server", () => {
    setSecureItem("litellm-mcp-oauth-edit-state", JSON.stringify({ serverId: "srv-1" }));

    renderView({}, {}, "?server_tab=settings");

    expect(within(activePanel()).getByText("edit form")).toBeInTheDocument();
  });

  it("stays read only when the edit OAuth redirect was for another server", () => {
    setSecureItem("litellm-mcp-oauth-edit-state", JSON.stringify({ serverId: "other" }));

    renderView({}, {}, "?server_tab=settings");

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
});
