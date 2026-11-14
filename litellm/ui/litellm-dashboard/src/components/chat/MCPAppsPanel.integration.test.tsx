import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPAppsPanel, { type LockedOn, type ServerListReport } from "./MCPAppsPanel";
import {
  deleteMCPOAuthUserCredential,
  fetchMCPServers,
  getMCPOAuthUserCredentialStatus,
  listMCPTools,
} from "../networking";
import type { MCPOAuthUserCredentialStatus } from "../networking";
import type { MCPServer } from "../mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  listMCPTools: vi.fn(),
  deleteMCPOAuthUserCredential: vi.fn(),
}));

const oauthReturn = vi.hoisted(() => ({ serverId: null as string | null, status: "idle", onSuccess: () => {} }));
vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  pendingUserMcpOAuthServerId: () => oauthReturn.serverId,
  useUserMcpOAuthFlow: ({ serverId, onSuccess }: { serverId: string; onSuccess: () => void }) => {
    if (serverId !== oauthReturn.serverId) return { startOAuthFlow: vi.fn(), status: "idle" };
    oauthReturn.onSuccess = onSuccess;
    return { startOAuthFlow: vi.fn(), status: oauthReturn.status };
  },
}));

const servers = [
  {
    server_id: "s-ext",
    server_name: "external_logo",
    auth_type: "none",
    mcp_info: { server_name: "external_logo", logo_url: "https://cdn.example.com/ext.png" },
  },
  {
    server_id: "s-local",
    server_name: "local_logo",
    auth_type: "none",
    mcp_info: { server_name: "local_logo", logo_url: "/ui/assets/logos/github.svg" },
  },
  {
    server_id: "s-none",
    server_name: "no_logo",
    auth_type: "none",
  },
] as MCPServer[];

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>,
  );

describe("MCPAppsPanel logos", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it("resolves backend logo_url values in the server grid", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("external_logo")).toBeInTheDocument();
    expect(screen.getByAltText("external_logo logo")).toHaveAttribute("src", "https://cdn.example.com/ext.png");
    expect(screen.getByAltText("local_logo logo")).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });

  it("renders a colored letter avatar for servers without logo_url", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("no_logo")).toBeInTheDocument();
    expect(screen.queryByAltText("no_logo logo")).not.toBeInTheDocument();
    expect(screen.getByText("N")).toBeInTheDocument();
  });

  it("resolves the logo_url in the detail header", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    fireEvent.click(await screen.findByText("local_logo"));

    expect(await screen.findByRole("heading", { name: "local_logo" })).toBeInTheDocument();
    expect(screen.getByAltText("local_logo logo")).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });
});

const connectServers = [
  {
    server_id: "s-m2m",
    server_name: "service_tool",
    auth_type: "oauth2",
    oauth2_flow: "client_credentials",
    connected_app_reachable: true,
  },
  {
    server_id: "s-reach",
    server_name: "reachable_srv",
    auth_type: "none",
    connected_app_reachable: true,
  },
  {
    server_id: "s-unreach",
    server_name: "unreachable_srv",
    auth_type: "none",
    connected_app_reachable: false,
  },
] as MCPServer[];

const renderConnectPanel = (connectMode: boolean, selectedServers: string[] = []) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={selectedServers} onChange={vi.fn()} connectMode={connectMode} />
    </QueryClientProvider>,
  );

const oauthServer = {
  server_id: "s-oauth",
  server_name: "oauth_srv",
  auth_type: "oauth2",
  oauth2_flow: "authorization_code",
  connected_app_reachable: true,
} as MCPServer;

const SwitchPage = ({
  onConnectedServers,
  initialOff = [],
  lockedOn,
}: {
  onConnectedServers: (servers: ServerListReport) => void;
  initialOff?: string[];
  lockedOn?: LockedOn;
}) => {
  const [off, setOff] = React.useState<string[]>(initialOff);
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel
        accessToken="tok"
        selectedServers={[]}
        onChange={vi.fn()}
        connectMode
        offServerIds={off}
        onOffServerIdsChange={setOff}
        lockedOn={lockedOn}
        onConnectedServers={onConnectedServers}
      />
    </QueryClientProvider>
  );
};

const lastReport = (onConnectedServers: ReturnType<typeof vi.fn>) => onConnectedServers.mock.lastCall?.[0];

describe("MCPAppsPanel switches in connect mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    oauthReturn.serverId = null;
    oauthReturn.status = "idle";
    oauthReturn.onSuccess = () => {};
  });

  it("reports every reachable server as on before any login is checked", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([
      oauthServer,
      ...connectServers,
      { server_id: "s-unsup", server_name: "unsupported_srv", auth_type: "oauth2_token_exchange" },
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockReturnValue(new Promise(() => {}));
    const onConnectedServers = vi.fn();

    render(<SwitchPage onConnectedServers={onConnectedServers} />);

    await waitFor(() =>
      expect(lastReport(onConnectedServers)).toEqual({
        on: ["s-oauth", "s-m2m", "s-reach"],
        all: ["s-oauth", "s-m2m", "s-reach"],
      }),
    );
    expect(screen.getByRole("switch", { name: "Include oauth_srv in this connection" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Include service_tool in this connection" })).toBeChecked();
    expect(
      screen.queryByRole("switch", { name: "Include unsupported_srv in this connection" }),
    ).not.toBeInTheDocument();
  });

  it("leaves a server out once it is turned off, and back in once it is turned on", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    render(<SwitchPage onConnectedServers={onConnectedServers} />);
    const reachable = await screen.findByRole("switch", { name: "Include reachable_srv in this connection" });

    fireEvent.click(reachable);

    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m"], all: ["s-m2m", "s-reach"] });
    expect(reachable).not.toBeChecked();
    expect(screen.queryByRole("heading", { name: "reachable_srv" })).not.toBeInTheDocument();

    fireEvent.click(reachable);

    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m", "s-reach"], all: ["s-m2m", "s-reach"] });
  });

  it("lists only the servers that are on under the Connected tab", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    render(<SwitchPage onConnectedServers={vi.fn()} />);
    fireEvent.click(await screen.findByRole("switch", { name: "Include reachable_srv in this connection" }));

    fireEvent.click(screen.getByRole("tab", { name: "On (1)" }));

    expect(await screen.findByText("service_tool")).toBeInTheDocument();
    expect(screen.queryByText("reachable_srv")).not.toBeInTheDocument();
  });

  it("turns a server off from its detail view too", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    render(<SwitchPage onConnectedServers={onConnectedServers} />);
    fireEvent.click(await screen.findByText("service_tool"));
    await screen.findByRole("heading", { name: "service_tool" });

    fireEvent.click(screen.getByRole("switch", { name: "Include service_tool in this connection" }));

    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-reach"], all: ["s-m2m", "s-reach"] });
    expect(screen.getByText("Off")).toBeInTheDocument();
  });

  it("keeps an OAuth server with no saved login on and offers a sign in", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([oauthServer]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: false,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);

    render(<SwitchPage onConnectedServers={vi.fn()} />);

    expect(await screen.findByText("Sign in to use")).toBeInTheDocument();
    expect(screen.getByText("Connect", { exact: true })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Include oauth_srv in this connection" })).toBeChecked();
  });

  it("keeps the saved login when an OAuth server is turned off", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([oauthServer, ...connectServers]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: true,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);
    const onConnectedServers = vi.fn();
    render(<SwitchPage onConnectedServers={onConnectedServers} />);
    await waitFor(() => expect(getMCPOAuthUserCredentialStatus).toHaveBeenCalled());

    fireEvent.click(await screen.findByRole("switch", { name: "Include oauth_srv in this connection" }));

    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m", "s-reach"], all: ["s-oauth", "s-m2m", "s-reach"] });
    expect(deleteMCPOAuthUserCredential).not.toHaveBeenCalled();
    expect(screen.queryByText("Sign in to use")).not.toBeInTheDocument();
  });

  it("keeps the switch where it was when a login returning from OAuth is saved", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    vi.mocked(fetchMCPServers).mockResolvedValue([oauthServer, ...connectServers]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: false,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);
    const onConnectedServers = vi.fn();
    render(<SwitchPage onConnectedServers={onConnectedServers} initialOff={["s-oauth"]} />);
    await screen.findByText("oauth_srv");

    expect(lastReport(onConnectedServers)).toBe("saving-login");

    await act(async () => {
      oauthReturn.status = "success";
      oauthReturn.onSuccess();
    });

    expect(screen.getByRole("switch", { name: "Include oauth_srv in this connection" })).not.toBeChecked();
    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m", "s-reach"], all: ["s-oauth", "s-m2m", "s-reach"] });
  });

  it("reports a returning login as being saved from the first report, before the hook starts", async () => {
    oauthReturn.serverId = "s-oauth";
    vi.mocked(fetchMCPServers).mockResolvedValue([oauthServer, ...connectServers]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockReturnValue(new Promise(() => {}));
    const onConnectedServers = vi.fn();

    render(<SwitchPage onConnectedServers={onConnectedServers} />);
    await screen.findByText("oauth_srv");

    expect(new Set(onConnectedServers.mock.calls.map(([report]) => report))).toEqual(new Set(["saving-login"]));
  });

  it("shows every switch on and locked when the flow cannot be narrowed", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    render(<SwitchPage onConnectedServers={vi.fn()} initialOff={["s-reach"]} lockedOn="all" />);

    const reachable = await screen.findByRole("switch", { name: "Include reachable_srv in this connection" });
    expect(reachable).toBeChecked();
    expect(reachable).toHaveAttribute("data-disabled");
    expect(screen.queryByText("Required by this client")).not.toBeInTheDocument();

    fireEvent.click(reachable);

    expect(reachable).toBeChecked();
  });

  it("keeps a server the client requires on and locked, and leaves the others switchable", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    render(<SwitchPage onConnectedServers={onConnectedServers} initialOff={["s-reach"]} lockedOn={["s-reach"]} />);
    const required = await screen.findByRole("switch", { name: "Include reachable_srv in this connection" });
    expect(required).toBeChecked();
    expect(required).toHaveAttribute("data-disabled");
    expect(screen.getByText("Required by this client")).toBeInTheDocument();

    fireEvent.click(required);
    fireEvent.click(screen.getByRole("switch", { name: "Include service_tool in this connection" }));

    expect(required).toBeChecked();
    expect(lastReport(onConnectedServers)).toEqual({ on: ["s-reach"], all: ["s-m2m", "s-reach"] });
  });

  it("ignores a stored off switch for a server that is no longer listed", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();

    render(<SwitchPage onConnectedServers={onConnectedServers} initialOff={["s-deleted"]} />);

    await waitFor(() =>
      expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m", "s-reach"], all: ["s-m2m", "s-reach"] }),
    );
  });

  it("reports nothing while the list is still loading", async () => {
    vi.mocked(fetchMCPServers).mockReturnValue(new Promise(() => {}));
    const onConnectedServers = vi.fn();

    render(<SwitchPage onConnectedServers={onConnectedServers} />);

    await waitFor(() => expect(fetchMCPServers).toHaveBeenCalled());
    expect(onConnectedServers.mock.calls.map(([servers]) => servers)).toEqual(["loading"]);
  });

  it("reports nothing and says so when the list fails to load", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValue(new Error("list failed"));
    const onConnectedServers = vi.fn();

    render(<SwitchPage onConnectedServers={onConnectedServers} />);

    expect(
      await screen.findByText("Could not load your MCP servers. Reload the page to try again."),
    ).toBeInTheDocument();
    expect(onConnectedServers.mock.calls.map(([servers]) => servers)).toEqual(["loading", "failed"]);
  });

  it("reports again once a reload succeeds after the list failed to load", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValueOnce(new Error("list failed")).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    const panel = (token: string) => (
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MCPAppsPanel
          accessToken={token}
          selectedServers={[]}
          onChange={vi.fn()}
          connectMode
          onConnectedServers={onConnectedServers}
        />
      </QueryClientProvider>
    );
    const { rerender } = render(panel("tok"));
    expect(await screen.findByText(/could not load your mcp servers/i)).toBeInTheDocument();

    rerender(panel("tok-refreshed"));

    await waitFor(() =>
      expect(lastReport(onConnectedServers)).toEqual({ on: ["s-m2m", "s-reach"], all: ["s-m2m", "s-reach"] }),
    );
  });

  it("leaves a login returning from OAuth to the server's own badge outside connect mode", async () => {
    oauthReturn.serverId = "s-oauth";
    oauthReturn.status = "exchanging";
    vi.mocked(fetchMCPServers).mockResolvedValue([
      { server_id: "s-oauth", server_name: "oauth_srv", auth_type: "oauth2", oauth2_flow: "authorization_code" },
    ] as MCPServer[]);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue({
      server_id: "s-oauth",
      has_credential: false,
      is_expired: false,
    } as MCPOAuthUserCredentialStatus);

    renderConnectPanel(false);

    expect(await screen.findByText("Connecting…")).toBeInTheDocument();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  it("keeps the plain panel's empty text when the list fails to load outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockRejectedValue(new Error("list failed"));

    renderConnectPanel(false);

    expect(await screen.findByText(/no mcp servers configured/i)).toBeInTheDocument();
  });
});

describe("MCPAppsPanel connected-app reachability (LIT-4861)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    oauthReturn.serverId = null;
    oauthReturn.status = "idle";
  });

  it("requests the connected-app view and hides unreachable servers in connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, true);
    expect(screen.queryByText("unreachable_srv")).not.toBeInTheDocument();
    expect(screen.getByText("On (2)")).toBeInTheDocument();
    expect(screen.getByText("service_tool")).toBeInTheDocument();
    expect(screen.queryByText("Connect", { exact: true })).not.toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-reach");
    expect(toolCountFetchedIds).not.toContain("s-unreach");
  });

  it("blocks connecting an unsupported server from the detail view in connect mode", async () => {
    const detailServers = [
      ...connectServers,
      {
        server_id: "s-unsup",
        server_name: "unsupported_srv",
        auth_type: "oauth2_token_exchange",
        connected_app_reachable: true,
      },
    ] as MCPServer[];
    vi.mocked(fetchMCPServers).mockResolvedValue(detailServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true);

    fireEvent.click(await screen.findByText("unsupported_srv"));
    expect(await screen.findByRole("heading", { name: "unsupported_srv" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.getByText("Not supported on this connection")).toBeInTheDocument();
  });

  it("keeps the detail-view Connect action outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(false);

    fireEvent.click(await screen.findByText("unreachable_srv"));
    expect(await screen.findByRole("heading", { name: "unreachable_srv" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Connect" })).toBeInTheDocument();
  });

  it("ignores the flag and skips no server outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(false, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("unreachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, false);
    expect(screen.queryByText("Not available to connected apps")).not.toBeInTheDocument();
    expect(screen.getByText("Connected (2)")).toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-unreach");
  });

  const revocable = (reachable: boolean) =>
    [
      { server_id: "s-reach", server_name: "reachable_srv", auth_type: "none", connected_app_reachable: true },
      { server_id: "s-drop", server_name: "revoked_srv", auth_type: "none", connected_app_reachable: reachable },
    ] as MCPServer[];

  const ConnectPanel = ({
    token,
    client,
    onConnectedServers = vi.fn(),
  }: {
    token: string;
    client: QueryClient;
    onConnectedServers?: (servers: ServerListReport) => void;
  }) => (
    <QueryClientProvider client={client}>
      <MCPAppsPanel
        accessToken={token}
        selectedServers={[]}
        onChange={vi.fn()}
        connectMode
        onConnectedServers={onConnectedServers}
      />
    </QueryClientProvider>
  );

  const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

  it("drops an open detail view when a refetch removes that server from the reachable set", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    rerender(<ConnectPanel token="tok-refreshed" client={client} />);

    await waitFor(() => expect(screen.queryByRole("heading", { name: "revoked_srv" })).not.toBeInTheDocument());
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
    expect(screen.getByText("reachable_srv")).toBeInTheDocument();
  });

  it("stops reporting a server once a refetch removes it from the reachable set", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
    const onConnectedServers = vi.fn();
    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" client={client} onConnectedServers={onConnectedServers} />);
    await waitFor(() =>
      expect(lastReport(onConnectedServers)).toEqual({ on: ["s-reach", "s-drop"], all: ["s-reach", "s-drop"] }),
    );

    rerender(<ConnectPanel token="tok-refreshed" client={client} onConnectedServers={onConnectedServers} />);

    await waitFor(() => expect(lastReport(onConnectedServers)).toEqual({ on: ["s-reach"], all: ["s-reach"] }));
  });

  it("does not let a superseded list load overwrite the current reachable set", async () => {
    let finishStaleLoad: (servers: MCPServer[]) => void = () => {};
    vi.mocked(fetchMCPServers)
      .mockImplementationOnce(() => new Promise((resolve) => (finishStaleLoad = resolve)))
      .mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" client={client} />);
    rerender(<ConnectPanel token="tok-refreshed" client={client} />);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();

    await act(async () => {
      finishStaleLoad(revocable(true));
    });

    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
  });
});
