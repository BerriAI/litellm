import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPAppsPanel from "./MCPAppsPanel";
import { fetchMCPServers, listMCPTools } from "../networking";
import type { MCPServer } from "../mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  listMCPTools: vi.fn(),
  deleteMCPOAuthUserCredential: vi.fn(),
}));

vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle" }),
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

describe("MCPAppsPanel connected-app reachability (LIT-4861)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("requests the connected-app view and hides unreachable servers in connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, true);
    expect(screen.queryByText("unreachable_srv")).not.toBeInTheDocument();
    expect(screen.getByText("Connected (1)")).toBeInTheDocument();
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
    onChange,
    client,
  }: {
    token: string;
    onChange: (servers: string[]) => void;
    client: QueryClient;
  }) => (
    <QueryClientProvider client={client}>
      <MCPAppsPanel accessToken={token} selectedServers={[]} onChange={onChange} connectMode />
    </QueryClientProvider>
  );

  const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

  it("drops an open detail view when a refetch removes that server from the reachable set", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={vi.fn()} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    rerender(<ConnectPanel token="tok-refreshed" onChange={vi.fn()} client={client} />);

    await waitFor(() => expect(screen.queryByRole("heading", { name: "revoked_srv" })).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
    expect(screen.getByText("reachable_srv")).toBeInTheDocument();
  });

  it("does not select a server whose Connect finishes after a refetch removed it", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValueOnce(revocable(true)).mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const onChange = vi.fn();
    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={onChange} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    let finishConnect: (result: { tools: never[] }) => void = () => {};
    vi.mocked(listMCPTools).mockImplementationOnce(() => new Promise((resolve) => (finishConnect = resolve)));
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    rerender(<ConnectPanel token="tok-refreshed" onChange={onChange} client={client} />);
    await waitFor(() => expect(screen.queryByRole("heading", { name: "revoked_srv" })).not.toBeInTheDocument());

    await act(async () => {
      finishConnect({ tools: [] });
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
    expect(screen.getByText("Connected", { exact: false })).toHaveTextContent("Connected");
  });

  it("does not select a server when Connect resolves in the same tick the refetch drops it", async () => {
    let finishRefetch: (servers: MCPServer[]) => void = () => {};
    vi.mocked(fetchMCPServers)
      .mockResolvedValueOnce(revocable(true))
      .mockImplementationOnce(() => new Promise((resolve) => (finishRefetch = resolve)));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const onChange = vi.fn();
    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={onChange} client={client} />);

    fireEvent.click(await screen.findByText("revoked_srv"));
    expect(await screen.findByRole("heading", { name: "revoked_srv" })).toBeInTheDocument();

    let finishConnect: (result: { tools: never[] }) => void = () => {};
    vi.mocked(listMCPTools).mockImplementationOnce(() => new Promise((resolve) => (finishConnect = resolve)));
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    rerender(<ConnectPanel token="tok-refreshed" onChange={onChange} client={client} />);

    await act(async () => {
      finishRefetch(revocable(false));
      finishConnect({ tools: [] });
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
  });

  it("does not let a superseded list load overwrite the current reachable set", async () => {
    let finishStaleLoad: (servers: MCPServer[]) => void = () => {};
    vi.mocked(fetchMCPServers)
      .mockImplementationOnce(() => new Promise((resolve) => (finishStaleLoad = resolve)))
      .mockResolvedValueOnce(revocable(false));
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    const client = newClient();
    const { rerender } = render(<ConnectPanel token="tok" onChange={vi.fn()} client={client} />);
    rerender(<ConnectPanel token="tok-refreshed" onChange={vi.fn()} client={client} />);

    expect(await screen.findByText("reachable_srv")).toBeInTheDocument();

    await act(async () => {
      finishStaleLoad(revocable(true));
    });

    expect(screen.queryByText("revoked_srv")).not.toBeInTheDocument();
  });
});
