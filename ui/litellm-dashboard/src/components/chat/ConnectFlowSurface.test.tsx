import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConnectFlowSurface from "./ConnectFlowSurface";
import { fetchConnectFlow } from "@/components/networking";

interface PanelProps {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
  connectMode?: boolean;
  scopedServerId?: string | null;
  autoStartKey?: string | null;
  onScopedConnected?: () => void;
}

interface BannerProps {
  flowHandle: string;
  clientOrigin: string | null;
  serverLabel?: string | null;
  canFinish?: boolean;
  canCancel?: boolean;
}

const { mockReplace, mockPanel, mockBanner, state } = vi.hoisted(() => {
  const state = { oauthReturn: null as string | null, connectFlow: null as string | null };
  return {
    state,
    mockReplace: vi.fn(),
    mockPanel: vi.fn((_props: PanelProps) => <div data-testid="mcp-apps-panel" />),
    mockBanner: vi.fn((_props: BannerProps) => <div data-testid="connect-flow-banner" />),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => ({
    get: (key: string) => {
      if (key === "mcpOauthReturn") return state.oauthReturn;
      if (key === "connect_flow") return state.connectFlow;
      return null;
    },
  }),
}));
vi.mock("@/components/networking", () => ({ fetchConnectFlow: vi.fn() }));
vi.mock("@/components/chat/MCPAppsPanel", () => ({ default: mockPanel }));
vi.mock("@/components/chat/ConnectFlowBanner", () => ({ default: mockBanner }));

const scopedFlow = { client_origin: "https://claude.ai", server_id: "s-design", server_name: "design_tool" };
const unscopedFlow = {
  state: "unscoped" as const,
  client_origin: "https://claude.ai",
  server_id: null,
  server_name: null,
  connected: null,
};
const scopedUnconnected = { ...scopedFlow, state: "interactive" as const, connected: false };
const scopedConnected = { ...scopedFlow, state: "interactive" as const, connected: true };
const m2mFlow = { ...scopedFlow, state: "m2m" as const, connected: true };
const staleFlow = {
  state: "stale" as const,
  client_origin: "https://claude.ai",
  server_id: null,
  server_name: null,
  connected: null,
};

const renderSurface = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ConnectFlowSurface accessToken="token-123" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>,
  );

const lastPanelProps = (): PanelProps => mockPanel.mock.calls[mockPanel.mock.calls.length - 1][0];
const lastBannerProps = (): BannerProps => mockBanner.mock.calls[mockBanner.mock.calls.length - 1][0];

describe("ConnectFlowSurface", () => {
  afterEach(() => {
    state.oauthReturn = null;
    state.connectFlow = null;
    vi.clearAllMocks();
  });

  it("asks the gateway about the flow handle and mounts the banner from its answer, never from the URL", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(unscopedFlow);
    renderSurface();

    expect(await screen.findByTestId("mcp-apps-panel")).toBeInTheDocument();
    expect(fetchConnectFlow).toHaveBeenCalledWith("flow-handle-123");
    const banner = lastBannerProps();
    expect(banner.flowHandle).toBe("flow-handle-123");
    expect(banner.clientOrigin).toBe("https://claude.ai");
    expect(banner.serverLabel).toBeNull();
    expect(banner.canFinish).toBe(true);
    expect(lastPanelProps()).toMatchObject({ connectMode: true, scopedServerId: null, autoStartKey: null });
  });

  it("narrows to the gateway's scoped server, arms one automatic vendor trip, and withholds Finish", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(scopedUnconnected);
    renderSurface();

    await screen.findByTestId("mcp-apps-panel");
    expect(lastPanelProps()).toMatchObject({
      scopedServerId: "s-design",
      autoStartKey: "litellm-mcp-autostart:flow-handle-123",
    });
    expect(lastBannerProps()).toMatchObject({ serverLabel: "design_tool", canFinish: false, canCancel: true });
  });

  it("offers Finish and arms nothing once the gateway reports the server authorized", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(scopedConnected);
    renderSurface();

    await screen.findByTestId("mcp-apps-panel");
    expect(lastPanelProps()).toMatchObject({ scopedServerId: "s-design", autoStartKey: null });
    expect(lastBannerProps()).toMatchObject({ serverLabel: "design_tool", canFinish: true, canCancel: true });
  });

  it("shows M2M as ready without arming an interactive OAuth trip", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(m2mFlow);
    renderSurface();

    await screen.findByTestId("mcp-apps-panel");
    expect(lastPanelProps()).toMatchObject({ scopedServerId: "s-design", autoStartKey: null });
    expect(lastBannerProps()).toMatchObject({ serverLabel: "design_tool", canFinish: true });
  });

  it("keeps a stale scoped flow out of the grid and offers only Cancel", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(staleFlow);
    renderSurface();

    await screen.findByTestId("connect-flow-banner");
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(lastBannerProps()).toMatchObject({
      serverLabel: "the requested MCP server",
      canFinish: false,
      canCancel: true,
    });
  });

  it("re-asks the gateway when the panel reports the vendor step done, so Finish appears", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValueOnce(scopedUnconnected).mockResolvedValueOnce(scopedConnected);
    renderSurface();

    await screen.findByTestId("mcp-apps-panel");
    expect(lastBannerProps().canFinish).toBe(false);
    const panel = lastPanelProps();
    await act(async () => {
      panel.onScopedConnected?.();
    });

    await waitFor(() => expect(lastBannerProps().canFinish).toBe(true));
  });

  it("shows a Cancel-only flow while the gateway status request fails", async () => {
    state.connectFlow = "made-up";
    vi.mocked(fetchConnectFlow).mockRejectedValue(new Error("unknown or expired connect flow"));
    renderSurface();

    await screen.findByTestId("connect-flow-banner");
    expect(lastBannerProps()).toMatchObject({ canFinish: false, canCancel: true, serverUnavailable: true });
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
  });

  it("does not mount the grid while the gateway status request is pending", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockReturnValue(new Promise(() => {}));
    renderSurface();

    expect(screen.getByTestId("connect-flow-banner")).toBeInTheDocument();
    expect(lastBannerProps()).toMatchObject({ canFinish: false, canCancel: true, serverUnavailable: true });
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
  });

  it("does not ask the gateway or show a banner for a visit without a flow handle", () => {
    renderSurface();

    expect(fetchConnectFlow).not.toHaveBeenCalled();
    expect(screen.queryByTestId("connect-flow-banner")).not.toBeInTheDocument();
    expect(lastPanelProps().connectMode).toBe(false);
  });

  it("strips the mcpOauthReturn param while keeping the connect flow handle", () => {
    state.oauthReturn = "apps";
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(scopedConnected);
    window.history.replaceState({}, "", "/connect?connect_flow=flow-handle-123&mcpOauthReturn=apps");
    renderSurface();

    expect(mockReplace).toHaveBeenCalledWith("/connect?connect_flow=flow-handle-123");
  });

  it("does not rewrite the URL when there is no OAuth return param", () => {
    renderSurface();

    expect(mockReplace).not.toHaveBeenCalled();
  });
});
