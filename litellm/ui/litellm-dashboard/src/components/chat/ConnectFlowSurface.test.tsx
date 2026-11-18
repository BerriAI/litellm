import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConnectFlowSurface from "./ConnectFlowSurface";
import type { FinishSelection } from "./ConnectFlowBanner";
import type { LockedOn, ServerListReport } from "./MCPAppsPanel";
import { fetchConnectFlow, type ConnectFlowStatus } from "@/components/networking";
import { pendingUserMcpOAuthServerId } from "@/hooks/useUserMcpOAuthFlow";

const { state, panel, banner } = vi.hoisted(() => ({
  panel: {
    offServerIds: undefined as readonly string[] | undefined,
    lockedOn: undefined as LockedOn,
    onOffServerIdsChange: undefined as ((serverIds: string[]) => void) | undefined,
    onConnectedServers: undefined as ((servers: ServerListReport) => void) | undefined,
  },
  banner: {
    flow: undefined as ConnectFlowStatus | undefined,
    failed: undefined as boolean | undefined,
    finish: undefined as FinishSelection | undefined,
  },
  state: { oauthReturn: null as string | null, connectFlow: null as string | null },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => ({
    get: (key: string) => ({ mcpOauthReturn: state.oauthReturn, connect_flow: state.connectFlow })[key] ?? null,
  }),
}));
vi.mock("@/components/networking", () => ({ fetchConnectFlow: vi.fn() }));
vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({ pendingUserMcpOAuthServerId: vi.fn(() => null) }));
vi.mock("@/components/chat/MCPAppsPanel", () => ({
  default: ({ offServerIds, lockedOn, onOffServerIdsChange, onConnectedServers }: typeof panel) => {
    panel.offServerIds = offServerIds;
    panel.lockedOn = lockedOn;
    panel.onOffServerIdsChange = onOffServerIdsChange;
    panel.onConnectedServers = onConnectedServers;
    return <div data-testid="mcp-apps-panel" />;
  },
}));
vi.mock("@/components/chat/ConnectFlowBanner", () => ({
  default: ({ flow, failed, finish }: typeof banner) => {
    banner.flow = flow;
    banner.failed = failed;
    banner.finish = finish;
    return <div data-testid="connect-flow-banner" />;
  },
}));

const flow = (state: ConnectFlowStatus["state"], connected: boolean | null = null): ConnectFlowStatus => ({
  state,
  client_origin: "https://claude.ai",
  server_id: state === "interactive" || state === "m2m" ? "s-design" : null,
  server_name: state === "interactive" || state === "m2m" ? "design_tool" : null,
  connected,
  servers: null,
});

const everyServer: FinishSelection = { status: "ready", serverIds: [] };
const waiting: FinishSelection = { status: "waiting" };
const finishHanded = () => banner.finish ?? everyServer;

const renderSurface = (onChange = vi.fn()) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ConnectFlowSurface accessToken="token-123" selectedServers={[]} onChange={onChange} />
    </QueryClientProvider>,
  );

const openConnectPage = async (result: ConnectFlowStatus = flow("unscoped"), handle = "flow-handle-123") => {
  state.connectFlow = handle;
  vi.mocked(fetchConnectFlow).mockResolvedValue(result);
  const page = renderSurface();
  await screen.findByTestId("mcp-apps-panel");
  return page;
};

afterEach(() => {
  state.oauthReturn = null;
  state.connectFlow = null;
  banner.flow = undefined;
  banner.failed = undefined;
  banner.finish = undefined;
  sessionStorage.clear();
  vi.clearAllMocks();
});

describe("ConnectFlowSurface", () => {
  it.each([
    { result: flow("unscoped"), grid: true },
    { result: flow("interactive", false), grid: false },
    { result: flow("interactive", true), grid: false },
    { result: flow("m2m", true), grid: false },
    { result: flow("stale"), grid: false },
  ])("hands the banner a $result.state flow and shows the grid only when it is unscoped", async ({ result, grid }) => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockResolvedValue(result);
    renderSurface();

    await waitFor(() => expect(banner.flow).toEqual(result));
    expect(screen.queryByTestId("mcp-apps-panel") !== null).toBe(grid);
  });

  it("keeps the grid hidden until the gateway accepts a handle", () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockReturnValue(new Promise(() => {}));
    renderSurface();

    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(banner.flow).toBeUndefined();
    expect(banner.failed).toBe(false);
  });

  it("keeps the grid hidden and tells the banner when flow validation fails", async () => {
    state.connectFlow = "invalid-handle";
    vi.mocked(fetchConnectFlow).mockRejectedValue(new Error("invalid flow"));
    renderSurface();

    await waitFor(() => expect(banner.failed).toBe(true));
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
  });

  it("renders the ordinary panel without a flow handle", () => {
    renderSurface();

    expect(fetchConnectFlow).not.toHaveBeenCalled();
    expect(screen.getByTestId("mcp-apps-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("connect-flow-banner")).not.toBeInTheDocument();
  });

  it("keeps the servers turned off across the OAuth round trip", async () => {
    const { unmount } = await openConnectPage();
    act(() => panel.onOffServerIdsChange?.(["orders-id"]));
    unmount();

    await openConnectPage();

    expect(panel.offServerIds).toEqual(["orders-id"]);
  });

  it("starts a new connect flow with every server on", async () => {
    const { unmount } = await openConnectPage();
    act(() => panel.onOffServerIdsChange?.(["orders-id"]));
    unmount();

    await openConnectPage(flow("unscoped"), "flow-handle-456");

    expect(panel.offServerIds).toEqual([]);
    expect(panel.lockedOn).toBeUndefined();
  });

  it("locks every switch and never narrows a flow that cannot be narrowed", async () => {
    await openConnectPage({ ...flow("unscoped"), selectable: false });
    expect(panel.lockedOn).toBe("all");

    act(() => panel.onConnectedServers?.({ on: ["catalog-id"], all: ["catalog-id", "orders-id"] }));

    expect(finishHanded()).toEqual(everyServer);
  });

  it("locks only the servers the client requires and still narrows to the servers that are on", async () => {
    await openConnectPage({ ...flow("unscoped"), required_server_ids: ["tracker-id"] });
    expect(panel.lockedOn).toEqual(["tracker-id"]);

    act(() => panel.onConnectedServers?.({ on: ["tracker-id"], all: ["tracker-id", "orders-id"] }));

    expect(banner.finish).toEqual({ status: "ready", serverIds: ["tracker-id"] });
  });

  it("holds Finish on return from a vendor sign-in before the panel has said anything", async () => {
    vi.mocked(pendingUserMcpOAuthServerId).mockReturnValueOnce("vault-id");

    await openConnectPage();

    expect(banner.finish).toEqual(waiting);
  });

  it.each<{ when: string; off: string[]; report: ServerListReport; finish: FinishSelection }>([
    { when: "the list is loading and nothing is off", off: [], report: "loading", finish: everyServer },
    {
      when: "the list is loading after a server was turned off",
      off: ["orders-id"],
      report: "loading",
      finish: waiting,
    },
    { when: "the list failed to load and nothing is off", off: [], report: "failed", finish: everyServer },
    { when: "the list failed after a server was turned off", off: ["orders-id"], report: "failed", finish: waiting },
    { when: "a login returning from OAuth is being saved", off: [], report: "saving-login", finish: waiting },
    {
      when: "every server is on",
      off: [],
      report: { on: ["catalog-id", "orders-id"], all: ["catalog-id", "orders-id"] },
      finish: everyServer,
    },
    {
      when: "one server is off",
      off: ["orders-id"],
      report: { on: ["catalog-id"], all: ["catalog-id", "orders-id"] },
      finish: { status: "ready", serverIds: ["catalog-id"] },
    },
    {
      when: "every server is off",
      off: ["catalog-id", "orders-id"],
      report: { on: [], all: ["catalog-id", "orders-id"] },
      finish: { status: "nothing-on" },
    },
    { when: "no server is reachable", off: [], report: { on: [], all: [] }, finish: everyServer },
  ])("tells the banner Finish is $finish.status when $when", async ({ off, report, finish }) => {
    await openConnectPage();

    act(() => panel.onOffServerIdsChange?.(off));
    act(() => panel.onConnectedServers?.(report));

    expect(banner.finish).toEqual(finish);
  });
});
