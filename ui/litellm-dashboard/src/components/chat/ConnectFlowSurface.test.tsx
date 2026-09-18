import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConnectFlowSurface from "./ConnectFlowSurface";
import { fetchConnectFlow } from "@/components/networking";

const { startOAuthFlow, state, onSuccess } = vi.hoisted(() => ({
  startOAuthFlow: vi.fn(),
  onSuccess: { current: undefined as (() => void) | undefined },
  state: { oauthReturn: null as string | null, connectFlow: null as string | null },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => ({
    get: (key: string) => ({ mcpOauthReturn: state.oauthReturn, connect_flow: state.connectFlow })[key] ?? null,
  }),
}));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  fetchConnectFlow: vi.fn(),
  getProxyBaseUrl: () => "https://gateway.example.com",
}));
vi.mock("@/components/chat/MCPAppsPanel", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/chat/MCPAppsPanel")>()),
  default: () => <div data-testid="mcp-apps-panel" />,
}));
vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: ({ onSuccess: success }: { onSuccess: () => void }) => {
    onSuccess.current = success;
    return { startOAuthFlow, status: "idle" };
  },
}));

const flow = (state: "unscoped" | "interactive" | "m2m" | "stale", connected: boolean | null = null) => ({
  state,
  client_origin: "https://claude.ai",
  server_id: state === "interactive" || state === "m2m" ? "s-design" : null,
  server_name: state === "interactive" || state === "m2m" ? "design_tool" : null,
  connected,
});

const renderSurface = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <ConnectFlowSurface accessToken="token-123" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>,
  );

afterEach(() => {
  state.oauthReturn = null;
  state.connectFlow = null;
  onSuccess.current = undefined;
  sessionStorage.clear();
  vi.clearAllMocks();
});

describe("ConnectFlowSurface", () => {
  it.each([
    { result: flow("unscoped"), grid: true, finish: true, cancel: false, oauthStarts: 0 },
    { result: flow("interactive", false), grid: false, finish: false, cancel: true, oauthStarts: 1 },
    { result: flow("interactive", true), grid: false, finish: true, cancel: true, oauthStarts: 0 },
    { result: flow("m2m", true), grid: false, finish: true, cancel: true, oauthStarts: 0 },
    { result: flow("stale"), grid: false, finish: false, cancel: true, oauthStarts: 0 },
  ])(
    "renders $result.state without widening its action surface",
    async ({ result, grid, finish, cancel, oauthStarts }) => {
      state.connectFlow = "flow-handle-123";
      vi.mocked(fetchConnectFlow).mockResolvedValue(result);
      renderSurface();

      await screen.findByRole("button", { name: /finish connecting|cancel|connect/i });
      await waitFor(() => expect(startOAuthFlow).toHaveBeenCalledTimes(oauthStarts));
      expect(screen.queryByTestId("mcp-apps-panel") !== null).toBe(grid);
      expect(screen.queryByRole("button", { name: /finish connecting/i }) !== null).toBe(finish);
      expect(screen.queryByRole("button", { name: "Cancel" }) !== null).toBe(cancel);
    },
  );

  it("keeps the grid and Finish hidden until the gateway accepts a handle", () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow).mockReturnValue(new Promise(() => {}));
    renderSurface();

    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveAttribute("value", "deny");
  });

  it("keeps the grid and Finish hidden when flow validation fails", async () => {
    state.connectFlow = "invalid-handle";
    vi.mocked(fetchConnectFlow).mockRejectedValue(new Error("invalid flow"));
    renderSurface();

    await screen.findByRole("button", { name: "Cancel" });
    expect(screen.queryByTestId("mcp-apps-panel")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
  });

  it("refetches the sealed flow after the vendor connection completes", async () => {
    state.connectFlow = "flow-handle-123";
    vi.mocked(fetchConnectFlow)
      .mockResolvedValueOnce(flow("interactive", false))
      .mockResolvedValueOnce(flow("interactive", true));
    renderSurface();

    await waitFor(() => expect(startOAuthFlow).toHaveBeenCalledOnce());
    await act(async () => onSuccess.current?.());

    await screen.findByRole("button", { name: /finish connecting/i });
  });

  it("renders the ordinary panel without a flow handle", () => {
    renderSurface();
    expect(fetchConnectFlow).not.toHaveBeenCalled();
    expect(screen.getByTestId("mcp-apps-panel")).toBeInTheDocument();
  });
});
