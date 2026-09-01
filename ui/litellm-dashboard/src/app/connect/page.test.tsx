import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectPage from "./page";

interface PanelProps {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
  connectMode?: boolean;
}

interface BannerProps {
  flowHandle: string;
  clientOrigin: string | null;
}

const { mockReplace, mockPanel, mockBanner, state } = vi.hoisted(() => {
  const state = {
    oauthReturn: null as string | null,
    connectFlow: null as string | null,
    connectClient: null as string | null,
  };
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
      if (key === "connect_client") return state.connectClient;
      return null;
    },
  }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "token-123" }),
}));
vi.mock("@/components/chat/MCPAppsPanel", () => ({ default: mockPanel }));
vi.mock("@/components/chat/ConnectFlowBanner", () => ({ default: mockBanner }));

describe("ConnectPage", () => {
  afterEach(() => {
    state.oauthReturn = null;
    state.connectFlow = null;
    state.connectClient = null;
    mockReplace.mockClear();
    mockPanel.mockClear();
    mockBanner.mockClear();
  });

  it("renders the MCP connect panel with the user's access token", () => {
    render(<ConnectPage />);
    expect(screen.getByTestId("mcp-apps-panel")).toBeInTheDocument();
    expect(mockPanel.mock.calls[0][0]).toMatchObject({ accessToken: "token-123", selectedServers: [] });
  });

  it("strips the mcpOauthReturn param from the URL after an OAuth return", () => {
    state.oauthReturn = "apps";
    window.history.replaceState({}, "", "/connect?mcpOauthReturn=apps");
    render(<ConnectPage />);
    expect(mockReplace).toHaveBeenCalledWith("/connect");
  });

  it("does not rewrite the URL when there is no OAuth return param", () => {
    render(<ConnectPage />);
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("mounts the gateway connect banner and puts the panel in connect mode for a DCR flow", () => {
    state.connectFlow = "flow-handle-123";
    state.connectClient = "https://claude.ai";
    render(<ConnectPage />);
    expect(screen.getByTestId("connect-flow-banner")).toBeInTheDocument();
    expect(mockBanner.mock.calls[0][0]).toMatchObject({
      flowHandle: "flow-handle-123",
      clientOrigin: "https://claude.ai",
    });
    expect(mockPanel.mock.calls[0][0].connectMode).toBe(true);
  });

  it("shows no connect banner and leaves connect mode off for a plain visit", () => {
    render(<ConnectPage />);
    expect(screen.queryByTestId("connect-flow-banner")).not.toBeInTheDocument();
    expect(mockBanner).not.toHaveBeenCalled();
    expect(mockPanel.mock.calls[0][0].connectMode).toBe(false);
  });

  it("keeps the connect flow handle in the URL while stripping the OAuth return param", () => {
    state.oauthReturn = "apps";
    state.connectFlow = "flow-handle-123";
    window.history.replaceState({}, "", "/connect?connect_flow=flow-handle-123&mcpOauthReturn=apps");
    render(<ConnectPage />);
    expect(mockReplace).toHaveBeenCalledWith("/connect?connect_flow=flow-handle-123");
  });
});
