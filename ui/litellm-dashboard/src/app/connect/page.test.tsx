import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectPage from "./page";

interface PanelProps {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const { mockReplace, mockPanel, state } = vi.hoisted(() => {
  const state = {
    oauthReturn: null as string | null,
  };
  return {
    state,
    mockReplace: vi.fn(),
    mockPanel: vi.fn((_props: PanelProps) => <div data-testid="mcp-apps-panel" />),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => ({ get: (key: string) => (key === "mcpOauthReturn" ? state.oauthReturn : null) }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "token-123" }),
}));
vi.mock("@/components/chat/MCPAppsPanel", () => ({ default: mockPanel }));

describe("ConnectPage", () => {
  afterEach(() => {
    state.oauthReturn = null;
    mockReplace.mockClear();
    mockPanel.mockClear();
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
});
