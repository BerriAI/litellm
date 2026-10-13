import * as networking from "@/components/networking";
import userEvent from "@testing-library/user-event";
import { act } from "@testing-library/react";
import type { MCPServerData } from "./MCPHubTableColumns";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import ModelHubTable from "./ModelHubTable";

// useUserDisplayNames must not reach useAuthorized from the public page, so
// this file exercises the real hook instead of the global mock in setupTests.
vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockGetCookie = vi.hoisted(() => vi.fn());
const mockCheckTokenValidity = vi.hoisted(() => vi.fn());
const mockRouterReplace = vi.hoisted(() => vi.fn());
const mockLocationReplace = vi.hoisted(() => vi.fn());

vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(),
  modelHubPublicModelsCall: vi.fn(),
  modelHubCall: vi.fn(),
  getConfigFieldSetting: vi.fn(),
  getGlobalLitellmHeaderName: vi.fn(() => "X-Litellm-Key"),
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  getAgentsList: vi.fn(),
  fetchMCPServers: vi.fn(),
  userListCall: vi.fn(),
  makeMCPPublicCall: vi.fn(),
  getUiSettings: vi.fn(),
  getClaudeCodePluginsList: vi.fn(() => Promise.resolve({ plugins: [] })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockRouterReplace,
  }),
}));

vi.mock("@/components/public_model_hub", () => ({
  default: () => <div>Public Model Hub</div>,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

vi.mock("@/utils/cookieUtils", () => ({
  getCookie: mockGetCookie,
  clearTokenCookies: vi.fn(),
}));

vi.mock("@/utils/jwtUtils", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  checkTokenValidity: mockCheckTokenValidity,
}));

const renderPublicHub = (accessToken: string | null) =>
  renderWithProviders(
    <ModelHubTable
      accessToken={accessToken}
      publicPage={true}
      premiumUser={false}
      userRole={null}
      canOpenPlayground={false}
    />,
  );

describe("ModelHubTable public access", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    Object.defineProperty(window, "location", {
      value: {
        href: "http://localhost:4000/ui/model_hub_table",
        origin: "http://localhost:4000",
        hostname: "localhost",
        pathname: "/ui/model_hub_table",
        search: "",
        protocol: "http:",
        replace: mockLocationReplace,
      },
      writable: true,
    });

    mockUseUISettings.mockReturnValue({
      data: { values: { require_auth_for_public_ai_hub: false } },
      isLoading: false,
    });
    mockGetCookie.mockReturnValue(null);
    vi.mocked(networking.getUiConfig).mockResolvedValue({} as networking.LiteLLMWellKnownUiConfig);
    vi.mocked(networking.modelHubPublicModelsCall).mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: { require_auth_for_public_ai_hub: false },
    });
    vi.mocked(networking.userListCall).mockResolvedValue({ users: [] } as networking.UserListResponse);
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      value: originalLocation,
      writable: true,
    });
    vi.clearAllMocks();
  });

  it("keeps an anonymous visitor on the public AI Hub without redirecting to login", async () => {
    renderPublicHub(null);

    expect(await screen.findByText("Public Model Hub")).toBeInTheDocument();
    await waitFor(() => expect(networking.getUiConfig).toHaveBeenCalled());
    await act(async () => {});

    expect(mockLocationReplace).not.toHaveBeenCalled();
  });

  it("never lists users for the public AI Hub even when ?key= provides a token", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({ data: [] });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
    vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [] });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      { server_id: "s1", server_name: "hub-server", created_by: "u-ada" } as MCPServerData,
    ]);

    renderPublicHub("sk-from-key-param");

    await waitFor(() => expect(networking.modelHubCall).toHaveBeenCalledWith("sk-from-key-param"));
    await act(async () => {});

    expect(mockLocationReplace).not.toHaveBeenCalled();
    expect(networking.userListCall).not.toHaveBeenCalled();
  });

  it("shows creator ids without user links for a viewer who cannot list users", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({ data: [] });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
    vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [] });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      { server_id: "s1", server_name: "hub-server", created_by: "u-ada", updated_by: "u-bob" } as MCPServerData,
    ]);

    const user = userEvent.setup();
    renderWithProviders(
      <ModelHubTable
        accessToken="test-token"
        publicPage={false}
        premiumUser={false}
        userRole="Internal User"
        canOpenPlayground
      />,
    );

    await user.click(await screen.findByRole("tab", { name: "MCP Hub" }));

    expect(await screen.findByText("u-ada")).toBeInTheDocument();
    expect(document.querySelector('a[href*="users?user="]')).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "u-ada" })).not.toBeInTheDocument();
  });

  it("resolves creator display names on the admin MCP Hub tab", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({
      data: [{ model_group: "claude-opus-4-8", providers: ["anthropic"], mode: "chat" }],
    });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
    vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [] });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      { server_id: "s1", server_name: "hub-server", created_by: "u-ada" } as MCPServerData,
    ]);
    vi.mocked(networking.userListCall).mockResolvedValue({
      users: [{ user_id: "u-ada", user_alias: "Ada" }],
      total: 1,
      page: 1,
      page_size: 100,
      total_pages: 1,
    } as networking.UserListResponse);

    const user = userEvent.setup();
    renderWithProviders(
      <ModelHubTable
        accessToken="test-token"
        publicPage={false}
        premiumUser={false}
        userRole="Admin"
        canOpenPlayground
      />,
    );

    await user.click(await screen.findByRole("tab", { name: "MCP Hub" }));
    const link = await screen.findByRole("link", { name: "Ada" });
    expect(link).toHaveAttribute("href", expect.stringContaining("users?user=u-ada"));
  });
});
