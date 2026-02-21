import * as networking from "@/components/networking";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import ModelHubTable from "./ModelHubTable";

const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockGetCookie = vi.hoisted(() => vi.fn());
const mockCheckTokenValidity = vi.hoisted(() => vi.fn());
const mockRouterReplace = vi.hoisted(() => vi.fn());

vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(),
  modelHubPublicModelsCall: vi.fn(),
  modelHubCall: vi.fn(),
  getConfigFieldSetting: vi.fn(),
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  getAgentsList: vi.fn(),
  fetchMCPServers: vi.fn(),
  getUiSettings: vi.fn(),
  getClaudeCodeMarketplace: vi.fn(),
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
}));

vi.mock("@/utils/jwtUtils", () => ({
  checkTokenValidity: mockCheckTokenValidity,
}));

describe("ModelHubTable", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  // Reusable helper function to setup mocks for auth redirect tests
  const setupAuthRedirectTest = (
    requireAuth: boolean,
    tokenValue: string | null,
    isTokenValid: boolean
  ) => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: {
          require_auth_for_public_ai_hub: requireAuth,
        },
      },
      isLoading: false,
    });
    mockGetCookie.mockReturnValue(tokenValue);
    mockCheckTokenValidity.mockReturnValue(isTokenValid);
    mockRouterReplace.mockClear();

    // Setup other required mocks
    vi.mocked(networking.getUiConfig).mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: "http://localhost:4000",
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });
    vi.mocked(networking.modelHubPublicModelsCall).mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {
        require_auth_for_public_ai_hub: requireAuth,
      },
    });
  };

  // Reusable test function for auth redirect scenarios
  const testAuthRedirect = (
    requireAuth: boolean,
    tokenValue: string | null,
    isTokenValid: boolean,
    shouldRedirect: boolean,
    description: string
  ) => {
    it(description, async () => {
      setupAuthRedirectTest(requireAuth, tokenValue, isTokenValid);

      renderWithProviders(
        <ModelHubTable accessToken={null} publicPage={true} premiumUser={false} userRole={null} />
      );

      await waitFor(() => {
        if (shouldRedirect) {
          expect(mockRouterReplace).toHaveBeenCalledWith("http://localhost:4000/ui/login");
        } else {
          expect(mockRouterReplace).not.toHaveBeenCalled();
        }
      });
    });
  };

  it("should render", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({
      data: [],
    });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({
      field_value: false,
    });
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [],
    });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {},
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
      isLoading: false,
    });

    renderWithProviders(<ModelHubTable accessToken="test-token" publicPage={false} premiumUser={false} userRole={null} />);

    await waitFor(() => {
      expect(screen.getByText("AI Hub")).toBeInTheDocument();
    });
  });

  it("should call getUiConfig before modelHubPublicModelsCall when publicPage is true", async () => {
    const getUiConfigMock = vi.mocked(networking.getUiConfig);
    const modelHubPublicModelsCallMock = vi.mocked(networking.modelHubPublicModelsCall);

    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: "http://localhost:4000",
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });
    modelHubPublicModelsCallMock.mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {},
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
      isLoading: false,
    });

    renderWithProviders(<ModelHubTable accessToken={null} publicPage={true} premiumUser={false} userRole={null} />);

    await waitFor(() => {
      expect(getUiConfigMock).toHaveBeenCalled();
      expect(modelHubPublicModelsCallMock).toHaveBeenCalled();
    });

    const getUiConfigCallOrder = getUiConfigMock.mock.invocationCallOrder[0];
    const modelHubPublicModelsCallOrder = modelHubPublicModelsCallMock.mock.invocationCallOrder[0];

    expect(getUiConfigCallOrder).toBeLessThan(modelHubPublicModelsCallOrder);
  });

  describe("authentication redirect behavior", () => {
    // Test cases where requireAuth is true - should redirect on invalid tokens
    testAuthRedirect(
      true,
      null,
      false,
      true,
      "should redirect to login when requireAuth is true and there is no token"
    );

    testAuthRedirect(
      true,
      "expired-token",
      false,
      true,
      "should redirect to login when requireAuth is true and token is expired"
    );

    testAuthRedirect(
      true,
      "malformed-token",
      false,
      true,
      "should redirect to login when requireAuth is true and token is malformed"
    );

    // Test cases where requireAuth is false - should NOT redirect regardless of token state
    testAuthRedirect(
      false,
      null,
      false,
      false,
      "should not redirect when requireAuth is false and there is no token"
    );

    testAuthRedirect(
      false,
      "expired-token",
      false,
      false,
      "should not redirect when requireAuth is false and token is expired"
    );

    testAuthRedirect(
      false,
      "malformed-token",
      false,
      false,
      "should not redirect when requireAuth is false and token is malformed"
    );
  });
});
