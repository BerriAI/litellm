import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import CreateKeyPage from "./page";

interface KeyRow {
  token: string;
}

const { mockReplace, mockUseKeys, mockUiHref, state, uiSettingsState, menuState } = vi.hoisted(() => {
  const state = {
    search: "login=success",
    userRole: "Internal User",
    keys: [] as KeyRow[],
    returnUrl: null as string | null,
  };
  const uiSettingsState = {
    isLoading: false,
    enabledPages: null as string[] | null,
  };
  const menuState = {
    groups: [
      {
        groupLabel: "AI GATEWAY",
        items: [
          { key: "home", page: "home", label: "Home" },
          { key: "api-keys", page: "api-keys", label: "Virtual Keys" },
        ],
      },
    ],
  };
  return {
    state,
    uiSettingsState,
    menuState,
    mockReplace: vi.fn(),
    mockUiHref: vi.fn((segment: string) => `/mocked-ui/${segment}`),
    mockUseKeys: vi.fn(() => ({
      data: { keys: state.keys, total_count: state.keys.length },
      isLoading: false,
    })),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => new URLSearchParams(state.search),
}));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    authLoading: false,
    token: "tok",
    userRole: state.userRole,
    userID: "user-1",
  }),
}));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({
    isLoading: uiSettingsState.isLoading,
    data: { values: { enabled_ui_pages_internal_users: uiSettingsState.enabledPages } },
  }),
}));
vi.mock("@/components/CommandPalette/useVisibleMenuGroups", () => ({
  useVisibleMenuGroups: () => menuState.groups,
}));
vi.mock("@/components/CommandPalette/utils", () => ({
  flattenNavItems: (groups: typeof menuState.groups) =>
    groups.flatMap((group) =>
      group.items.map((item) => ({
        route: item.key === "llm-playground" ? "playground" : item.key,
      })),
    ),
}));
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({ useKeys: mockUseKeys }));
vi.mock("@/app/(dashboard)/home/_components/HomePage", () => ({
  default: () => <div data-testid="home-page" />,
}));
vi.mock("@/components/common_components/LoadingScreen", () => ({
  default: () => <div data-testid="loading-screen" />,
}));
vi.mock("@/components/networking", async () => {
  const { normalizeBackendPageUrlToUiOrigin } =
    await vi.importActual<typeof import("@/utils/returnUrlUtils")>("@/utils/returnUrlUtils");
  return {
    getRuntimeBackendBaseUrl: () => "https://backend.example",
    getUiPageUrl: (url: string) =>
      normalizeBackendPageUrlToUiOrigin(url, "https://backend.example", "http://localhost:3000"),
    proxyBaseUrl: "",
  };
});
vi.mock("@/utils/uiHref", () => ({ uiHref: mockUiHref }));
vi.mock("@/utils/returnUrlUtils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/returnUrlUtils")>()),
  buildLoginUrlWithReturn: (u: string) => u,
  consumeReturnUrl: () => state.returnUrl,
  getLoginUrl: () => "/login",
  storeReturnUrl: () => undefined,
}));

const realLocation = window.location;
const mockLocationReplace = vi.fn();

describe("dashboard landing", () => {
  beforeEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        origin: "http://localhost:3000",
        href: "http://localhost:3000/ui/?login=success",
        replace: mockLocationReplace,
      },
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { configurable: true, value: realLocation });
    state.search = "login=success";
    state.userRole = "Internal User";
    state.keys = [];
    state.returnUrl = null;
    uiSettingsState.isLoading = false;
    uiSettingsState.enabledPages = null;
    menuState.groups = [
      {
        groupLabel: "AI GATEWAY",
        items: [
          { key: "home", page: "home", label: "Home" },
          { key: "api-keys", page: "api-keys", label: "Virtual Keys" },
        ],
      },
    ];
    mockReplace.mockClear();
    mockUseKeys.mockClear();
    mockUiHref.mockClear();
    mockLocationReplace.mockClear();
  });

  it.each(["Internal User", "Internal Viewer", "Admin", "Admin Viewer", "Org Admin", ""])(
    "lands a keyless %s on the keys dashboard, never on the MCP connect page",
    (role) => {
      state.userRole = role;
      render(<CreateKeyPage />);
      expect(screen.getByTestId("home-page")).toBeInTheDocument();
      expect(screen.queryByTestId("loading-screen")).not.toBeInTheDocument();
      expect(mockReplace).not.toHaveBeenCalled();
      expect(mockUiHref).not.toHaveBeenCalledWith("connect");
    },
  );

  it("lands a user who already owns a key on the keys dashboard", () => {
    state.keys = [{ token: "sk-abc" }];
    render(<CreateKeyPage />);
    expect(screen.getByTestId("home-page")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("redirects an internal user to an enabled page when Home is hidden", () => {
    uiSettingsState.enabledPages = ["api-keys"];
    menuState.groups = [
      {
        groupLabel: "AI GATEWAY",
        items: [{ key: "api-keys", page: "api-keys", label: "Virtual Keys" }],
      },
    ];

    render(<CreateKeyPage />);

    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/api-keys");
    expect(screen.getByTestId("loading-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("home-page")).not.toBeInTheDocument();
  });

  it("does not render Home while internal-user page visibility is loading", () => {
    uiSettingsState.isLoading = true;

    render(<CreateKeyPage />);

    expect(screen.getByTestId("loading-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("home-page")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("never looks a user's keys up to decide where the landing goes", () => {
    render(<CreateKeyPage />);
    expect(mockUseKeys).not.toHaveBeenCalled();
  });

  it("redirects an old ?page= bookmark to its path route without rendering the keys dashboard", () => {
    state.search = "page=logs";
    render(<CreateKeyPage />);
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/logs");
    expect(screen.getByTestId("loading-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("home-page")).not.toBeInTheDocument();
  });

  it("carries the MCP env-var deep link's other params through the legacy redirect", () => {
    state.search = "page=mcp-servers&fill_env_vars=srv-1";
    render(<CreateKeyPage />);
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/mcp-servers?fill_env_vars=srv-1");
  });

  it("still sends the user to an explicit stored return URL", () => {
    state.returnUrl = "/ui/models-and-endpoints";
    render(<CreateKeyPage />);
    expect(mockLocationReplace).toHaveBeenCalledWith("http://localhost:3000/ui/models-and-endpoints");
  });

  it.each([
    "/ui/?login=success",
    "http://localhost:3000/ui?login=success",
    "https://backend.example/ui/?login=success",
  ])("does not reload when the return URL %s already matches the current page", (returnUrl) => {
    state.returnUrl = returnUrl;
    render(<CreateKeyPage />);
    expect(mockLocationReplace).not.toHaveBeenCalled();
  });
});
