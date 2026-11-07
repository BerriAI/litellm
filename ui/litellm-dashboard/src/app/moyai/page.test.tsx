import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders as render } from "@/../tests/test-utils";
import MoyaiPage from "./page";

const { mockReplace, state } = vi.hoisted(() => ({
  mockReplace: vi.fn(),
  state: {
    values: {} as Record<string, unknown>,
    isLoading: false,
  },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "token-123", userRole: "Admin" }),
}));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: state.values }, isLoading: state.isLoading, refetch: vi.fn() }),
}));
vi.mock("@/utils/uiHref", () => ({ uiHref: (segment: string) => `/mocked-ui/${segment}` }));
vi.mock("@/components/navbar", () => ({ default: () => <div data-testid="navbar" /> }));
vi.mock("@/contexts/ThemeContext", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/contexts/PluginModeContext", () => ({
  PluginModeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/components/moyai/MoyaiLanding", () => ({ default: () => <div data-testid="moyai-landing" /> }));

describe("MoyaiPage", () => {
  afterEach(() => {
    state.values = {};
    state.isLoading = false;
    mockReplace.mockClear();
    vi.unstubAllGlobals();
  });

  it("should show the Moyai landing page when the setting is not stored", () => {
    render(<MoyaiPage />);
    expect(screen.getByTestId("moyai-landing")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("should redirect home instead of showing Moyai when an admin turns it off", () => {
    state.values = { enable_moyai_ui: false };
    render(<MoyaiPage />);
    expect(screen.queryByTestId("moyai-landing")).not.toBeInTheDocument();
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/");
  });

  it("should not open a connected Moyai deployment when an admin turns Moyai off", () => {
    const replace = vi.fn();
    vi.stubGlobal("location", { ...window.location, search: "", replace });
    state.values = { enable_moyai_ui: false, moyai_url: "https://moyai.example.com" };
    render(<MoyaiPage />);
    expect(replace).not.toHaveBeenCalled();
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/");
  });

  it("should not redirect while UI settings are loading", () => {
    state.isLoading = true;
    render(<MoyaiPage />);
    expect(screen.queryByTestId("moyai-landing")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });
});
