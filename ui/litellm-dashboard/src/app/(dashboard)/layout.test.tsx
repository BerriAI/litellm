import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AuthProvider } from "@/contexts/AuthContext";
import Layout from "./layout";

const { replaceMock } = vi.hoisted(() => ({ replaceMock: vi.fn() }));

let searchParamsValue = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: vi.fn(), replace: replaceMock })),
  useSearchParams: vi.fn(() => searchParamsValue),
  usePathname: vi.fn(() => "/ui/guardrails"),
}));

vi.mock("@/components/DashboardHeader", () => ({
  DashboardHeader: () => <div data-testid="dashboard-header" />,
}));

vi.mock("@/app/(dashboard)/components/SidebarProvider", () => ({
  default: () => <div data-testid="sidebar" />,
}));

vi.mock("@/components/DebugWarningBanner", () => ({
  DebugWarningBanner: () => null,
}));

vi.mock("@/components/LicenseExpiryBanner", () => ({
  LicenseExpiryBanner: () => null,
}));

vi.mock("@/contexts/ThemeContext", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/common_components/LoadingScreen", () => ({
  default: () => <div data-testid="loading-screen" />,
}));

type Deferred = { promise: Promise<void>; resolve: () => void };

const createDeferred = (): Deferred => {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

let pendingUiConfig: Deferred;

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getUiConfig: vi.fn(() => pendingUiConfig.promise),
    setGlobalLitellmHeaderName: vi.fn(),
  };
});

describe("(dashboard) Layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pendingUiConfig = createDeferred();
    searchParamsValue = new URLSearchParams();
  });

  it("does not mount route content until getUiConfig has resolved", async () => {
    render(
      <AuthProvider>
        <Layout>
          <div data-testid="page-content" />
        </Layout>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("loading-screen")).toBeTruthy());
    expect(screen.queryByTestId("page-content")).toBeNull();
    expect(screen.queryByTestId("dashboard-header")).toBeNull();

    pendingUiConfig.resolve();

    await waitFor(() => expect(screen.getByTestId("page-content")).toBeTruthy());
    expect(screen.getByTestId("dashboard-header")).toBeTruthy();
    expect(screen.queryByTestId("loading-screen")).toBeNull();
  });

  it("redirects an invitation link to the onboarding route instead of rendering the dashboard shell", async () => {
    searchParamsValue = new URLSearchParams("invitation_id=abc123");

    render(
      <AuthProvider>
        <Layout>
          <div data-testid="page-content" />
        </Layout>
      </AuthProvider>,
    );

    pendingUiConfig.resolve();

    await waitFor(() =>
      expect(replaceMock).toHaveBeenCalledWith(expect.stringContaining("/onboarding?invitation_id=abc123")),
    );
    expect(screen.queryByTestId("page-content")).toBeNull();
    expect(screen.queryByTestId("dashboard-header")).toBeNull();
    expect(screen.queryByTestId("sidebar")).toBeNull();
  });
});
