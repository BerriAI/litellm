import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
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

vi.mock("@/components/NoRedisWarningBanner", () => ({
  NoRedisWarningBanner: () => null,
}));

vi.mock("@/components/EnvCredentialLoginWarningBanner", () => ({
  EnvCredentialLoginWarningBanner: () => null,
}));

vi.mock("@/components/LicenseExpiryBanner", () => ({
  LicenseExpiryBanner: () => null,
}));

vi.mock("@/components/UserBanner", () => ({
  UserBanner: () => null,
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

    expect(await screen.findByTestId("loading-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("dashboard-header")).not.toBeInTheDocument();

    pendingUiConfig.resolve();

    expect(await screen.findByTestId("page-content")).toBeInTheDocument();
    expect(screen.getByTestId("dashboard-header")).toBeInTheDocument();
    expect(screen.queryByTestId("loading-screen")).not.toBeInTheDocument();
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
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("dashboard-header")).not.toBeInTheDocument();
    expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument();
  });

  describe("forced password reset routing", () => {
    const sessionCookie = (claims: Record<string, unknown>) => {
      const encode = (part: Record<string, unknown>) =>
        btoa(JSON.stringify(part)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
      const exp = Math.floor(Date.now() / 1000) + 3600;
      return `${encode({ alg: "HS256", typ: "JWT" })}.${encode({ ...claims, exp })}.sig`;
    };

    afterEach(() => {
      document.cookie = "token=; Max-Age=0; Path=/";
    });

    it("routes a session flagged password_reset_required to the change-password page", async () => {
      const flaggedClaims = {
        user_id: "flagged-user",
        key: "sk-session",
        login_method: "username_password",
        password_reset_required: true,
      };
      document.cookie = `token=${sessionCookie(flaggedClaims)}; Path=/`;

      render(
        <AuthProvider>
          <Layout>
            <div data-testid="page-content" />
          </Layout>
        </AuthProvider>,
      );

      pendingUiConfig.resolve();

      await waitFor(() => expect(replaceMock).toHaveBeenCalledWith(expect.stringContaining("/change-password")));
    });

    it("does not reroute an unflagged session", async () => {
      document.cookie = `token=${sessionCookie({
        user_id: "normal-user",
        key: "sk-session",
        login_method: "username_password",
      })}; Path=/`;

      render(
        <AuthProvider>
          <Layout>
            <div data-testid="page-content" />
          </Layout>
        </AuthProvider>,
      );

      pendingUiConfig.resolve();

      expect(await screen.findByTestId("page-content")).toBeInTheDocument();
      expect(replaceMock).not.toHaveBeenCalled();
    });
  });
});
