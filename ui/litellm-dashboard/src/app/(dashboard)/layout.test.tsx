import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { usePathname } from "next/navigation";
import { AuthProvider } from "@/contexts/AuthContext";
import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import Layout from "./layout";

const { replaceMock } = vi.hoisted(() => ({ replaceMock: vi.fn() }));

let searchParamsValue = new URLSearchParams();

vi.mock("@/lib/telemetry/uiEvents", () => ({
  recordUiEvent: vi.fn(() => Promise.resolve()),
}));

vi.mock("@/app/(dashboard)/telemetry/_components/TelemetryEnvBanner", () => ({ default: () => null }));

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: vi.fn(), replace: replaceMock })),
  useSearchParams: vi.fn(() => searchParamsValue),
  usePathname: vi.fn(),
}));

vi.mock("@/components/liteadmin/LiteAdmin", () => ({
  LiteAdminFrame: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("@/components/DashboardHeader", () => ({
  DashboardHeader: ({ navigationTrigger }: { navigationTrigger?: React.ReactNode }) => (
    <div data-testid="dashboard-header">{navigationTrigger}</div>
  ),
}));

vi.mock("@/app/(dashboard)/components/SidebarProvider", () => ({
  default: ({ sidebarCollapsed, onToggleCollapsed }: { sidebarCollapsed: boolean; onToggleCollapsed: () => void }) => (
    <div data-testid="sidebar" data-collapsed={String(sidebarCollapsed)}>
      <button onClick={onToggleCollapsed}>Close navigation</button>
      <a href="#settings">Settings</a>
    </div>
  ),
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

vi.mock("@/components/UpgradeBanner", () => ({
  UpgradeBanner: () => null,
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
    vi.mocked(usePathname).mockReturnValue("/ui/guardrails");
  });

  it("starts mobile navigation closed, opens a modal drawer and closes it after choosing a page", async () => {
    render(
      <AuthProvider>
        <Layout>
          <p>Gateway content</p>
        </Layout>
      </AuthProvider>,
    );
    pendingUiConfig.resolve();

    const trigger = await screen.findByRole("button", { name: "Open navigation" });
    expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument();

    fireEvent.click(trigger);
    const navigation = await screen.findByRole("dialog", { name: "Navigation" });
    expect(within(navigation).getByRole("button", { name: "Close navigation" })).toBeInTheDocument();
    fireEvent.click(within(navigation).getByRole("link", { name: "Settings" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument());
  });

  it("records one page view per route segment, not per query change", async () => {
    const dashboard = () => (
      <AuthProvider>
        <Layout>
          <p>Gateway content</p>
        </Layout>
      </AuthProvider>
    );
    const { rerender } = render(dashboard());
    pendingUiConfig.resolve();
    await screen.findByRole("button", { name: "Open navigation" });
    searchParamsValue = new URLSearchParams("tab=members");
    rerender(dashboard());
    vi.mocked(usePathname).mockReturnValue("/ui/teams/abc-123");
    rerender(dashboard());
    vi.mocked(usePathname).mockReturnValue("/ui/teams/def-456");
    rerender(dashboard());

    await vi.waitFor(() =>
      expect(vi.mocked(recordUiEvent).mock.calls).toEqual([
        [{ page: "guardrails", action: "view" }],
        [{ page: "teams", action: "view" }],
      ]),
    );
  });

  it("closes the mobile drawer when navigation changes outside the drawer", async () => {
    const dashboard = () => (
      <AuthProvider>
        <Layout>
          <p>Gateway content</p>
        </Layout>
      </AuthProvider>
    );
    const { rerender } = render(dashboard());
    pendingUiConfig.resolve();
    fireEvent.click(await screen.findByRole("button", { name: "Open navigation" }));
    expect(await screen.findByRole("dialog", { name: "Navigation" })).toBeInTheDocument();

    vi.mocked(usePathname).mockReturnValue("/ui/api-keys");
    rerender(dashboard());
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument());
    vi.mocked(usePathname).mockReturnValue("/ui/guardrails");
    rerender(dashboard());
    expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument();
  });

  it("collapses the sidebar on Logs for a full-screen view and expands it again after leaving", async () => {
    const dashboard = () => (
      <AuthProvider>
        <Layout>
          <div data-testid="page-content" />
        </Layout>
      </AuthProvider>
    );
    const { rerender } = render(dashboard());
    pendingUiConfig.resolve();
    expect(await screen.findByTestId("sidebar")).toHaveAttribute("data-collapsed", "false");

    vi.mocked(usePathname).mockReturnValue("/ui/logs");
    rerender(dashboard());
    expect(screen.getByTestId("sidebar")).toHaveAttribute("data-collapsed", "true");

    vi.mocked(usePathname).mockReturnValue("/ui/api-keys");
    rerender(dashboard());
    expect(screen.getByTestId("sidebar")).toHaveAttribute("data-collapsed", "false");
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
