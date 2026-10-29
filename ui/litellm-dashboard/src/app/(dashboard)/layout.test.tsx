import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, renderWithProviders as render, screen, waitFor, within } from "../../../tests/test-utils";
import { usePathname } from "next/navigation";
import { AuthProvider } from "@/contexts/AuthContext";
import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import Layout from "./layout";

const { replaceMock } = vi.hoisted(() => ({ replaceMock: vi.fn() }));

const CURRENT_USER_PATH = "/management/v1/users/me";
const fetchMock = vi.fn<(input: RequestInfo | URL) => Promise<Response>>();
const pathOf = (input: RequestInfo | URL) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
let currentUserReply: () => Promise<Response>;

const currentUserRequests = () =>
  fetchMock.mock.calls
    .map(([input]) => input)
    .filter((input): input is Request => input instanceof Request && pathOf(input) === CURRENT_USER_PATH);

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
    vi.mocked(usePathname).mockReturnValue("/ui/guardrails");
    currentUserReply = async () =>
      Response.json({ user_id: "normal-user", user_role: "internal_user", user_email: null });
    fetchMock.mockReset();
    fetchMock.mockImplementation((input) =>
      pathOf(input) === CURRENT_USER_PATH ? currentUserReply() : Promise.resolve(Response.json({}, { status: 404 })),
    );
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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
      document.cookie = "litellm_return_url=; Max-Age=0; Path=/";
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

    it("shows the loading screen instead of the shell until the current user resolves", async () => {
      let resolveCurrentUser!: (response: Response) => void;
      currentUserReply = () => new Promise((resolve) => (resolveCurrentUser = resolve));
      document.cookie = `token=${sessionCookie({ user_id: "normal-user", key: "sk-pending-session", user_role: "proxy_admin" })}; Path=/`;

      render(
        <AuthProvider>
          <Layout>
            <div data-testid="page-content" />
          </Layout>
        </AuthProvider>,
      );
      pendingUiConfig.resolve();

      await waitFor(() =>
        expect(currentUserRequests().map((request) => request.headers.get("Authorization"))).toEqual([
          "Bearer sk-pending-session",
        ]),
      );
      expect(screen.getByTestId("loading-screen")).toBeInTheDocument();
      expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument();
      expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();

      resolveCurrentUser(Response.json({ user_id: "normal-user", user_role: "internal_user", user_email: null }));

      expect(await screen.findByTestId("page-content")).toBeInTheDocument();
      expect(screen.getByTestId("sidebar")).toBeInTheDocument();
    });

    it("signs out instead of mounting the shell when the current user cannot be loaded", async () => {
      const originalLocation = window.location;
      const locationReplace = vi.fn();
      Object.defineProperty(window, "location", {
        value: { ...originalLocation, replace: locationReplace },
        writable: true,
      });
      currentUserReply = async () => Response.json({ detail: "rejected" }, { status: 401 });
      document.cookie = `token=${sessionCookie({ user_id: "gone-user", key: "sk-rejected-session" })}; Path=/`;

      try {
        render(
          <AuthProvider>
            <Layout>
              <div data-testid="page-content" />
            </Layout>
          </AuthProvider>,
        );
        pendingUiConfig.resolve();

        await waitFor(() => expect(locationReplace).toHaveBeenCalledWith(expect.stringContaining("/login")), {
          timeout: 4000,
        });
        expect(document.cookie).not.toContain("token=");
        expect(document.cookie).toContain("litellm_return_url=");
        expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument();
        expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
        expect(currentUserRequests()).toHaveLength(2);
      } finally {
        Object.defineProperty(window, "location", { value: originalLocation, writable: true });
      }
    });

    it("keeps the session of an invitation link whose current user cannot be loaded", async () => {
      const originalLocation = window.location;
      const locationReplace = vi.fn();
      Object.defineProperty(window, "location", {
        value: { ...originalLocation, replace: locationReplace },
        writable: true,
      });
      searchParamsValue = new URLSearchParams("invitation_id=abc123");
      currentUserReply = async () => Response.json({ detail: "rejected" }, { status: 401 });
      document.cookie = `token=${sessionCookie({ user_id: "invited-user", key: "sk-invited-session" })}; Path=/`;

      try {
        render(
          <AuthProvider>
            <Layout>
              <div data-testid="page-content" />
            </Layout>
          </AuthProvider>,
        );
        pendingUiConfig.resolve();

        await waitFor(() => expect(currentUserRequests()).toHaveLength(2), { timeout: 4000 });
        await waitFor(() =>
          expect(replaceMock).toHaveBeenCalledWith(expect.stringContaining("/onboarding?invitation_id=abc123")),
        );
        expect(locationReplace).not.toHaveBeenCalled();
        expect(document.cookie).toContain("token=");
      } finally {
        Object.defineProperty(window, "location", { value: originalLocation, writable: true });
      }
    });
  });
});
