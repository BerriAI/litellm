import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, renderWithProviders as render, screen, waitFor, within } from "../../../tests/test-utils";
import { usePathname } from "next/navigation";
import userEvent from "@testing-library/user-event";
import { AuthProvider } from "@/contexts/AuthContext";
import { Dialog, DialogClose, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Sheet, SheetClose, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import Layout from "./layout";

const { replaceMock } = vi.hoisted(() => ({ replaceMock: vi.fn() }));

let searchParamsValue = new URLSearchParams();

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

  it.each([
    {
      name: "dialog",
      role: "dialog",
      slot: "dialog-overlay",
      page: (
        <Dialog>
          <DialogTrigger>Open page overlay</DialogTrigger>
          <DialogContent showCloseButton={false}>
            <DialogTitle>Page overlay</DialogTitle>
            <DialogClose>Close page overlay</DialogClose>
          </DialogContent>
        </Dialog>
      ),
    },
    {
      name: "alert dialog",
      role: "alertdialog",
      slot: "alert-dialog-overlay",
      page: (
        <AlertDialog>
          <AlertDialogTrigger>Open page overlay</AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogTitle>Page overlay</AlertDialogTitle>
            <AlertDialogCancel>Close page overlay</AlertDialogCancel>
          </AlertDialogContent>
        </AlertDialog>
      ),
    },
    {
      name: "sheet",
      role: "dialog",
      slot: "sheet-overlay",
      page: (
        <Sheet>
          <SheetTrigger>Open page overlay</SheetTrigger>
          <SheetContent showCloseButton={false}>
            <SheetTitle>Page overlay</SheetTitle>
            <SheetClose>Close page overlay</SheetClose>
          </SheetContent>
        </Sheet>
      ),
    },
  ])("should display the page $name backdrop while navigation is closed", async ({ role, slot, page }) => {
    render(
      <AuthProvider>
        <Layout>{page}</Layout>
      </AuthProvider>,
    );
    pendingUiConfig.resolve();

    fireEvent.click(await screen.findByRole("button", { name: "Open page overlay" }));
    await screen.findByRole(role, { name: "Page overlay" });
    expect(
      screen.getAllByRole("presentation", { hidden: true }).find((element) => element.dataset.slot === slot),
    ).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Close page overlay" }));
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

  it("should return focus to the navigation trigger after closing the drawer with Escape", async () => {
    const user = userEvent.setup();
    render(
      <AuthProvider>
        <Layout>
          <p>Gateway content</p>
        </Layout>
      </AuthProvider>,
    );
    pendingUiConfig.resolve();

    const trigger = await screen.findByRole("button", { name: "Open navigation" });
    await user.click(trigger);
    await screen.findByRole("dialog", { name: "Navigation" });
    await user.keyboard("{Escape}");
    await waitFor(() => expect(trigger).toHaveFocus());
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
