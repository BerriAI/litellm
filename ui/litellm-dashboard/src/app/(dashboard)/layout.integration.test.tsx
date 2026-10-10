import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
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

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ui/guardrails",
}));

beforeEach(() => {
  testQueryClient.clear();
  localStorage.clear();
  const claims = {
    key: "sk-layout-test",
    user_id: "layout-admin",
    user_role: "proxy_admin",
    login_method: "sso",
    exp: Date.now() / 1000 + 3600,
  };
  const token = `${btoa(JSON.stringify({ alg: "none" }))}.${btoa(JSON.stringify(claims))}.test`;
  document.cookie = `token=${token}; Path=/`;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = input instanceof Request ? input.url : String(input);
      const path = new URL(url, "http://localhost").pathname;
      switch (path) {
        case "/litellm/.well-known/litellm-ui-config":
          return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
        case "/get/ui_settings":
        case "/get/ui_theme_settings":
          return Response.json({ values: {} });
        case "/api/plugins":
        case "/team/list":
        case "/organization/list":
          return Response.json([]);
        case "/health/readiness/details":
          return Response.json({ status: "healthy" });
        case "/health/license": {
          const license = {
            has_license: false,
            license_type: null,
            expiration_date: null,
            allowed_features: [],
            limits: { max_users: null, max_teams: null },
          };
          return Response.json(license);
        }
        case "/user/available_users": {
          const usage = {
            total_users: null,
            total_users_used: 1,
            total_users_remaining: null,
            total_teams: null,
            total_teams_used: 0,
            total_teams_remaining: null,
          };
          return Response.json(usage);
        }
        case "/sso/get/ui_settings":
          return Response.json({});
        case "/get/user_banner":
          return Response.json({ enabled: false, message: "", severity: "info" });
        case "/get/latest_release_info": {
          const release = { version: "", release_url: "", bug_fixes: 0, new_features: 0, other_updates: 0 };
          return Response.json(release);
        }
        case "/public/litellm_blog_posts":
          return Response.json({ posts: [] });
        default:
          throw new Error(`Unexpected request: ${url}`);
      }
    }),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
  document.cookie = "token=; Max-Age=0; Path=/";
  localStorage.clear();
});

describe("dashboard overlay boundaries", () => {
  it("should render the page with mobile navigation closed", async () => {
    renderWithProviders(
      <AuthProvider>
        <Layout>
          <h1>Gateway page</h1>
        </Layout>
      </AuthProvider>,
    );

    expect(await screen.findByRole("heading", { name: "Gateway page" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Open navigation" })).toBeVisible();
    expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument();
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
    const user = userEvent.setup();
    renderWithProviders(
      <AuthProvider>
        <Layout>{page}</Layout>
      </AuthProvider>,
    );

    await user.click(await screen.findByRole("button", { name: "Open page overlay" }));
    await screen.findByRole(role, { name: "Page overlay" });
    expect(
      screen.getAllByRole("presentation", { hidden: true }).find((element) => element.dataset.slot === slot),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Close page overlay" }));
  });

  it("should return focus to the navigation trigger after closing the drawer with Escape", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AuthProvider>
        <Layout>
          <h1>Gateway page</h1>
        </Layout>
      </AuthProvider>,
    );

    const trigger = await screen.findByRole("button", { name: "Open navigation" });
    await user.click(trigger);
    const navigation = await screen.findByRole("dialog", { name: "Navigation" });
    expect(within(navigation).getByRole("link", { name: "Virtual Keys" })).toBeVisible();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(trigger).toHaveFocus());
  });
});
