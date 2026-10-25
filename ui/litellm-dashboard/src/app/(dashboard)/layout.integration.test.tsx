import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
import UISettings from "@/components/Settings/AdminSettings/UISettings/UISettings";
import { AuthProvider } from "@/contexts/AuthContext";
import DashboardLanguageProvider from "@/contexts/DashboardLanguageProvider";
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
let deploymentWarnings: { warnings: { id: string; severity: string; ignore: null }[]; ignore_available: boolean };
let plugins: { name: string; display_name: string; url: string }[];
let defaultLanguage: "en" | "zh-CN";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/ui/guardrails",
}));

beforeEach(() => {
  testQueryClient.clear();
  localStorage.clear();
  deploymentWarnings = { warnings: [], ignore_available: true };
  plugins = [];
  defaultLanguage = "en";
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
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = input instanceof Request ? input.url : String(input);
      const path = new URL(url, "http://localhost").pathname;
      switch (path) {
        case "/litellm/.well-known/litellm-ui-config":
          return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
        case "/get/ui_settings":
          return Response.json({
            values: { default_ui_language: defaultLanguage },
            source: { default_ui_language: "db" },
            field_schema: { properties: { default_ui_language: { enum: ["en", "zh-CN"] } } },
          });
        case "/update/ui_settings":
          defaultLanguage = JSON.parse(String(init?.body)).default_ui_language;
          return Response.json({ status: "success" });
        case "/get/ui_theme_settings":
          return Response.json({ values: {} });
        case "/api/plugins":
          return Response.json(plugins);
        case "/api/plugins/auth-token":
          return Response.json({});
        case "/team/list":
        case "/organization/list":
          return Response.json([]);
        case "/health/readiness/details":
          return Response.json({ status: "healthy" });
        case "/health/deployment_warnings":
          return Response.json(deploymentWarnings);
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
        case "/management/v1/users/me":
          return Response.json({ user_id: "layout-admin", user_role: "proxy_admin", user_email: null });
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
  it("全站语言保存后更新导航，个人选择优先并可恢复跟随", async () => {
    defaultLanguage = "zh-CN";
    renderWithProviders(
      <AuthProvider>
        {/* 与根布局一致，设置页和导航共用加载全站设置的语言上下文。 */}
        <DashboardLanguageProvider>
          <Layout>
            <UISettings />
          </Layout>
        </DashboardLanguageProvider>
      </AuthProvider>,
    );
    // 先等待认证后的页面挂载，再检查异步加载的全站语言
    await screen.findByRole("combobox", { name: /^(Interface language|界面语言)$/ });
    const siteLanguage = await screen.findByRole("combobox", { name: "全站默认语言" });
    expect(screen.getByRole("link", { name: "虚拟密钥" })).toBeVisible();
    fireEvent.change(siteLanguage, { target: { value: "en" } });
    expect(await screen.findByRole("combobox", { name: "Default UI language" })).toHaveValue("en");
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "en" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Default UI language" }), { target: { value: "zh-CN" } });
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Default UI language" })).toHaveValue("zh-CN"));
    expect(screen.getByRole("link", { name: "Virtual Keys" })).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "default" } });
    expect(screen.getByRole("link", { name: "虚拟密钥" })).toBeVisible();
  });

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

describe("deployment health in the dashboard shells", () => {
  it.each([
    { shell: "gateway header", pluginMode: null },
    { shell: "plugin navbar", pluginMode: "agent-plane" },
  ])("should show the icon, the critical row and the panel in the $shell", async ({ pluginMode }) => {
    const user = userEvent.setup();
    deploymentWarnings = {
      warnings: [{ id: "env_credential_login", severity: "critical", ignore: null }],
      ignore_available: true,
    };
    if (pluginMode) {
      plugins = [{ name: pluginMode, display_name: "Agent plane", url: "" }];
      localStorage.setItem("litellm_plugin_mode", pluginMode);
    }
    renderWithProviders(
      <AuthProvider>
        <Layout>
          <h1>Gateway page</h1>
        </Layout>
      </AuthProvider>,
    );

    // 两种布局的认证就绪标志不同，插件布局不挂载普通网关页面。
    if (pluginMode) {
      await screen.findByRole("button", { name: "Switch workspace, current: Agent plane" });
    } else {
      await screen.findByRole("heading", { name: "Gateway page" });
    }
    expect(
      (await screen.findAllByRole("button", { name: "Deployment health, 1 active warning" })).length,
    ).toBeGreaterThan(0);
    await user.click(await screen.findByRole("button", { name: "Review" }));

    const panel = await screen.findByRole("dialog", { name: "Deployment health" });
    expect(within(panel).getByRole("article", { name: "Environment-credential login is enabled" })).toBeVisible();
  });
});
