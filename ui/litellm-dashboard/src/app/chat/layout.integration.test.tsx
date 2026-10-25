import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, testQueryClient } from "@/../tests/test-utils";
import { setServerRootPath } from "@/lib/serverRootPath";
import ChatLayout from "./layout";
import DashboardLanguageProvider from "@/contexts/DashboardLanguageProvider";

const { router, route, settings } = vi.hoisted(() => ({
  router: { push: vi.fn(), replace: vi.fn() },
  route: { pathname: "/ui/chat" },
  settings: { defaultLanguage: "en" },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
  usePathname: () => route.pathname,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "chat-layout-test-token",
    userId: "chat-layout-user",
    userEmail: "developer@example.com",
    userRole: "internal_user",
    premiumUser: false,
    isLoading: false,
    isAuthorized: true,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({
    data: { values: { enable_chat_ui: true, default_ui_language: settings.defaultLanguage } },
    isLoading: false,
  }),
}));
// 实际根布局的 AuthProvider 与 useAuthorized 返回同一账户。
vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ userID: "chat-layout-user" }) }));

const plugin = { name: "observability", display_name: "Observability", url: "https://plugin.example.com" };
const originalLocation = Object.getOwnPropertyDescriptor(window, "location")!;
const assign = vi.fn();

describe("Chat layout product navigation", () => {
  beforeEach(() => {
    route.pathname = "/ui/chat";
    settings.defaultLanguage = "en";
    localStorage.clear();
    localStorage.setItem("litellm_plugin_mode", plugin.name);
    testQueryClient.clear();
    assign.mockReset();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, assign },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = input instanceof Request ? input.url : String(input);
        const path = new URL(url, "http://localhost").pathname;
        switch (path) {
          case "/api/plugins":
            expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer chat-layout-test-token");
            return Response.json([plugin]);
          case "/get/ui_theme_settings":
            return Response.json({ values: {} });
          case "/sso/get/ui_settings":
            return Response.json({});
          default:
            throw new Error(`Unexpected request: ${path}`);
        }
      }),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    testQueryClient.clear();
    localStorage.clear();
    setServerRootPath("/");
    Object.defineProperty(window, "location", originalLocation);
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each([
    { pathname: "/ui/chat", root: "/", gateway: "/ui/" },
    { pathname: "/gateway/ui/chat/models/", root: "/gateway", gateway: "/gateway/ui/" },
  ])(
    "should render Chat at $pathname and clear a saved plugin when returning to AI Gateway",
    async ({ pathname, root, gateway }) => {
      route.pathname = pathname;
      setServerRootPath(root);
      const user = userEvent.setup();
      renderWithProviders(
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>,
      );

      expect(screen.getByRole("heading", { name: "Chat content" })).toBeVisible();
      await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
      expect(await screen.findByRole("menuitem", { name: "Observability" })).toBeInTheDocument();
      expect(localStorage.getItem("litellm_plugin_mode")).toBe(plugin.name);

      await user.click(screen.getByRole("menuitem", { name: "AI Gateway", exact: true }));

      expect(localStorage.getItem("litellm_plugin_mode")).toBe("ai-gateway");
      expect(assign).toHaveBeenCalledWith(gateway);
    },
  );

  it.each(["SecurityError", "QuotaExceededError"])(
    "should still navigate to AI Gateway when browser storage throws %s",
    async (errorName) => {
      if (errorName === "SecurityError") {
        vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
          throw new DOMException("Storage is unavailable", errorName);
        });
        vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
          throw new DOMException("Storage is unavailable", errorName);
        });
      }
      vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
        throw new DOMException("Storage is unavailable", errorName);
      });
      const user = userEvent.setup();
      renderWithProviders(
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>,
      );

      expect(screen.getByRole("heading", { name: "Chat content" })).toBeVisible();
      await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
      expect(await screen.findByRole("menuitem", { name: "Observability" })).toBeInTheDocument();
      await user.click(screen.getByRole("menuitem", { name: "AI Gateway", exact: true }));

      expect(assign).toHaveBeenCalledWith("/ui/");
      if (errorName === "QuotaExceededError") {
        expect(localStorage.getItem("litellm_plugin_mode")).toBeNull();
      }
    },
  );

  it("沿用全站默认语言，并记住个人选择用于聊天导航", async () => {
    settings.defaultLanguage = "zh-CN";
    const user = userEvent.setup();
    const view = renderWithProviders(
      <DashboardLanguageProvider>
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>
      </DashboardLanguageProvider>,
    );
    await user.click(screen.getByRole("button", { name: "打开聊天导航" }));
    expect(screen.getByRole("button", { name: "新建对话" })).toBeVisible();
    await user.selectOptions(screen.getByRole("combobox", { name: "界面语言" }), "en");
    expect(screen.getByRole("button", { name: "New Chat" })).toBeVisible();
    view.unmount();
    renderWithProviders(
      <DashboardLanguageProvider>
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>
      </DashboardLanguageProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Open chat navigation" }));
    expect(screen.getByRole("button", { name: "New Chat" })).toBeVisible();
  });
});
