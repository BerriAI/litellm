import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, testQueryClient, within } from "@/../tests/test-utils";
import { setServerRootPath } from "@/lib/serverRootPath";
import { recordChatQuery } from "@/lib/telemetry/chat";
import ChatLayout from "./layout";

vi.mock("@/lib/telemetry/chat", () => ({ recordChatQuery: vi.fn(async () => {}) }));

const { router, route } = vi.hoisted(() => ({
  router: { push: vi.fn(), replace: vi.fn() },
  route: { pathname: "/ui/chat" },
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
  useUISettings: () => ({ data: { values: { enable_chat_ui: true } }, isLoading: false }),
}));

const plugin = { name: "observability", display_name: "Observability", url: "https://plugin.example.com" };
const originalLocation = Object.getOwnPropertyDescriptor(window, "location")!;
const assign = vi.fn();
let branding: { logo_url?: string; logo_url_dark?: string } = {};

describe("Chat layout product navigation", () => {
  beforeEach(() => {
    route.pathname = "/ui/chat";
    branding = {};
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
            return Response.json({ values: branding });
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
      expect(recordChatQuery).not.toHaveBeenCalled();
      await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
      expect(await screen.findByRole("menuitem", { name: "Observability" })).toBeInTheDocument();
      expect(localStorage.getItem("litellm_plugin_mode")).toBe(plugin.name);

      await user.click(screen.getByRole("menuitem", { name: "AI Gateway" }));

      expect(localStorage.getItem("litellm_plugin_mode")).toBe("ai-gateway");
      expect(assign).toHaveBeenCalledWith(gateway);
      expect(recordChatQuery).not.toHaveBeenCalled();
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
      await user.click(screen.getByRole("menuitem", { name: "AI Gateway" }));

      expect(assign).toHaveBeenCalledWith("/ui/");
      if (errorName === "QuotaExceededError") {
        expect(localStorage.getItem("litellm_plugin_mode")).toBeNull();
      }
    },
  );

  it.each([
    { label: "separate light and dark logos", light: "/branding/light.svg", dark: "/branding/dark.svg" },
    { label: "a light logo only", light: "/branding/light.svg", dark: undefined },
    { label: "a dark logo only", light: undefined, dark: "/branding/dark.svg" },
  ])(
    "should show configured organization branding with $label and retain the Chat switcher",
    async ({ light, dark }) => {
      branding = { logo_url: light, logo_url_dark: dark };
      renderWithProviders(
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>,
      );

      await userEvent.setup().click(screen.getByRole("button", { name: "Open chat navigation" }));
      const logo = await screen.findByRole("img", { name: "Organization logo" });
      const [lightImage, darkImage] = within(logo).getAllByRole("presentation", { hidden: true });
      expect(lightImage).toHaveAttribute("src", light || dark);
      expect(lightImage).toHaveClass("dark:hidden");
      expect(darkImage).toHaveAttribute("src", dark || light);
      expect(darkImage).toHaveClass("hidden", "dark:block");
      expect(screen.getByRole("button", { name: "Switch workspace, current: Chat" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    },
  );

  it("should omit organization branding when no custom logo is configured", async () => {
    renderWithProviders(
      <ChatLayout>
        <h1>Chat content</h1>
      </ChatLayout>,
    );

    await userEvent.setup().click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
    expect(await screen.findByRole("menuitem", { name: "Observability" })).toBeInTheDocument();
    await userEvent.setup().keyboard("{Escape}");
    await userEvent.setup().click(screen.getByRole("button", { name: "Open chat navigation" }));
    expect(await screen.findByRole("dialog", { name: "LiteLLM Chat" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Organization logo" })).not.toBeInTheDocument();
  });
});
