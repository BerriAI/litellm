import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, testQueryClient, within } from "@/../tests/test-utils";
import { setServerRootPath } from "@/lib/serverRootPath";
import { getProxyBaseUrl } from "@/components/networking";
import ChatLayout from "./layout";

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
const LIGHT_LOGO = "https://cdn.test/light.svg";
const DARK_LOGO = "https://cdn.test/dark.svg";
let themeResponse: unknown = { values: {} };

describe("Chat layout product navigation", () => {
  beforeEach(() => {
    themeResponse = { values: {} };
    route.pathname = "/ui/chat";
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
            return Response.json(themeResponse);
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
    ["public", [LIGHT_LOGO, DARK_LOGO], [false, false], [LIGHT_LOGO, DARK_LOGO]],
    ["local", [null, null], [true, true], ["/get_image", "/get_image?theme=dark"]],
    ["public light and local dark", [LIGHT_LOGO, null], [false, true], [LIGHT_LOGO, "/get_image?theme=dark"]],
    ["local light and public dark", [null, DARK_LOGO], [true, false], ["/get_image", DARK_LOGO]],
    ["public light only", [LIGHT_LOGO, null], [false, false], [LIGHT_LOGO, LIGHT_LOGO]],
    ["public dark only", [null, DARK_LOGO], [false, false], [DARK_LOGO, DARK_LOGO]],
    ["local light only", [null, null], [true, false], ["/get_image", "/get_image"]],
    ["local dark only", [null, null], [false, true], ["/get_image?theme=dark", "/get_image?theme=dark"]],
  ] as const)(
    "should render %s organization logos from the theme response",
    async (
      _label,
      [logo_url, logo_url_dark],
      [logo_url_via_proxy, logo_url_dark_via_proxy],
      [lightSource, darkSource],
    ) => {
      themeResponse = { values: { logo_url, logo_url_dark }, logo_url_via_proxy, logo_url_dark_via_proxy };
      const user = userEvent.setup();
      renderWithProviders(
        <ChatLayout>
          <h1>Chat content</h1>
        </ChatLayout>,
      );

      await user.click(screen.getByRole("button", { name: "Open chat navigation" }));
      const logo = await screen.findByRole("img", { name: "Organization logo" });
      const [light, dark] = within(logo).getAllByAltText("");
      const source = (path: string) => (path.startsWith("/") ? `${getProxyBaseUrl()}${path}` : path);
      expect(light).toHaveAttribute("src", source(lightSource));
      expect(dark).toHaveAttribute("src", source(darkSource));
      expect(light).toHaveClass("dark:hidden");
      expect(dark).toHaveClass("hidden", "dark:block");
    },
  );

  it("should omit organization branding when no logo is configured", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ChatLayout>
        <h1>Chat content</h1>
      </ChatLayout>,
    );

    await user.click(screen.getByRole("button", { name: "Open chat navigation" }));
    expect(await screen.findByRole("dialog", { name: "LiteLLM Chat" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Organization logo" })).not.toBeInTheDocument();
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
});
