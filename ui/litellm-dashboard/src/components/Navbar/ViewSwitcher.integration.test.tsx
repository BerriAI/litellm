import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { renderWithProviders, screen, testQueryClient, within } from "@/../tests/test-utils";
import { PluginModeProvider, type Plugin } from "@/contexts/PluginModeContext";
import { setServerRootPath } from "@/lib/serverRootPath";

import ViewSwitcher from "./ViewSwitcher";

const { state, getUiSettingsMock } = vi.hoisted(() => ({
  state: { pathname: "/ui/", plugins: [] as Plugin[] },
  getUiSettingsMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({ usePathname: () => state.pathname }));
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "",
  getUiSettings: getUiSettingsMock,
}));

function renderSwitcher() {
  return renderWithProviders(
    <PluginModeProvider accessToken="test-session">
      <ViewSwitcher />
    </PluginModeProvider>,
  );
}

describe("ViewSwitcher", () => {
  const originalLocation = window.location;
  const assignSpy = vi.fn();

  beforeEach(() => {
    state.pathname = "/ui/";
    state.plugins = [];
    setServerRootPath("/");
    localStorage.clear();
    getUiSettingsMock.mockResolvedValue({ values: { enable_chat_ui: false } });
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { href: "http://localhost:3000/ui/", origin: "http://localhost:3000", assign: assignSpy },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input) === "/api/plugins") return Response.json(state.plugins);
        throw new Error(`Unexpected switcher request: ${String(input)}`);
      }),
    );
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
    testQueryClient.clear();
    localStorage.clear();
    setServerRootPath("/");
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("should render the workspace selector and keep disabled Chat unavailable", async () => {
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: AI Gateway" }));
    const chat = await screen.findByRole("menuitem", {
      name: "Chat Chat UI for using LLMs, MCPs, Agents Admins can enable in Settings",
    });
    expect(chat).toHaveAttribute("aria-disabled", "true");
    expect(within(chat).queryByRole("img")).not.toBeInTheDocument();
    expect(assignSpy).not.toHaveBeenCalled();
  });

  it("should show the active plugin and switch plugin modes within the gateway", async () => {
    state.plugins = [
      { name: "metrics", display_name: "Metrics", url: "/metrics" },
      { name: "obs", display_name: "Observability", url: "/observability" },
    ];
    localStorage.setItem("litellm_plugin_mode", "metrics");
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(await screen.findByRole("button", { name: "Switch workspace, current: Metrics" }));
    await user.click(await screen.findByRole("menuitem", { name: "Observability" }));
    expect(screen.getByRole("button", { name: "Switch workspace, current: Observability" })).toBeInTheDocument();
    expect(assignSpy).not.toHaveBeenCalled();
  });

  it("should navigate from the gateway to enabled Chat without changing plugin mode", async () => {
    getUiSettingsMock.mockResolvedValue({ values: { enable_chat_ui: true } });
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: AI Gateway" }));
    const chat = await screen.findByRole("menuitem", { name: "Chat Chat UI for using LLMs, MCPs, Agents" });
    await user.click(chat);
    expect(assignSpy).toHaveBeenCalledWith("/ui/chat");
    expect(localStorage.getItem("litellm_plugin_mode")).toBeNull();
  });

  it.each(["/ui/chat", "/ui/chat/", "/ui/chat/models"])(
    "should identify Chat at %s and navigate back while settings are loading",
    async (pathname) => {
      state.pathname = pathname;
      getUiSettingsMock.mockImplementation(() => new Promise(() => {}));
      const user = userEvent.setup();
      renderSwitcher();

      const trigger = screen.getByRole("button", { name: "Switch workspace, current: Chat" });
      expect(trigger).toHaveTextContent("Chat");
      expect(within(trigger).queryByRole("img")).not.toBeInTheDocument();
      await user.click(trigger);
      await user.click(await screen.findByRole("menuitem", { name: "AI Gateway" }));
      expect(assignSpy).toHaveBeenCalledWith("/ui/");
      expect(localStorage.getItem("litellm_plugin_mode")).toBe("ai-gateway");
    },
  );

  it("should preserve a proxy mount path when switching from a Chat subpage to a plugin", async () => {
    setServerRootPath("/gateway");
    state.pathname = "/gateway/ui/chat/models/";
    state.plugins = [{ name: "metrics", display_name: "Metrics", url: "/metrics" }];
    getUiSettingsMock.mockImplementation(() => new Promise(() => {}));
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
    await user.click(await screen.findByRole("menuitem", { name: "Metrics" }));
    expect(assignSpy).toHaveBeenCalledWith("/gateway/ui/");
    expect(localStorage.getItem("litellm_plugin_mode")).toBe("metrics");
  });

  it.each(["/chat", "/ui/chat", "/ui/chat/models"])(
    "should keep Chat identity and gateway navigation at the development route %s",
    async (pathname) => {
      vi.stubEnv("NODE_ENV", "development");
      state.pathname = pathname;
      const user = userEvent.setup();
      renderSwitcher();

      await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
      await user.click(await screen.findByRole("menuitem", { name: "AI Gateway" }));
      expect(assignSpy).toHaveBeenCalledWith("/");
    },
  );

  it.each(["/ui/moyai", "/ui/moyai/sessions/one"])(
    "should show Moyai's icon at %s and let users switch to Chat",
    async (pathname) => {
      state.pathname = pathname;
      getUiSettingsMock.mockResolvedValue({ values: { enable_chat_ui: true } });
      const user = userEvent.setup();
      renderSwitcher();

      const trigger = screen.getByRole("button", { name: "Switch workspace, current: Moyai" });
      expect(within(trigger).getByAltText("")).toHaveAttribute("src", expect.stringContaining("moyai-head.svg"));
      await user.click(trigger);
      await user.click(await screen.findByRole("menuitem", { name: "Chat Chat UI for using LLMs, MCPs, Agents" }));
      expect(assignSpy).toHaveBeenCalledWith("/ui/chat");
    },
  );

  it("should switch from Chat to the connected Moyai deployment", async () => {
    state.pathname = "/ui/chat";
    const settings = { values: { enable_chat_ui: true, moyai_url: "https://moyai.example.com" } };
    getUiSettingsMock.mockResolvedValue(settings);
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
    await user.click(await screen.findByRole("menuitem", { name: "Moyai Cloud Coding Agent" }));
    expect(assignSpy).toHaveBeenCalledWith("https://moyai.example.com");
  });

  it("should keep the local Moyai route available when Chat is disabled", async () => {
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: AI Gateway" }));
    await user.click(await screen.findByRole("menuitem", { name: "Moyai Cloud Coding Agent" }));
    expect(assignSpy).toHaveBeenCalledWith("/ui/moyai");
  });

  it("should navigate from Moyai back to the gateway", async () => {
    state.pathname = "/ui/moyai";
    const user = userEvent.setup();
    renderSwitcher();

    await user.click(screen.getByRole("button", { name: "Switch workspace, current: Moyai" }));
    await user.click(await screen.findByRole("menuitem", { name: "AI Gateway" }));
    expect(assignSpy).toHaveBeenCalledWith("/ui/");
  });
});
