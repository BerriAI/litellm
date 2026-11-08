import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { PluginModeProvider, usePluginMode } from "./PluginModeContext";
import type { Plugin } from "./PluginModeContext";

const { getMock } = vi.hoisted(() => ({ getMock: vi.fn() }));

vi.mock("@/lib/http/client", () => ({
  createApiClient: () => ({ get: getMock }),
}));
vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "" }));

function ModeProbe() {
  const { mode, activePlugin } = usePluginMode();
  return (
    <div>
      <span data-testid="mode">{mode}</span>
      <span data-testid="active">{activePlugin?.name ?? "none"}</span>
    </div>
  );
}

const renderWithPlugins = (plugins: Plugin[]) => {
  getMock.mockResolvedValueOnce(plugins);
  return render(
    <PluginModeProvider accessToken="sk-test">
      <ModeProbe />
    </PluginModeProvider>,
  );
};

describe("PluginModeProvider effectiveMode fallback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("litellm_plugin_mode", "my-plugin");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it("should fall back to ai-gateway once an empty plugins list loads", async () => {
    renderWithPlugins([]);

    await waitFor(() => expect(getMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("mode")).toHaveTextContent("ai-gateway"));
    expect(screen.getByTestId("active")).toHaveTextContent("none");
  });

  it("should keep the stored mode when it is still registered", async () => {
    renderWithPlugins([{ name: "my-plugin", display_name: "My Plugin", url: "https://p.example.com" }]);

    await waitFor(() => expect(screen.getByTestId("active")).toHaveTextContent("my-plugin"));
    expect(screen.getByTestId("mode")).toHaveTextContent("my-plugin");
  });

  it("should fall back to ai-gateway when the plugins fetch fails, never stranding the user", async () => {
    getMock.mockRejectedValueOnce(new Error("network down"));
    render(
      <PluginModeProvider accessToken="sk-test">
        <ModeProbe />
      </PluginModeProvider>,
    );

    await waitFor(() => expect(getMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("mode")).toHaveTextContent("ai-gateway"));
  });

  it("should render the gateway when browser storage cannot be read", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("Storage is blocked", "SecurityError");
    });

    renderWithPlugins([]);

    expect(screen.getByTestId("mode")).toHaveTextContent("ai-gateway");
    await waitFor(() => expect(getMock).toHaveBeenCalled());
  });

  it("should retain the selected workspace in memory when browser storage is full", async () => {
    getMock.mockResolvedValueOnce([{ name: "my-plugin", display_name: "My Plugin", url: "https://p.example.com" }]);
    const { result } = renderHook(usePluginMode, {
      wrapper: ({ children }) => <PluginModeProvider accessToken="sk-test">{children}</PluginModeProvider>,
    });
    await waitFor(() => expect(result.current.activePlugin?.name).toBe("my-plugin"));
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Storage is full", "QuotaExceededError");
    });

    act(() => result.current.setMode("ai-gateway"));

    expect(result.current.mode).toBe("ai-gateway");
    expect(result.current.activePlugin).toBeNull();
    expect(localStorage.getItem("litellm_plugin_mode")).toBeNull();
  });
});
