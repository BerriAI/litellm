import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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

  it("falls back to ai-gateway once an empty plugins list loads", async () => {
    renderWithPlugins([]);

    await waitFor(() => expect(getMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("ai-gateway"));
    expect(screen.getByTestId("active").textContent).toBe("none");
  });

  it("keeps the stored mode when it is still registered", async () => {
    renderWithPlugins([{ name: "my-plugin", display_name: "My Plugin", url: "https://p.example.com" }]);

    await waitFor(() => expect(screen.getByTestId("active").textContent).toBe("my-plugin"));
    expect(screen.getByTestId("mode").textContent).toBe("my-plugin");
  });

  it("falls back to ai-gateway when the plugins fetch fails, never stranding the user", async () => {
    getMock.mockRejectedValueOnce(new Error("network down"));
    render(
      <PluginModeProvider accessToken="sk-test">
        <ModeProbe />
      </PluginModeProvider>,
    );

    await waitFor(() => expect(getMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("ai-gateway"));
  });
});
