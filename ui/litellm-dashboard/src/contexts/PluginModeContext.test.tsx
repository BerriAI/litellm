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

const renderAs = (userRole: string | null) =>
  render(
    <PluginModeProvider accessToken="sk-test" userRole={userRole}>
      <ModeProbe />
    </PluginModeProvider>,
  );

const renderWithPlugins = (plugins: Plugin[]) => {
  getMock.mockResolvedValueOnce(plugins);
  return renderAs("Admin");
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
    renderAs("Admin");

    await waitFor(() => expect(getMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("ai-gateway"));
  });
});

describe("PluginModeProvider role gating", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("litellm_plugin_mode", "my-plugin");
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin", "App User"])(
    "never requests /api/plugins for %s, and still falls back to ai-gateway",
    async (role) => {
      renderAs(role);

      await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("ai-gateway"));
      expect(getMock).not.toHaveBeenCalled();
      expect(screen.getByTestId("active").textContent).toBe("none");
    },
  );

  it("holds the stored mode while the role is still resolving, so an admin does not flash to ai-gateway", async () => {
    renderAs(null);

    await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("my-plugin"));
    expect(getMock).not.toHaveBeenCalled();
  });
});
