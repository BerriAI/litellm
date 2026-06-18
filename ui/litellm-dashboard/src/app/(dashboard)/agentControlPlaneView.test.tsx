import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import { AgentControlPlaneView } from "./layout";

const pluginModeValue = {
  mode: "litellm-platform-plugin" as string,
  setMode: vi.fn(),
  plugins: [{ name: "litellm-platform-plugin", display_name: "Chat UI", url: "http://localhost:3300" }],
  activePlugin: { name: "litellm-platform-plugin", display_name: "Chat UI", url: "http://localhost:3300" } as {
    name: string;
    display_name: string;
    url: string;
  } | null,
};

vi.mock("@/contexts/PluginModeContext", () => ({ usePluginMode: () => pluginModeValue }));
vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ accessToken: "sk-test-token" }) }));

// Stub the proxy client so the auth-token fetch is inert in tests.
vi.mock("@/lib/http/client", () => ({
  createApiClient: () => ({ get: vi.fn(() => Promise.resolve({ encrypted_token: "enc" })) }),
}));
vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "" }));

describe("AgentControlPlaneView iframe", () => {
  it("embeds the plugin at its ROOT url, never a hardcoded subpath like /sessions", () => {
    const { container } = render(<AgentControlPlaneView />);
    const iframe = container.querySelector("iframe");

    expect(iframe).not.toBeNull();
    const src = iframe!.getAttribute("src")!;
    expect(src).toBe("http://localhost:3300/");
    expect(src).not.toContain("/sessions");
    // title comes from the plugin's display_name, not a hardcoded label
    expect(iframe!.getAttribute("title")).toBe("Chat UI");
  });

  it("does not double the slash when the plugin url has a trailing slash", () => {
    pluginModeValue.activePlugin = {
      name: "litellm-platform-plugin",
      display_name: "Chat UI",
      url: "http://localhost:3300/",
    };
    const { container } = render(<AgentControlPlaneView />);
    expect(container.querySelector("iframe")!.getAttribute("src")).toBe("http://localhost:3300/");
    pluginModeValue.activePlugin = {
      name: "litellm-platform-plugin",
      display_name: "Chat UI",
      url: "http://localhost:3300",
    };
  });

  it("does not leak the raw token in the iframe src (token goes via encrypted postMessage)", () => {
    const { container } = render(<AgentControlPlaneView />);
    expect(container.querySelector("iframe")!.getAttribute("src")).not.toContain("token");
  });
});
