import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Regression for the chat sidebar / first-message navigation under SERVER_ROOT_PATH.
// getChatRoutes() must read the server root path at call time. The previous
// module-level `CHAT_ROUTES` captured it once at import, before the UI-config
// bootstrap runs setServerRootPath, so every chat route was permanently
// unprefixed and router.push() navigated to a 404 (which, mid-stream, also
// aborted the first message). These tests deliberately apply the root path
// AFTER importing the module so a frozen-at-import implementation fails.
describe("getChatRoutes under server_root_path", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", "test");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("reflects a server root path applied after the module is loaded", async () => {
    const { getChatRoutes } = await import("./ChatShell");
    const { setServerRootPath } = await import("@/lib/serverRootPath");

    setServerRootPath("/gw");

    const routes = getChatRoutes();
    expect(routes.chats).toBe("/gw/ui/chat");
    expect(routes.integrations).toBe("/gw/ui/chat/integrations");
    expect(routes.credentials).toBe("/gw/ui/chat/credentials");
    expect(routes.apiKeys).toBe("/gw/ui/chat/api-keys");
    expect(routes.usage).toBe("/gw/ui/chat/usage");
  });

  it("builds /ui-rooted paths when no server root path is set", async () => {
    const { getChatRoutes } = await import("./ChatShell");
    const { setServerRootPath } = await import("@/lib/serverRootPath");

    setServerRootPath("/");

    expect(getChatRoutes().chats).toBe("/ui/chat");
    expect(getChatRoutes().integrations).toBe("/ui/chat/integrations");
  });
});
