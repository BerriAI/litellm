import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  document.head.innerHTML = "";
  vi.resetModules();
});

describe("initial server root path", () => {
  it("uses the root when runtime metadata is absent", async () => {
    const { serverRootPath } = await import("./serverRootPath");
    expect(serverRootPath).toBe("/");
  });

  it("uses image runtime metadata before backend configuration arrives", async () => {
    document.head.innerHTML = '<meta name="litellm-server-root-path" content="/services/llm">';
    const { serverRootPath, setServerRootPath } = await import("./serverRootPath");
    expect(serverRootPath).toBe("/services/llm");
    setServerRootPath("/configured");
    const updated = await import("./serverRootPath");
    expect(updated.serverRootPath).toBe("/configured");
  });
});
