import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("migratedHref", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.unstubAllEnvs();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns an absolute path rooted at / when serverRootPath is /", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/api-reference");
    expect(migratedHref("virtual-keys")).toBe("/virtual-keys");
  });

  it("prefixes a non-root serverRootPath without duplicating slashes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/team-x/" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/team-x/api-reference");
  });

  it("honors NEXT_PUBLIC_BASE_URL as a base path segment", async () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_URL", "ui");
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/ui/api-reference");
  });

  it("combines NEXT_PUBLIC_BASE_URL with a non-root serverRootPath", async () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_URL", "ui");
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/team-x" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/team-x/ui/api-reference");
  });
});
