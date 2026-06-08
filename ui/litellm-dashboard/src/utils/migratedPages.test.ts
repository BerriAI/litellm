import { describe, it, expect, vi, beforeEach } from "vitest";

describe("migratedHref / legacyPageHref", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("builds a /ui-rooted path when serverRootPath is /", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { migratedHref, legacyPageHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/ui/api-reference");
    expect(legacyPageHref("models")).toBe("/ui/?page=models");
  });

  it("prefixes a non-root serverRootPath without duplicating slashes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/team-x/" }));
    const { migratedHref, legacyPageHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/team-x/ui/api-reference");
    expect(legacyPageHref("models")).toBe("/team-x/ui/?page=models");
  });

  it("tolerates a leading slash in the route segment", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("/api-reference")).toBe("/ui/api-reference");
  });

  it("maps the api_ref legacy id to the api-reference route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.api_ref).toBe("api-reference");
  });
});
