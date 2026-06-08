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

  it("maps legacy page ids (and the hyphenated api-reference alias) to their route segments", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.api_ref).toBe("api-reference");
    expect(MIGRATED_PAGES["api-reference"]).toBe("api-reference");
    expect(MIGRATED_PAGES["llm-playground"]).toBe("playground");
  });
});

describe("legacyKeyForPathname", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("maps a migrated path back to its legacy sidebar key (including trailing slash)", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { legacyKeyForPathname } = await import("./migratedPages");

    // Resolves to the sidebar key api_ref, not the hyphenated alias, so highlighting works.
    expect(legacyKeyForPathname("/ui/api-reference")).toBe("api_ref");
    expect(legacyKeyForPathname("/ui/api-reference/")).toBe("api_ref");
    expect(legacyKeyForPathname("/ui/playground")).toBe("llm-playground");
    expect(legacyKeyForPathname("/ui/playground/")).toBe("llm-playground");
  });

  it("returns null for a not-yet-migrated path", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { legacyKeyForPathname } = await import("./migratedPages");

    expect(legacyKeyForPathname("/ui/")).toBeNull();
    expect(legacyKeyForPathname("/ui/some-legacy-page")).toBeNull();
  });

  it("strips a non-root serverRootPath prefix before matching", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/team-x/" }));
    const { legacyKeyForPathname } = await import("./migratedPages");

    expect(legacyKeyForPathname("/team-x/ui/api-reference")).toBe("api_ref");
    expect(legacyKeyForPathname("/ui/api-reference")).toBeNull();
  });
});
