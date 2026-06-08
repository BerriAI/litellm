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

  it("maps both the api_ref id and the hyphenated alias to the api-reference route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.api_ref).toBe("api-reference");
    expect(MIGRATED_PAGES["api-reference"]).toBe("api-reference");
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

describe("MIGRATED_PAGES leaf cutover", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("routes every migrated leaf page to its segment and back to a sidebar key", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES, legacyKeyForPathname } = await import("./migratedPages");

    const cases: [string, string][] = [
      ["budgets", "budgets"],
      ["caching", "caching"],
      ["cost-tracking", "cost-tracking"],
      ["guardrails", "guardrails"],
      ["guardrails-monitor", "guardrails-monitor"],
      ["logs", "logs"],
      ["mcp-servers", "mcp-servers"],
      ["memory", "memory"],
      ["policies", "policies"],
      ["projects", "projects"],
      ["prompts", "prompts"],
      ["search-tools", "search-tools"],
      ["skills", "skills"],
      ["claude-code-plugins", "skills"],
      ["tag-management", "tag-management"],
      ["tool-policies", "tool-policies"],
      ["transform-request", "transform-request"],
      ["ui-theme", "ui-theme"],
      ["vector-stores", "vector-stores"],
      ["workflows", "workflows"],
      ["access-groups", "access-groups"],
    ];
    for (const [key, seg] of cases) {
      expect(MIGRATED_PAGES[key]).toBe(seg);
      expect(legacyKeyForPathname(`/ui/${seg}`)).not.toBeNull();
      expect(legacyKeyForPathname(`/ui/${seg}/`)).not.toBeNull();
    }

    // The skills alias resolves to the "skills" sidebar key, not "claude-code-plugins".
    expect(legacyKeyForPathname("/ui/skills")).toBe("skills");
    expect(legacyKeyForPathname("/ui/budgets")).toBe("budgets");
  });
});
