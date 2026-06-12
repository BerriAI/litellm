import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("migratedHref / legacyPageHref", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", "test");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
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

  it("maps the llm-playground sidebar id to the playground route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES["llm-playground"]).toBe("playground");
  });

  it("maps the projects and access-groups sidebar ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.projects).toBe("projects");
    expect(MIGRATED_PAGES["access-groups"]).toBe("access-groups");
  });

  it("maps the budgets, workflows, and guardrails-monitor sidebar ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.budgets).toBe("budgets");
    expect(MIGRATED_PAGES.workflows).toBe("workflows");
    expect(MIGRATED_PAGES["guardrails-monitor"]).toBe("guardrails-monitor");
  });

  it("maps the mcp-servers, search-tools, tag-management, vector-stores, and memory ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES["mcp-servers"]).toBe("mcp-servers");
    expect(MIGRATED_PAGES["search-tools"]).toBe("search-tools");
    expect(MIGRATED_PAGES["tag-management"]).toBe("tag-management");
    expect(MIGRATED_PAGES["vector-stores"]).toBe("vector-stores");
    expect(MIGRATED_PAGES.memory).toBe("memory");
  });

  it("maps the policies, guardrails, prompts, tool-policies, and skills ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.policies).toBe("policies");
    expect(MIGRATED_PAGES.guardrails).toBe("guardrails");
    expect(MIGRATED_PAGES.prompts).toBe("prompts");
    expect(MIGRATED_PAGES["tool-policies"]).toBe("tool-policies");
    expect(MIGRATED_PAGES.skills).toBe("skills");
    // Old bookmarks used ?page=claude-code-plugins for the same panel.
    expect(MIGRATED_PAGES["claude-code-plugins"]).toBe("skills");
  });
});

describe("dev server (NODE_ENV=development)", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", "development");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("builds root-relative hrefs because next dev serves the app at /, not /ui", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { migratedHref, legacyPageHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/api-reference");
    expect(legacyPageHref("models")).toBe("/?page=models");
  });

  it("ignores serverRootPath, which only applies to proxy-mounted deployments", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/team-x/" }));
    const { migratedHref } = await import("./migratedPages");

    expect(migratedHref("api-reference")).toBe("/api-reference");
  });

  it("maps a bare migrated path back to its legacy sidebar key", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { legacyKeyForPathname } = await import("./migratedPages");

    expect(legacyKeyForPathname("/api-reference/")).toBe("api_ref");
    expect(legacyKeyForPathname("/")).toBeNull();
  });
});

describe("legacyKeyForPathname", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", "test");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
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
