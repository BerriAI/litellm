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

  it("maps the api-keys landing id to its route and builds its redirect href", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES, migratedHref } = await import("./migratedPages");

    expect(MIGRATED_PAGES["api-keys"]).toBe("api-keys");
    expect(migratedHref(MIGRATED_PAGES["api-keys"])).toBe("/ui/api-keys");
  });

  it("maps the llm-playground sidebar id to the playground route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES["llm-playground"]).toBe("playground");
  });

  it("maps the models sidebar id to the models-and-endpoints route and builds its redirect href", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES, migratedHref } = await import("./migratedPages");

    expect(MIGRATED_PAGES.models).toBe("models-and-endpoints");
    expect(migratedHref(MIGRATED_PAGES.models)).toBe("/ui/models-and-endpoints");
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

  it("maps the caching, cost-tracking, transform-request, ui-theme, and logs ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.caching).toBe("caching");
    expect(MIGRATED_PAGES["cost-tracking"]).toBe("cost-tracking");
    expect(MIGRATED_PAGES["transform-request"]).toBe("transform-request");
    expect(MIGRATED_PAGES["ui-theme"]).toBe("ui-theme");
    expect(MIGRATED_PAGES.logs).toBe("logs");
  });

  it("maps the admin-panel, logging-and-alerts, model-hub-table, and new_usage ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES["admin-panel"]).toBe("admin-panel");
    expect(MIGRATED_PAGES["logging-and-alerts"]).toBe("logging-and-alerts");
    expect(MIGRATED_PAGES["model-hub-table"]).toBe("model-hub-table");
    // new_usage routes to /usage; the legacy ?page=usage report routes to /old-usage (asserted below).
    expect(MIGRATED_PAGES.new_usage).toBe("usage");
  });

  it("maps the legacy usage report id to the old-usage route and builds its redirect href", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES, migratedHref } = await import("./migratedPages");

    expect(MIGRATED_PAGES.usage).toBe("old-usage");
    expect(migratedHref(MIGRATED_PAGES.usage)).toBe("/ui/old-usage");
  });

  it("maps the agents and router-settings ids to their routes", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.agents).toBe("agents");
    expect(MIGRATED_PAGES["router-settings"]).toBe("router-settings");
  });

  it("maps the users id to its route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.users).toBe("users");
  });

  it("maps the teams id to its route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.teams).toBe("teams");
  });

  it("maps the organizations id to its route", async () => {
    vi.doMock("@/components/networking", () => ({ serverRootPath: "/" }));
    const { MIGRATED_PAGES } = await import("./migratedPages");

    expect(MIGRATED_PAGES.organizations).toBe("organizations");
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
    // Same for skills: the claude-code-plugins alias maps to the same segment,
    // and first-match-wins iteration must keep returning the sidebar key.
    expect(legacyKeyForPathname("/ui/skills")).toBe("skills");
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
