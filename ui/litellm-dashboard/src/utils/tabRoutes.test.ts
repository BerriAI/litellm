/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { createTabRoutes } from "./tabRoutes";

const routes = createTabRoutes("logs", ["audit", "deleted-keys", "deleted-teams"] as const);

describe("createTabRoutes.slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(routes.slugFromPathname("/logs")).toBe("");
    expect(routes.slugFromPathname("/logs/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(routes.slugFromPathname("/logs/audit")).toBe("audit");
    expect(routes.slugFromPathname("/ui/logs/deleted-teams/")).toBe("deleted-teams");
  });

  it("returns the raw segment for an unknown tab so the caller can redirect to base", () => {
    expect(routes.slugFromPathname("/ui/logs/bogus")).toBe("bogus");
  });

  it("returns empty string when the base segment is not in the path", () => {
    expect(routes.slugFromPathname("/teams")).toBe("");
  });
});

describe("createTabRoutes.tabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(routes.tabHref("")).toBe("/ui/logs/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of routes.slugs) {
      expect(routes.tabHref(slug)).toBe(`/ui/logs/${slug}/`);
    }
  });
});

describe("createTabRoutes metadata", () => {
  it("preserves the base segment and slug tuple", () => {
    expect(routes.baseSegment).toBe("logs");
    expect(routes.slugs).toEqual(["audit", "deleted-keys", "deleted-teams"]);
  });
});
