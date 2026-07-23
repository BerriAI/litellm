/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { CACHE_TAB_SLUGS, cacheTabHref, slugFromPathname } from "./tabRoutes";

describe("slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(slugFromPathname("/caching")).toBe("");
    expect(slugFromPathname("/caching/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(slugFromPathname("/caching/health")).toBe("health");
    expect(slugFromPathname("/ui/caching/coordination-redis/")).toBe("coordination-redis");
  });

  it("returns the raw segment for an unknown tab so the layout can redirect to base", () => {
    expect(slugFromPathname("/ui/caching/bogus")).toBe("bogus");
  });

  it("returns empty string when the caching base segment is not in the path", () => {
    expect(slugFromPathname("/teams")).toBe("");
  });
});

describe("cacheTabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(cacheTabHref("")).toBe("/ui/caching/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of CACHE_TAB_SLUGS) {
      expect(cacheTabHref(slug)).toBe(`/ui/caching/${slug}/`);
    }
  });
});
