/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { COST_OPTIMIZATION_TAB_SLUGS, costOptimizationTabHref, slugFromPathname } from "./tabRoutes";

describe("slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(slugFromPathname("/cost-optimization")).toBe("");
    expect(slugFromPathname("/cost-optimization/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(slugFromPathname("/cost-optimization/autorouter")).toBe("autorouter");
    expect(slugFromPathname("/ui/cost-optimization/compression/")).toBe("compression");
  });

  it("returns the raw segment for an unknown tab so the layout can redirect to base", () => {
    expect(slugFromPathname("/ui/cost-optimization/bogus")).toBe("bogus");
  });

  it("returns empty string when the cost-optimization base segment is not in the path", () => {
    expect(slugFromPathname("/teams")).toBe("");
  });
});

describe("costOptimizationTabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(costOptimizationTabHref("")).toBe("/ui/cost-optimization/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of COST_OPTIMIZATION_TAB_SLUGS) {
      expect(costOptimizationTabHref(slug)).toBe(`/ui/cost-optimization/${slug}/`);
    }
  });
});
