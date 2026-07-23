/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { ROUTER_SETTINGS_TAB_SLUGS, routerSettingsTabHref, slugFromPathname } from "./tabRoutes";

describe("slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(slugFromPathname("/router-settings")).toBe("");
    expect(slugFromPathname("/router-settings/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(slugFromPathname("/router-settings/fallbacks")).toBe("fallbacks");
    expect(slugFromPathname("/ui/router-settings/prompt-caching/")).toBe("prompt-caching");
  });

  it("returns the raw segment for an unknown tab so the layout can redirect to base", () => {
    expect(slugFromPathname("/ui/router-settings/bogus")).toBe("bogus");
  });

  it("returns empty string when the router-settings base segment is not in the path", () => {
    expect(slugFromPathname("/teams")).toBe("");
  });
});

describe("routerSettingsTabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(routerSettingsTabHref("")).toBe("/ui/router-settings/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of ROUTER_SETTINGS_TAB_SLUGS) {
      expect(routerSettingsTabHref(slug)).toBe(`/ui/router-settings/${slug}/`);
    }
  });
});
