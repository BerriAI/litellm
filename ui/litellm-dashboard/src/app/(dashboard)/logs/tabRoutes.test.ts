/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { LOGS_TAB_SLUGS, logsTabHref, slugFromPathname } from "./tabRoutes";

describe("slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(slugFromPathname("/logs")).toBe("");
    expect(slugFromPathname("/logs/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(slugFromPathname("/logs/audit")).toBe("audit");
    expect(slugFromPathname("/ui/logs/deleted-teams/")).toBe("deleted-teams");
  });

  it("returns the raw segment for an unknown tab so the layout can redirect to base", () => {
    expect(slugFromPathname("/ui/logs/bogus")).toBe("bogus");
  });

  it("returns empty string when the logs base segment is not in the path", () => {
    expect(slugFromPathname("/teams")).toBe("");
  });
});

describe("logsTabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(logsTabHref("")).toBe("/ui/logs/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of LOGS_TAB_SLUGS) {
      expect(logsTabHref(slug)).toBe(`/ui/logs/${slug}/`);
    }
  });
});
