import { afterEach, describe, expect, it, vi } from "vitest";
import { setServerRootPath } from "@/lib/serverRootPath";
import { routeSegmentForPathname, uiHref } from "./uiHref";

afterEach(() => {
  setServerRootPath("/");
  vi.unstubAllEnvs();
});

describe("uiHref", () => {
  it("builds a /ui-rooted path when serverRootPath is /", () => {
    expect(uiHref("api-reference")).toBe("/ui/api-reference");
  });

  it("prefixes a non-root serverRootPath without duplicating slashes", () => {
    setServerRootPath("/team-x/");
    expect(uiHref("api-reference")).toBe("/team-x/ui/api-reference");
  });

  it("tolerates a leading slash in the route segment", () => {
    expect(uiHref("/api-reference")).toBe("/ui/api-reference");
  });

  it("stays root-relative under next dev, which serves the app at /", () => {
    vi.stubEnv("NODE_ENV", "development");
    expect(uiHref("logs")).toBe("/logs");
  });
});

describe("routeSegmentForPathname", () => {
  it("strips the /ui base and any trailing slash", () => {
    expect(routeSegmentForPathname("/ui/api-reference")).toBe("api-reference");
    expect(routeSegmentForPathname("/ui/api-reference/")).toBe("api-reference");
  });

  it("returns an empty segment for the dashboard root", () => {
    expect(routeSegmentForPathname("/ui/")).toBe("");
    expect(routeSegmentForPathname("/ui")).toBe("");
  });

  it("keeps only the first segment of a nested path", () => {
    expect(routeSegmentForPathname("/ui/models-and-endpoints/anything")).toBe("models-and-endpoints");
  });

  it("strips a non-root serverRootPath too", () => {
    setServerRootPath("/team-x/");
    expect(routeSegmentForPathname("/team-x/ui/guardrails")).toBe("guardrails");
  });

  it("reads the segment straight after / under next dev", () => {
    vi.stubEnv("NODE_ENV", "development");
    expect(routeSegmentForPathname("/logs")).toBe("logs");
    expect(routeSegmentForPathname("/")).toBe("");
  });
});
