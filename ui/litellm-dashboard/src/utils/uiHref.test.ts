import { afterEach, describe, expect, it } from "vitest";
import { setServerRootPath } from "@/lib/serverRootPath";
import { appHrefFromUiHref, routeSegmentForPathname, uiHref } from "./uiHref";

afterEach(() => {
  setServerRootPath("/");
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
});

describe("appHrefFromUiHref", () => {
  it("strips the /ui base so the basePath-aware router does not double-prefix", () => {
    expect(appHrefFromUiHref("/ui/teams?team=x")).toBe("/teams?team=x");
  });

  it("maps the bare base to the app root", () => {
    expect(appHrefFromUiHref("/ui")).toBe("/");
  });

  it("strips a non-root serverRootPath base too", () => {
    setServerRootPath("/team-x/");
    expect(appHrefFromUiHref("/team-x/ui/guardrails")).toBe("/guardrails");
  });

  it("passes through an href that is already app-relative", () => {
    expect(appHrefFromUiHref("/guardrails")).toBe("/guardrails");
  });
});

describe("routeSegmentForPathname", () => {
  it("reads the first segment of the basePath-stripped pathname", () => {
    expect(routeSegmentForPathname("/api-reference")).toBe("api-reference");
    expect(routeSegmentForPathname("/api-reference/")).toBe("api-reference");
  });

  it("returns an empty segment for the dashboard root", () => {
    expect(routeSegmentForPathname("/")).toBe("");
  });

  it("keeps only the first segment of a nested path", () => {
    expect(routeSegmentForPathname("/models-and-endpoints/anything")).toBe("models-and-endpoints");
  });
});
