/* @vitest-environment jsdom */
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { MODEL_TAB_SLUGS, modelTabHref, slugFromPathname } from "./tabRoutes";

describe("slugFromPathname", () => {
  it("returns empty string for the base path with or without a trailing slash", () => {
    expect(slugFromPathname("/models-and-endpoints")).toBe("");
    expect(slugFromPathname("/models-and-endpoints/")).toBe("");
  });

  it("extracts the tab slug from dev and proxy-mounted (/ui) paths", () => {
    expect(slugFromPathname("/models-and-endpoints/add")).toBe("add");
    expect(slugFromPathname("/ui/models-and-endpoints/llm-credentials/")).toBe("llm-credentials");
  });

  it("returns the raw segment for an unknown tab so the view can redirect to base", () => {
    expect(slugFromPathname("/ui/models-and-endpoints/bogus")).toBe("bogus");
  });

  it("returns empty string when the models base segment is not in the path", () => {
    expect(slugFromPathname("/teams")).toBe("");
  });
});

describe("modelTabHref", () => {
  it("builds the trailing-slash base href for the empty slug", () => {
    expect(modelTabHref("")).toBe("/ui/models-and-endpoints/");
  });

  it("builds a trailing-slash href for every tab slug (required by static export)", () => {
    for (const slug of MODEL_TAB_SLUGS) {
      expect(modelTabHref(slug)).toBe(`/ui/models-and-endpoints/${slug}/`);
    }
  });
});
