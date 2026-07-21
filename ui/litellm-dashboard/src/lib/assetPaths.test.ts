import { afterEach, describe, expect, it, vi } from "vitest";

import { withServerRoot } from "./assetPaths";

describe("withServerRoot", () => {
  it("leaves a root-relative path unchanged at the default root", () => {
    expect(withServerRoot("/ui/assets/logos/openai.svg", "/")).toBe("/ui/assets/logos/openai.svg");
    expect(withServerRoot("/ui/assets/logos/openai.svg", "")).toBe("/ui/assets/logos/openai.svg");
  });

  it("prefixes a custom server root path", () => {
    expect(withServerRoot("/ui/assets/logos/openai.svg", "/litellm")).toBe("/litellm/ui/assets/logos/openai.svg");
  });

  it("normalizes a trailing or missing slash in the root", () => {
    expect(withServerRoot("/ui/assets/logos/x.svg", "/litellm/")).toBe("/litellm/ui/assets/logos/x.svg");
    expect(withServerRoot("/ui/assets/logos/x.svg", "team-x")).toBe("/team-x/ui/assets/logos/x.svg");
  });

  it("guarantees a single leading slash on the path", () => {
    expect(withServerRoot("ui/assets/logos/x.svg", "/litellm")).toBe("/litellm/ui/assets/logos/x.svg");
  });
});

describe("resolveLogoSrc", () => {
  afterEach(() => {
    vi.resetModules();
    vi.doUnmock("@/lib/serverRootPath");
  });

  const importWithRoot = async (root: string) => {
    vi.resetModules();
    vi.doMock("@/lib/serverRootPath", () => ({ serverRootPath: root }));
    return import("./assetPaths");
  };

  it("returns undefined for empty values", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc(null)).toBeUndefined();
    expect(resolveLogoSrc(undefined)).toBeUndefined();
    expect(resolveLogoSrc("")).toBeUndefined();
  });

  it("passes external and inline sources through untouched", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc("https://www.zapier.com/logo.png")).toBe("https://www.zapier.com/logo.png");
    expect(resolveLogoSrc("data:image/png;base64,AAAA")).toBe("data:image/png;base64,AAAA");
    expect(resolveLogoSrc("//cdn.example.com/x.svg")).toBe("//cdn.example.com/x.svg");
  });

  it("passes bundled /_next/ asset URLs through untouched even under a sub-path mount", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc("/_next/static/media/openai_small.abc123.svg")).toBe(
      "/_next/static/media/openai_small.abc123.svg",
    );
    expect(resolveLogoSrc("/litellm-asset-prefix/_next/static/media/openai_small.abc123.svg")).toBe(
      "/litellm-asset-prefix/_next/static/media/openai_small.abc123.svg",
    );
  });

  it("roots a local asset path using the live server root path", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc("/ui/assets/logos/github.svg")).toBe("/litellm/ui/assets/logos/github.svg");
  });

  it("leaves a local asset path unchanged at the default root", async () => {
    const { resolveLogoSrc } = await importWithRoot("/");
    expect(resolveLogoSrc("/ui/assets/logos/github.svg")).toBe("/ui/assets/logos/github.svg");
  });

  it("does not double-prefix a stored value that already carries the server root path", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc("/litellm/ui/assets/logos/custom.svg")).toBe("/litellm/ui/assets/logos/custom.svg");
    expect(resolveLogoSrc("/litellm")).toBe("/litellm");
  });

  it("still prefixes a path whose first segment merely starts with the root path text", async () => {
    const { resolveLogoSrc } = await importWithRoot("/litellm");
    expect(resolveLogoSrc("/litellm-docs/logo.png")).toBe("/litellm/litellm-docs/logo.png");
  });
});
