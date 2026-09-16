import { describe, expect, it } from "vitest";

import { detectLocale, detectLocaleFromList, isSupportedLocale } from "./detectLocale";

describe("detectLocale", () => {
  it("maps bare zh to zh-CN", () => {
    expect(detectLocale("zh")).toBe("zh-CN");
  });

  it("maps zh-Hans to zh-CN", () => {
    expect(detectLocale("zh-Hans")).toBe("zh-CN");
  });

  it("maps zh-CN to zh-CN", () => {
    expect(detectLocale("zh-CN")).toBe("zh-CN");
  });

  it("maps lowercase zh variants to zh-CN", () => {
    expect(detectLocale("zh-cn")).toBe("zh-CN");
    expect(detectLocale("ZH")).toBe("zh-CN");
  });

  it("maps en to en", () => {
    expect(detectLocale("en")).toBe("en");
    expect(detectLocale("en-US")).toBe("en");
  });

  it("maps unsupported languages to en (fallback)", () => {
    expect(detectLocale("fr")).toBe("en");
    expect(detectLocale("de")).toBe("en");
  });

  it("falls back to en for empty/null/undefined input", () => {
    expect(detectLocale("")).toBe("en");
    expect(detectLocale(null)).toBe("en");
    expect(detectLocale(undefined)).toBe("en");
  });

  it("trims surrounding whitespace", () => {
    expect(detectLocale("  zh-Hans ")).toBe("zh-CN");
  });
});

describe("detectLocaleFromList", () => {
  it("returns the first supported locale honouring en priority", () => {
    expect(detectLocaleFromList(["en", "zh-CN"])).toBe("en");
    expect(detectLocaleFromList(["zh-CN", "en"])).toBe("zh-CN");
  });

  it("skips unsupported tags and finds the first supported one", () => {
    expect(detectLocaleFromList(["fr", "zh"])).toBe("zh-CN");
    expect(detectLocaleFromList(["fr-FR", "en-US"])).toBe("en");
  });

  it("falls back to en when nothing is supported", () => {
    expect(detectLocaleFromList(["fr", "de"])).toBe("en");
  });
});

describe("isSupportedLocale", () => {
  it("accepts only exact supported locales", () => {
    expect(isSupportedLocale("en")).toBe(true);
    expect(isSupportedLocale("zh-CN")).toBe(true);
    expect(isSupportedLocale("zh")).toBe(false);
    expect(isSupportedLocale(null)).toBe(false);
  });
});
