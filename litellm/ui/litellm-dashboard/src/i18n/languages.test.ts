import { describe, expect, it } from "vitest";

import { englishName, Locales, nativeName, resolveLanguage, supportedLanguages } from "./languages";

const locales: Locales = new Map([["ja", async () => ({})]]);

describe("resolveLanguage", () => {
  it("uses the stored language when it has translations", () => {
    expect(resolveLanguage("ja", locales)).toBe("ja");
  });

  it("prefers the stored language over the browser preference", () => {
    expect(resolveLanguage("ja", locales, ["fr-FR"])).toBe("ja");
  });

  it.each([null, "", "en", "fr", "JA", "ja-JP", "__proto__"])("shows English when the stored value is %j", (stored) => {
    expect(resolveLanguage(stored, locales)).toBe("en");
  });

  it("shows English when no language besides English has translations", () => {
    expect(resolveLanguage("ja", new Map())).toBe("en");
  });

  it("matches a browser locale case-insensitively", () => {
    expect(resolveLanguage(null, locales, ["JA-jp"])).toBe("ja");
  });

  it("falls back from a regional browser locale to its base translation", () => {
    expect(resolveLanguage(null, locales, ["ja-JP"])).toBe("ja");
  });

  it("uses browser language preferences in order", () => {
    const available: Locales = new Map([
      ["ja", async () => ({})],
      ["fr", async () => ({})],
    ]);

    expect(resolveLanguage(null, available, ["de-DE", "fr-CA", "ja-JP"])).toBe("fr");
  });

  it("does not guess between regional variants for an ambiguous base language", () => {
    const chinese: Locales = new Map([
      ["zh-CN", async () => ({})],
      ["zh-TW", async () => ({})],
    ]);

    expect(resolveLanguage(null, chinese, ["zh"])).toBe("en");
  });
});

describe("supportedLanguages", () => {
  it("puts English first, then every language with translations", () => {
    expect(supportedLanguages(locales)).toEqual(["en", "ja"]);
    expect(supportedLanguages(new Map())).toEqual(["en"]);
  });
});

describe("language names", () => {
  it("names a language in itself, starting with a capital letter", () => {
    const spanish = nativeName("es");

    expect(spanish.toLocaleLowerCase("es")).toBe(
      new Intl.DisplayNames(["es"], { type: "language" }).of("es")?.toLocaleLowerCase("es"),
    );
    expect(spanish.charAt(0)).toBe(spanish.charAt(0).toLocaleUpperCase("es"));
  });

  it("names a language in English for the secondary label", () => {
    expect(englishName("ja")).toBe(new Intl.DisplayNames(["en"], { type: "language" }).of("ja"));
    expect(englishName("ja")).not.toBe(nativeName("ja"));
  });
});
