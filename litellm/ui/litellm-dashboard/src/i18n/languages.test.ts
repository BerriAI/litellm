import { describe, expect, it } from "vitest";

import { englishName, Locales, nativeName, resolveLanguage, supportedLanguages } from "./languages";

const locales: Locales = new Map([["ja", async () => ({})]]);

describe("resolveLanguage", () => {
  it("uses the stored language when it has translations", () => {
    expect(resolveLanguage("ja", locales)).toBe("ja");
  });

  it.each([null, "", "en", "fr", "JA", "ja-JP", "__proto__"])("shows English when the stored value is %j", (stored) => {
    expect(resolveLanguage(stored, locales)).toBe("en");
  });

  it("shows English when no language besides English has translations", () => {
    expect(resolveLanguage("ja", new Map())).toBe("en");
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
