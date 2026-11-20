import { afterEach, describe, expect, it, vi } from "vitest";

import { cachedFormat } from "./formats";
import i18n from "./i18n";
import en from "./locales/en.json";

type Messages = { readonly [key: string]: string | Messages };

const entries = (node: Messages, prefix: string): ReadonlyArray<readonly [string, string]> =>
  Object.entries(node).flatMap(([key, value]) =>
    typeof value === "string" ? [[`${prefix}${key}`, value] as const] : entries(value, `${prefix}${key}.`),
  );

const strings = (node: Messages): readonly string[] => entries(node, "").map(([, text]) => text);

const formatsInEnglish = [
  ...new Set(strings(en).flatMap((text) => [...text.matchAll(/\{\{\s*\w+\s*,\s*([^}]+?)\s*\}\}/g)].map(([, f]) => f))),
];

const useLanguage = async (language: string) => {
  i18n.addResourceBundle(language, "translation", {});
  await i18n.changeLanguage(language);
};

afterEach(async () => {
  await i18n.changeLanguage("en");
  ["de", "zh-CN", "ja", "fr", "en-GB"].forEach((language) => i18n.removeResourceBundle(language, "translation"));
});

describe("format keys in English", () => {
  it("formats numbers and USD the way the dashboard shows them today", () => {
    expect(i18n.t("format.number", { val: 9123456 })).toBe("9,123,456");
    expect(i18n.t("format.number", { val: 80, minimumFractionDigits: 1, maximumFractionDigits: 1 })).toBe("80.0");
    expect(i18n.t("format.compactNumber", { val: 75000 })).toBe("75K");
    expect(i18n.t("format.compactNumber", { val: 41_730_000 })).toBe("41.7M");
    expect(i18n.t("format.compactNumber", { val: 75000, notation: "standard" })).toBe("75K");
    expect(i18n.t("format.usd", { val: 1234.5 })).toBe("$1,234.50");
    expect(i18n.t("format.usd", { val: 0.0838, minimumFractionDigits: 4, maximumFractionDigits: 4 })).toBe("$0.0838");
  });

  it("formats dates in local time with a 24-hour clock that starts at 00", () => {
    const morning = new Date(2026, 8, 5, 4, 3, 2);
    expect(i18n.t("format.date", { val: morning })).toBe("Sep 5, 2026");
    expect(i18n.t("format.date", { val: morning, month: "long" })).toBe("Sep 5, 2026");
    expect(i18n.t("format.monthDay", { val: morning })).toBe("Sep 5");
    expect(i18n.t("format.monthDayTime", { val: morning })).toBe("Sep 5, 04:03:02");
    expect(i18n.t("format.timestamp", { val: morning })).toBe("Sep 5, 2026, 04:03:02");
    expect(i18n.t("format.numericDateTime", { val: morning })).toBe("9/5/2026, 4:03:02 AM");
    expect(i18n.t("format.monthDayTime", { val: new Date(2026, 0, 2, 0, 0, 5) })).toBe("Jan 2, 00:00:05");
  });

  it("accepts epoch milliseconds and ISO strings as dates, and a time zone from the caller", () => {
    const iso = "2026-09-05T12:30:00Z";
    const expected = i18n.t("format.timestamp", { val: new Date(iso) });
    expect(i18n.t("format.timestamp", { val: iso })).toBe(expected);
    expect(i18n.t("format.timestamp", { val: Date.parse(iso) })).toBe(expected);
    expect(i18n.t("format.timestamp", { val: iso, timeZone: "UTC" })).toBe("Sep 5, 2026, 12:30:00");
  });

  it("formats relative time in the unit the caller names", () => {
    expect(i18n.t("format.relativeTime", { val: -3 })).toBe("3 days ago");
    expect(i18n.t("format.relativeTime", { val: 2, range: "hour" })).toBe("in 2 hours");
  });

  it("only names formatters that are registered", () => {
    expect(formatsInEnglish.length).toBeGreaterThan(0);
    const unformatted = formatsInEnglish.filter(
      (format) => typeof i18n.services.formatter?.format(0, format, "en", {}) !== "string",
    );
    expect(unformatted).toEqual([]);
  });

  it("keeps every formatted placeholder alone in a format key", () => {
    const formatted = entries(en, "").filter(([, text]) => /\{\{[^}]*,[^}]*\}\}/.test(text));
    expect(formatted.length).toBeGreaterThan(0);
    const mixed = formatted.filter(([key, text]) => !key.startsWith("format.") || !/^\{\{val, [a-z]+\}\}$/.test(text));
    expect(mixed).toEqual([]);
  });

  it("keeps a currency asked of the number formats in US dollars", () => {
    expect(i18n.t("format.number", { val: 1234.5, style: "currency", currency: "EUR" })).toBe("$1,234.50");
    expect(i18n.t("format.compactNumber", { val: 1_234_567, style: "currency", currency: "EUR" })).toBe("$1.2M");
  });

  it("reads underscore language codes the way i18next does", () => {
    expect(i18n.t("format.number", { val: 1234.5, lng: "de_DE" })).toBe(new Intl.NumberFormat("de-DE").format(1234.5));
  });

  it("still formats when the options cannot be serialized", () => {
    const circular: { self?: unknown } = {};
    circular.self = circular;
    expect(i18n.t("format.number", { val: 1234.5, circular })).toBe("1,234.5");
  });

  it("builds one formatter per language and options, not one per value", () => {
    const build = vi.fn((lng: string) => (value: number) => `${lng} ${value}`);
    i18n.services.formatter?.add("probe", cachedFormat(build));
    const render = (val: number, lng: string) =>
      i18n.services.interpolator.interpolate("{{val, probe}}", { val }, lng, {});
    expect([1, 2, 3].map((val) => render(val, "en"))).toEqual(["en 1", "en 2", "en 3"]);
    expect(build).toHaveBeenCalledTimes(1);
    expect(render(4, "de")).toBe("de 4");
    expect(build).toHaveBeenCalledTimes(2);
  });
});

describe("format keys in another UI language", () => {
  it("follows the selected UI language even when its bundle has no format keys", async () => {
    const english = i18n.t("format.number", { val: 9123456.5 });
    await useLanguage("de");
    const german = i18n.t("format.number", { val: 9123456.5 });
    expect(german).toBe(new Intl.NumberFormat("de").format(9123456.5));
    expect(german).not.toBe(english);
    expect(i18n.t("format.relativeTime", { val: -3 })).toBe(new Intl.RelativeTimeFormat("de").format(-3, "day"));
  });

  it.each([
    ["de", "EUR"],
    ["zh-CN", "CNY"],
    ["ja", "JPY"],
    ["fr", "EUR"],
    ["en-GB", "GBP"],
  ])("keeps money in US dollars in %s", async (language, localCurrency) => {
    await useLanguage(language);
    const shown = i18n.t("format.usd", { val: 1234.5, currency: localCurrency });
    expect(shown).toBe(new Intl.NumberFormat(language, { style: "currency", currency: "USD" }).format(1234.5));
    expect(shown).not.toBe(
      new Intl.NumberFormat(language, { style: "currency", currency: localCurrency }).format(1234.5),
    );
  });
});
