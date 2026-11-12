import { afterEach, describe, expect, it } from "vitest";

import i18n from "./i18n";
import en from "./locales/en.json";
import zhCN from "./locales/zh-CN.json";
import { LOCALES } from "./languages";

type Messages = { readonly [key: string]: string | Messages };

const isGroup = (value: unknown): value is Messages =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const leaves = (node: Messages, prefix: string): ReadonlyArray<readonly [string, unknown]> =>
  Object.entries(node).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return isGroup(value) ? leaves(value, path) : [[path, value] as const];
  });

const pluralCounterpart = (key: string): string | null => {
  if (key.endsWith("_one")) return `${key.slice(0, -"_one".length)}_other`;
  if (key.endsWith("_other")) return `${key.slice(0, -"_other".length)}_one`;
  return null;
};

const entries = leaves(en, "");
const keys = new Set(entries.map(([key]) => key));
const english = new Map(entries);
const chinese = leaves(zhCN, "");
const placeholders = (text: unknown): readonly string[] =>
  [...String(text).matchAll(/\{\{[^}]+\}\}|<\/?[A-Za-z][^>]*>/g)].map(([token]) => token).sort();

afterEach(async () => {
  await i18n.changeLanguage("en");
  i18n.removeResourceBundle("zh-CN", "translation");
});

describe("en.json", () => {
  it("has a non-empty string at every leaf", () => {
    expect(entries.length).toBeGreaterThan(0);
    expect(entries.filter(([, value]) => typeof value !== "string" || value === "").map(([key]) => key)).toEqual([]);
  });

  it("only uses placeholders named like identifiers, with an optional lowercase format", () => {
    expect(entries.some(([, value]) => String(value).includes("{{"))).toBe(true);
    const malformed = entries.filter(([, value]) => {
      const text = String(value);
      const wellFormed = text.match(/\{\{[A-Za-z_][A-Za-z0-9_]*(?:, [a-z]+)?\}\}/g)?.length ?? 0;
      return text.split("{{").length - 1 !== wellFormed || text.split("}}").length - 1 !== wellFormed;
    });
    expect(malformed.map(([key]) => key)).toEqual([]);
  });

  it("pairs every _one key with an _other key and back", () => {
    expect([...keys].some((key) => pluralCounterpart(key) !== null)).toBe(true);
    const unpaired = [...keys].filter((key) => {
      const counterpart = pluralCounterpart(key);
      return counterpart !== null && !keys.has(counterpart);
    });
    expect(unpaired).toEqual([]);
  });

  it("is what the i18n instance resolves keys from", () => {
    expect(i18n.t("workflows.drawer.eventCount", { count: 2 })).toBe("2 events");
  });
});

describe("zh-CN.json", () => {
  it("uses non-empty messages for known English keys", () => {
    expect(chinese.length).toBeGreaterThan(0);
    expect(chinese.filter(([key, value]) => !keys.has(key) || typeof value !== "string" || value === "")).toEqual([]);
  });

  it("preserves the placeholders and components needed to render each translated message", () => {
    const mismatches = chinese.filter(
      ([key, value]) => JSON.stringify(placeholders(value)) !== JSON.stringify(placeholders(english.get(key))),
    );
    expect(mismatches.map(([key]) => key)).toEqual([]);
  });

  it("renders Chinese counts and interpolation while untranslated dashboard messages fall back to English", async () => {
    const messages = await LOCALES.get("zh-CN")?.();
    expect(messages).toBeDefined();
    i18n.addResourceBundle("zh-CN", "translation", messages);
    await i18n.changeLanguage("zh-CN");

    expect(i18n.t("shell.notifications.bell.triggerUnread", { unread: 3 })).toBe("通知，3 条未读");
    expect(i18n.t("shell.notifications.upgrade.description", { summary: "更新", current: "1.2.3" })).toBe(
      "更新。当前版本为 v1.2.3",
    );
    for (const count of [0, 1, 2, 5]) {
      expect(i18n.t("shell.notifications.release.newFeatures", { count })).toBe(`${count} 项新功能`);
    }
    expect(i18n.t("workflows.drawer.eventCount", { count: 2 })).toBe("2 events");
    expect(i18n.t("format.usd", { val: 1234.5 })).toBe(
      new Intl.NumberFormat("zh-CN", { style: "currency", currency: "USD", currencyDisplay: "symbol" }).format(1234.5),
    );
  });
});
