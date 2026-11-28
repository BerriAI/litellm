import i18next from "i18next";
import { describe, expect, it } from "vitest";

import { registerFormats } from "../formats";
import { LOCALES, resolveLanguage, supportedLanguages, type Messages } from "../languages";

import en from "./en.json";
import zh from "./zh-TW.json";

function entries(messages: Messages, prefix = ""): readonly (readonly [string, string])[] {
  return Object.entries(messages).flatMap(([key, value]) => {
    const fullKey = prefix ? `${prefix}.${key}` : key;

    return typeof value === "string" ? [[fullKey, value] as const] : entries(value, fullKey);
  });
}

const markers = (value: string) => ({
  placeholders: [...value.matchAll(/\{\{([^}]+)\}\}/g)].map((match) => match[1]).sort(),
  tags: [...value.matchAll(/<\/?[\w-]+[^>]*>/g)].map((match) => match[0]).sort(),
  references: [...value.matchAll(/\$t\(([^)]+)\)/g)].map((match) => match[1]).sort(),
});

describe("zh-TW locale", () => {
  it("keeps placeholders, rich-text tags and $t references", () => {
    const english = new Map(entries(en));

    const incompatible = entries(zh).filter(([key, value]) => {
      const source = english.get(key);

      return (
        source === undefined ||
        value.trim() === "" ||
        JSON.stringify(markers(value)) !== JSON.stringify(markers(source))
      );
    });

    expect(entries(zh).length).toBeGreaterThan(0);
    expect(incompatible).toEqual([]);
  });

  it("loads through the language registry, keeps dynamic values and falls back to English", async () => {
    expect(supportedLanguages(LOCALES)).toContain("zh-TW");
    expect(resolveLanguage("zh-TW", LOCALES)).toBe("zh-TW");

    const load = LOCALES.get("zh-TW");

    if (!load) {
      throw new Error("zh-TW is not registered in LOCALES");
    }

    const messages = await load();
    const instance = i18next.createInstance();

    await instance.init({
      lng: "zh-TW",
      fallbackLng: "en",
      resources: {
        en: { translation: en },
        "zh-TW": { translation: messages },
      },
      interpolation: { escapeValue: false },
    });

    const formatter = instance.services.formatter;

    if (!formatter) {
      throw new Error("i18next formatter is not initialized");
    }

    registerFormats(formatter);

    expect(instance.t("shell.sidebarAccountMenu.language.label")).toBe("語言");

    const date = new Date("2026-11-11T04:03:02Z");

    const dateOptions: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    };

    expect(instance.t("format.date", { val: date, timeZone: "UTC" })).toBe(
      new Intl.DateTimeFormat("zh-TW", dateOptions).format(date),
    );

    expect(
      instance.t("shell.sidebarAccountMenu.triggerLabel", {
        role: "proxy_admin",
        identity: "unchanged-model-123",
      }),
    ).toContain("unchanged-model-123");

    instance.removeResourceBundle("zh-TW", "translation");

    expect(instance.t("shell.sidebarAccountMenu.logout")).toBe(en.shell.sidebarAccountMenu.logout);
  });
});
