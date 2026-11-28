import i18next from "i18next";
import { describe, expect, it } from "vitest";

import { registerFormats } from "../formats";
import { LOCALES, resolveLanguage, supportedLanguages, type Messages } from "../languages";
import en from "./en.json";
import ru from "./ru-RU.json";

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

describe("ru-RU locale", () => {
  it("only uses English keys and keeps their placeholders, rich-text tags and $t references", () => {
    const english = new Map(entries(en));
    const incompatible = entries(ru).filter(([key, value]) => {
      const pluralExtra = key.match(/^(.*)_(few|many)$/);
      const source = pluralExtra ? english.get(`${pluralExtra[1]}_one`) : english.get(key);
      return (
        source === undefined ||
        value.trim() === "" ||
        JSON.stringify(markers(value)) !== JSON.stringify(markers(source))
      );
    });
    expect(entries(ru).length).toBeGreaterThan(0);
    expect(incompatible).toEqual([]);
  });

  it("loads through the language registry, keeps dynamic values and falls back to English for missing keys", async () => {
    expect(supportedLanguages(LOCALES)).toContain("ru-RU");
    expect(resolveLanguage("ru-RU", LOCALES)).toBe("ru-RU");
    const load = LOCALES.get("ru-RU");
    if (!load) throw new Error("ru-RU is not registered in LOCALES");
    const messages = await load();
    const instance = i18next.createInstance();
    const options = {
      lng: "ru-RU",
      fallbackLng: "en",
      resources: { en: { translation: en }, "ru-RU": { translation: messages } },
      interpolation: { escapeValue: false },
    };
    await instance.init(options);
    const formatter = instance.services.formatter;
    if (!formatter) throw new Error("i18next formatter is not initialized");
    registerFormats(formatter);
    expect(instance.t("shell.sidebarAccountMenu.language.label")).toBe("Язык");
    const date = new Date("2026-11-11T04:03:02Z");
    const dateOptions: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    };
    expect(instance.t("format.date", { val: date, timeZone: "UTC" })).toBe(
      new Intl.DateTimeFormat("ru-RU", dateOptions).format(date),
    );
    expect(
      instance.t("shell.sidebarAccountMenu.triggerLabel", { role: "proxy_admin", identity: "unchanged-model-123" }),
    ).toContain("unchanged-model-123");
    expect(instance.t("apps.list.count", { count: 2 })).toBe("2 приложения");
    instance.removeResourceBundle("ru-RU", "translation");
    expect(instance.t("shell.sidebarAccountMenu.logout")).toBe(en.shell.sidebarAccountMenu.logout);
  });

  it("resolves all four Russian plural forms for count", async () => {
    const instance = i18next.createInstance();
    const options = {
      lng: "ru-RU",
      fallbackLng: "en",
      resources: { en: { translation: en }, "ru-RU": { translation: ru } },
      interpolation: { escapeValue: false },
    };
    await instance.init(options);
    const title = (count: number) => instance.t("apiKeys.virtualKeysTable.bulkDelete.title", { count });
    expect(title(1)).toBe("Удалить 1 виртуальный ключ");
    expect(title(2)).toBe("Удалить 2 виртуальных ключа");
    expect(title(5)).toBe("Удалить 5 виртуальных ключей");
    expect(title(1.5)).toBe("Удалить 1.5 виртуальных ключей");
  });
});
