import i18next from "i18next";
import { describe, expect, it } from "vitest";

import { registerFormats } from "../formats";
import { LOCALES, resolveLanguage, supportedLanguages, type Messages } from "../languages";
import en from "./en.json";
import zh from "./zh-CN.json";

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

describe("zh-CN locale", () => {
  it("only uses English keys and keeps their placeholders, rich-text tags and $t references", () => {
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

  it("loads through the language registry, keeps dynamic values and falls back to English for missing keys", async () => {
    expect(supportedLanguages(LOCALES)).toContain("zh-CN");
    expect(resolveLanguage("zh-CN", LOCALES)).toBe("zh-CN");
    const load = LOCALES.get("zh-CN");
    if (!load) throw new Error("zh-CN is not registered in LOCALES");
    const messages = await load();
    const instance = i18next.createInstance();
    const options = {
      lng: "zh-CN",
      fallbackLng: "en",
      resources: { en: { translation: en }, "zh-CN": { translation: messages } },
      interpolation: { escapeValue: false },
    };
    await instance.init(options);
    const formatter = instance.services.formatter;
    if (!formatter) throw new Error("i18next formatter is not initialized");
    registerFormats(formatter);
    expect(instance.t("shell.sidebarAccountMenu.language.label")).toBe("语言");
    const date = new Date("2026-11-11T04:03:02Z");
    const dateOptions: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    };
    expect(instance.t("format.date", { val: date, timeZone: "UTC" })).toBe(
      new Intl.DateTimeFormat("zh-CN", dateOptions).format(date),
    );
    expect(
      instance.t("shell.sidebarAccountMenu.triggerLabel", { role: "proxy_admin", identity: "unchanged-model-123" }),
    ).toContain("unchanged-model-123");
    expect(instance.t("vectorStores.table.files.count", { count: 2 })).toBe("2 个文件");
    instance.removeResourceBundle("zh-CN", "translation");
    expect(instance.t("shell.sidebarAccountMenu.logout")).toBe(en.shell.sidebarAccountMenu.logout);
  });

  it("says every enabled server gets access and resolves the customer organization terms", async () => {
    const instance = i18next.createInstance();
    const options = {
      lng: "zh-CN",
      fallbackLng: "en",
      resources: { en: { translation: en }, "zh-CN": { translation: zh } },
      interpolation: { escapeValue: false },
    };
    await instance.init(options);
    expect(instance.t("chat.mcpAppsPanel.connectModeSubtitle")).toBe(
      "请关闭此连接不应使用的服务器。点击服务器可查看工具或登录。",
    );
    expect(instance.t("chat.connectFlowBanner.unscoped.description", { client: "Test Client" })).toBe(
      "Test Client 将获得所有已开启服务器的访问权限。请关闭不应使用的服务器，按需登录，然后点击“完成连接”。",
    );
    expect(instance.t("customerPortal.list.columns.customer")).toBe("客户组织");
    expect(instance.t("customerPortal.createDialog.submit")).toBe("创建客户");
  });
});
