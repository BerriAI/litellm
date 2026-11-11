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
});

describe("简体中文词库", () => {
  it("沿用英文键并保留插值变量和富文本组件", () => {
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

  it("通过官方语言注册加载中文，保留动态值并回退缺失词条", async () => {
    expect(supportedLanguages(LOCALES)).toContain("zh-CN");
    expect(resolveLanguage("zh-CN", LOCALES)).toBe("zh-CN");
    const load = LOCALES.get("zh-CN");
    if (!load) throw new Error("中文词库尚未注册");
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
    if (!formatter) throw new Error("i18next 格式化器尚未初始化");
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
    // 未覆盖的词条必须按官方英文回退，不能显示原始键。
    instance.removeResourceBundle("zh-CN", "translation");
    expect(instance.t("shell.sidebarAccountMenu.logout")).toBe(en.shell.sidebarAccountMenu.logout);
  });
});
