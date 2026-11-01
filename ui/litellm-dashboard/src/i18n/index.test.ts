import { describe, expect, it } from "vitest";
import { createDashboardI18n, isLanguage } from ".";
import en from "./locales/en.json";
import zhCN from "./locales/zh-CN.json";

describe("dashboard translations", () => {
  it("keeps complete navigation catalogs with matching interpolation fields and markup", () => {
    const englishLabels = Object.fromEntries(Object.entries(en).filter(([, value]) => typeof value === "string"));
    expect(Object.keys(zhCN).sort()).toEqual(Object.keys(englishLabels).sort());
    for (const key of Object.keys(zhCN) as (keyof typeof zhCN)[]) {
      expect(zhCN[key].trim(), key).not.toBe("");
      expect(zhCN[key].match(/{{\w+}}|<\/?\w+>/g) ?? [], key).toEqual(en[key].match(/{{\w+}}|<\/?\w+>/g) ?? []);
    }
  });

  it("preserves punctuation in natural-language keys and falls back to English", () => {
    const instance = createDashboardI18n("zh-CN");
    expect(instance.t("Access your LiteLLM Admin UI.")).toBe("登录 LiteLLM 管理界面。");
    instance.addResource("en", "translation", "English fallback.", "English fallback.");
    expect(instance.t("English fallback.", { defaultValue: "Missing fallback" })).toBe("English fallback.");
    expect(instance.t("An untranslated label", { defaultValue: "An untranslated label" })).toBe("An untranslated label");
  });

  it("keeps nested workflow messages and pluralization available alongside literal display keys", () => {
    const instance = createDashboardI18n("zh-CN");
    expect(instance.t("workflows.header.title")).toBe(en.workflows.header.title);
    expect(instance.t("workflows.drawer.eventCount", { count: 1 })).toBe("1 event");
    expect(instance.t("workflows.drawer.eventCount", { count: 2 })).toBe("2 events");
    expect(instance.t("login.credentialsHelp")).toBe(zhCN["login.credentialsHelp"]);
  });

  it("keeps server-rendered and standalone instances independent of another user's selection", async () => {
    const first = createDashboardI18n();
    const second = createDashboardI18n();
    await first.changeLanguage("zh-CN");
    expect(first.t("Login")).toBe("登录");
    expect(second.t("Login")).toBe("Login");
  });

  it("only accepts the supported explicit preferences", () => {
    expect(["en", "zh-CN"].every(isLanguage)).toBe(true);
    expect([null, undefined, "", "fr", "zh-TW", "__proto__"].some(isLanguage)).toBe(false);
  });
});
