import { describe, expect, it } from "vitest";

import { getI18n } from "./i18n";

describe("getI18n", () => {
  it("returns a promise resolving to an initialised i18next instance", async () => {
    const i18n = await getI18n();
    expect(i18n.isInitialized).toBe(true);
  });

  it("returns the same singleton across calls", async () => {
    const first = await getI18n();
    const second = await getI18n();
    expect(first).toBe(second);
  });

  it("resolves enabled namespaces from the registry", async () => {
    const i18n = await getI18n();
    expect(i18n.t("common:languages.en")).toBe("English");
    expect(i18n.t("navigation:dashboard")).toBe("Dashboard");
  });

  it("switches to zh-CN and resolves Chinese resources", async () => {
    const i18n = await getI18n();
    await i18n.changeLanguage("zh-CN");
    expect(i18n.t("common:languages.zh-CN")).toBe("简体中文");
    expect(i18n.t("navigation:dashboard")).toBe("仪表盘");
    await i18n.changeLanguage("en");
  });
});
