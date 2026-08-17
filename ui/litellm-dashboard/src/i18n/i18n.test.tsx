import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useTranslation } from "react-i18next";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import I18nProvider from "./I18nProvider";
import i18n, { detectInitialLanguage, languageStorageKey, normalizeLanguage } from "./i18n";

function TranslationProbe() {
  const { t } = useTranslation();
  return (
    <>
      <span>{t("login.title")}</span>
      <span>{t("playground.configuration.title")}</span>
      <span>{t("playground.chat.emptyState")}</span>
      <span>{t("playground.compare.addComparison")}</span>
      <span>{t("virtualKeys.subtitle")}</span>
      <span>{t("dataTable.rowsPerPage")}</span>
    </>
  );
}

describe("UI internationalization", () => {
  afterEach(async () => {
    cleanup();
    localStorage.clear();
    document.documentElement.lang = "en";
    await act(async () => {
      await i18n.changeLanguage("en");
    });
  });

  it("normalizes Chinese browser locales and defaults other locales to English", () => {
    expect(normalizeLanguage("zh-CN")).toBe("zh-CN");
    expect(normalizeLanguage("zh-TW")).toBe("zh-CN");
    expect(normalizeLanguage("en-US")).toBe("en");
  });

  it("prefers a saved choice over the browser locale", () => {
    expect(detectInitialLanguage("en", ["zh-CN"])).toBe("en");
    expect(detectInitialLanguage("zh-CN", ["en-US"])).toBe("zh-CN");
    expect(detectInitialLanguage(null, ["zh-CN", "en-US"])).toBe("zh-CN");
  });

  it("restores Chinese and persists a language switch back to English", async () => {
    localStorage.setItem(languageStorageKey, "zh-CN");
    render(
      <I18nProvider>
        <TranslationProbe />
        <LanguageSwitcher />
      </I18nProvider>,
    );

    await waitFor(() => expect(screen.getByText("登录")).toBeInTheDocument());
    expect(screen.getByText("配置")).toBeInTheDocument();
    expect(screen.getByText("开始对话、生成图像或处理音频")).toBeInTheDocument();
    expect(screen.getByText("添加对比项")).toBeInTheDocument();
    expect(screen.getByText("用于验证网关请求的全部密钥。")).toBeInTheDocument();
    expect(screen.getByText("每页行数")).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("zh-CN");

    fireEvent.click(screen.getByRole("button", { name: "语言" }));
    fireEvent.click(await screen.findByText("English"));

    await waitFor(() => expect(screen.getByText("Login")).toBeInTheDocument());
    expect(localStorage.getItem(languageStorageKey)).toBe("en");
    expect(document.documentElement.lang).toBe("en");
  });
});
