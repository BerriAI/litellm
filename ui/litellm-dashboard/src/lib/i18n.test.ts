import dayjs from "dayjs";
import { afterEach, describe, expect, it, vi } from "vitest";
import en from "@/locales/en.json";
import zhCN from "@/locales/zh-CN.json";
import i18n, { detectLanguage, LANGUAGE_STORAGE_KEY } from "./i18n";

const mockBrowserLanguage = (language: string) => {
  vi.spyOn(window.navigator, "language", "get").mockReturnValue(language);
};

// Must run before any test changes the language: this fresh test environment
// imported @/lib/i18n via setupTests, so the initial language's side effects
// must already be applied. Catches listeners registered after init(), which
// miss the languageChanged event that init() emits synchronously.
it("applies the detected language's side effects at module load", () => {
  expect(i18n.language).not.toBe("");
  expect(document.documentElement.lang).toBe(i18n.language);
});

describe("detectLanguage", () => {
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("prefers a supported language stored in localStorage", () => {
    mockBrowserLanguage("en-US");
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "zh-CN");
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("ignores an unsupported stored value and falls back to the browser language", () => {
    mockBrowserLanguage("zh-CN");
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "fr");
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("matches the browser language exactly", () => {
    mockBrowserLanguage("zh-CN");
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("matches by primary subtag when there is no exact match", () => {
    mockBrowserLanguage("zh-TW");
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("defaults to English for unsupported browser languages", () => {
    mockBrowserLanguage("fr-FR");
    expect(detectLanguage()).toBe("en");
  });
});

describe("languageChanged side effects", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("syncs the html lang attribute and dayjs locale", async () => {
    await i18n.changeLanguage("zh-CN");
    expect(document.documentElement.lang).toBe("zh-CN");
    expect(dayjs.locale()).toBe("zh-cn");

    await i18n.changeLanguage("en");
    expect(document.documentElement.lang).toBe("en");
    expect(dayjs.locale()).toBe("en");
  });

  it("translates keys after switching language", async () => {
    expect(i18n.t("nav.virtualKeys")).toBe("Virtual Keys");
    await i18n.changeLanguage("zh-CN");
    expect(i18n.t("nav.virtualKeys")).toBe("虚拟密钥");
  });
});

describe("locale files", () => {
  const flattenKeys = (obj: Record<string, unknown>, prefix = ""): string[] =>
    Object.entries(obj).flatMap(([key, value]) =>
      typeof value === "object" && value !== null
        ? flattenKeys(value as Record<string, unknown>, `${prefix}${key}.`)
        : [`${prefix}${key}`],
    );

  it("keeps en and zh-CN key sets identical, so fallbackLng never papers over a missing translation", () => {
    expect(flattenKeys(zhCN).sort()).toEqual(flattenKeys(en).sort());
  });
});
