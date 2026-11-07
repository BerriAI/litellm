import { render, screen } from "@testing-library/react";
import i18next, { type InitOptions } from "i18next";
import { useState } from "react";
import { I18nextProvider, initReactI18next, useTranslation } from "react-i18next";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LanguageGate } from "./I18nProvider";
import { Locales, Messages } from "./languages";
import en from "./locales/en.json";

const JAPANESE_LOGOUT = "ログアウト";
const japanese: Messages = { shell: { sidebarAccountMenu: { logout: JAPANESE_LOGOUT } } };

const options: InitOptions = {
  lng: "en",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
  resources: { en: { translation: en } },
  initAsync: false,
};

const createI18n = () => {
  const instance = i18next.createInstance();
  void instance.use(initReactI18next).init(options);
  return instance;
};

function AccountMenuLabels() {
  const { t, i18n } = useTranslation();
  const [labelReadOnce] = useState(() => i18n.t("shell.sidebarAccountMenu.logout"));
  return (
    <>
      <p data-testid="logout">{t("shell.sidebarAccountMenu.logout")}</p>
      <p data-testid="logout-read-once">{labelReadOnce}</p>
      <p data-testid="change-password">{t("shell.sidebarAccountMenu.changePassword")}</p>
    </>
  );
}

const renderGate = (stored: string | null, locales: Locales) => {
  const instance = createI18n();
  render(
    <I18nextProvider i18n={instance}>
      <LanguageGate i18n={instance} locales={locales} store={{ read: () => stored, write: vi.fn() }}>
        <AccountMenuLabels />
      </LanguageGate>
    </I18nextProvider>,
  );
};

afterEach(() => {
  document.documentElement.lang = "en";
  vi.restoreAllMocks();
});

describe("LanguageGate", () => {
  it("shows the stored language, with English for strings it hasn't translated", async () => {
    renderGate("ja", new Map([["ja", async () => japanese]]));

    expect(await screen.findByTestId("logout")).toHaveTextContent(JAPANESE_LOGOUT);
    expect(screen.getByTestId("logout-read-once")).toHaveTextContent(JAPANESE_LOGOUT);
    expect(screen.getByTestId("change-password")).toHaveTextContent(en.shell.sidebarAccountMenu.changePassword);
    expect(document.documentElement.lang).toBe("ja");
  });

  it.each([null, "en", "fr"])("shows English without loading anything when the stored value is %j", (stored) => {
    const load = vi.fn(async () => japanese);
    renderGate(stored, new Map([["ja", load]]));

    expect(screen.getByTestId("logout")).toHaveTextContent(en.shell.sidebarAccountMenu.logout);
    expect(load).not.toHaveBeenCalled();
    expect(document.documentElement.lang).toBe("en");
  });

  it("falls back to English when the translations fail to load", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    renderGate("ja", new Map([["ja", () => Promise.reject(new Error("offline"))]]));

    expect(await screen.findByTestId("logout")).toHaveTextContent(en.shell.sidebarAccountMenu.logout);
    expect(document.documentElement.lang).toBe("en");
  });
});
