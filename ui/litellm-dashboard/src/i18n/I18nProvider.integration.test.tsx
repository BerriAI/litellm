import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useTranslation } from "react-i18next";
import { useState } from "react";

import { I18nProvider, useI18n } from "@/i18n";

/** Reports the active locale and a translated value so tests can assert wiring. */
function Consumer() {
  const { locale, setLocale } = useI18n();
  const { t } = useTranslation();
  const [pressed, setPressed] = useState<number>(0);

  return (
    <div>
      <span data-testid="locale">{locale}</span>
      <span data-testid="translated">{t("common:languages.en")}</span>
      <button
        type="button"
        onClick={() => {
          setLocale("zh-CN");
          setPressed((n) => n + 1);
        }}
      >
        to-zh
      </button>
      <span data-testid="pressed">{pressed}</span>
    </div>
  );
}

const ORIGINAL_LANG = "en";

beforeEach(() => {
  // Force a clean preference + language for each test.
  localStorage.clear();
  document.documentElement.lang = ORIGINAL_LANG;
});

afterEach(() => {
  localStorage.clear();
  document.documentElement.lang = ORIGINAL_LANG;
});

describe("I18nProvider readiness gate", () => {
  it("renders the business subtree only once the instance is ready (no key flash)", async () => {
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    );

    // Children must render (gate opened) with a real translated value, not a raw key.
    const translated = await screen.findByTestId("translated");
    expect(translated).toHaveTextContent("English");
  });

  it("syncs document.documentElement.lang to the resolved locale (en)", async () => {
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    );
    await waitFor(() => expect(document.documentElement.lang).toBe("en"));
  });
});

describe("I18nProvider zh-CN preference", () => {
  it("resolves a stored zh-CN preference, renders Chinese, and syncs <html lang>", async () => {
    localStorage.setItem("litellm.locale", "zh-CN");
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("locale")).toHaveTextContent("zh-CN"));
    expect(document.documentElement.lang).toBe("zh-CN");
  });
});

describe("useI18n setLocale", () => {
  it("switches language, applies it, and updates the consumer", async () => {
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    );

    await screen.findByTestId("translated");
    fireEvent.click(screen.getByRole("button", { name: "to-zh" }));

    await waitFor(() => expect(screen.getByTestId("locale")).toHaveTextContent("zh-CN"));
    expect(document.documentElement.lang).toBe("zh-CN");
    expect(screen.getByTestId("pressed")).toHaveTextContent("1");
  });

  it("persists the explicit choice to the unified preferences key", async () => {
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    );
    await screen.findByTestId("translated");

    fireEvent.click(screen.getByRole("button", { name: "to-zh" }));
    await waitFor(() => expect(screen.getByTestId("locale")).toHaveTextContent("zh-CN"));

    expect(localStorage.getItem("litellm.locale")).toBe("zh-CN");
  });
});
