import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { I18nProvider } from "@/i18n/I18nProvider";
import { LANGUAGE_STORAGE_KEY } from "@/i18n/language";
import LanguageSelector from "./LanguageSelector";

describe("LanguageSelector", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "en");
    vi.restoreAllMocks();
  });

  it("switches from English to Russian immediately and persists the choice", async () => {
    const user = userEvent.setup();
    render(
      <I18nProvider>
        <LanguageSelector />
      </I18nProvider>,
    );

    const trigger = await screen.findByRole("button", { name: "Language: English" });
    expect(trigger).toHaveTextContent("EN");
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Русский" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Язык: Русский" })).toHaveTextContent("RU"));
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("ru");
    expect(document.documentElement.lang).toBe("ru");
  });
});
