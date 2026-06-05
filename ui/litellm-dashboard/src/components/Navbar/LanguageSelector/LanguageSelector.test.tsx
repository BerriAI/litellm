import i18n, { LANGUAGE_STORAGE_KEY } from "@/lib/i18n";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import LanguageSelector from "./LanguageSelector";

describe("LanguageSelector", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
    window.localStorage.clear();
  });

  it("lists all supported languages in the dropdown", async () => {
    render(<LanguageSelector />);

    await userEvent.click(screen.getByRole("button", { name: "Language" }));

    expect(await screen.findByText("English")).toBeInTheDocument();
    expect(screen.getByText("简体中文")).toBeInTheDocument();
  });

  it("switches the language and persists the choice", async () => {
    render(<LanguageSelector />);

    await userEvent.click(screen.getByRole("button", { name: "Language" }));
    await userEvent.click(await screen.findByText("简体中文"));

    await waitFor(() => expect(i18n.language).toBe("zh-CN"));
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("zh-CN");
    // The selector's own label re-renders in the new language
    expect(screen.getByRole("button", { name: "语言" })).toBeInTheDocument();
  });
});
