import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, useDashboardLanguage } from "./I18nProvider";
import { LANGUAGE_STORAGE_KEY } from "./language";

const LanguageProbe = () => {
  const { language, setLanguage } = useDashboardLanguage();
  return (
    <div>
      <span>{language}</span>
      <button onClick={() => void setLanguage(language === "ru" ? "en" : "ru")}>change</button>
    </div>
  );
};

const renderProbe = () =>
  render(
    <I18nProvider>
      <LanguageProbe />
    </I18nProvider>,
  );

describe("I18nProvider", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.lang = "en";
    vi.restoreAllMocks();
  });

  it("selects Russian from the browser when no preference is saved", async () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("ru-RU");

    renderProbe();

    expect(await screen.findByText("ru")).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("ru");
  });

  it("uses a saved preference before the browser language", async () => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "en");
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("ru-RU");

    renderProbe();

    expect(await screen.findByText("en")).toBeInTheDocument();
  });

  it("changes the active language without reload and persists it", async () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("en-US");
    renderProbe();
    await screen.findByText("en");

    fireEvent.click(screen.getByRole("button", { name: "change" }));

    await waitFor(() => expect(screen.getByText("ru")).toBeInTheDocument());
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("ru");
    expect(document.documentElement.lang).toBe("ru");
  });

  it("keeps the session usable when preference storage throws", async () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("en-US");
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage unavailable");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage unavailable");
    });

    renderProbe();
    await screen.findByText("en");
    fireEvent.click(screen.getByRole("button", { name: "change" }));

    expect(await screen.findByText("ru")).toBeInTheDocument();
  });
});
