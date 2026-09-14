import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { PasswordInput } from "@/components/shared/PasswordInput";
import LanguageProvider from "./LanguageProvider";
import { LANGUAGE_STORAGE_KEY } from ".";

function LanguageExample() {
  return (
    <LanguageProvider>
      <LanguageSwitcher />
      <LoadingScreen />
      <PasswordInput aria-label="API key" />
    </LanguageProvider>
  );
}

async function selectChinese() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Language" }));
  await user.click(await screen.findByRole("menuitemradio", { name: "简体中文" }));
}

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("dashboard language preference", () => {
  it("switches all consumers, preserves password fields, and restores the explicit preference after remount", async () => {
    const user = userEvent.setup();
    const first = render(<LanguageExample />);
    await user.type(screen.getByLabelText("API key"), "opaque-value");
    await selectChinese();
    expect(screen.getByText("加载中…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "显示密码" })).toBeInTheDocument();
    expect(screen.getByLabelText("API key")).toHaveValue("opaque-value");
    expect(document.documentElement).toHaveAttribute("lang", "zh-CN");
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("zh-CN");
    first.unmount();
    render(<LanguageExample />);
    expect(await screen.findByRole("button", { name: "语言" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "语言" }));
    expect(await screen.findByRole("menuitemradio", { name: "简体中文" })).toBeChecked();
    await user.click(await screen.findByRole("menuitemradio", { name: "English" }));
    expect(screen.getByText("Loading...")).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "en");
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("en");
  });

  it("renders English on the server even when the browser has a Chinese preference", () => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "zh-CN");
    expect(renderToString(<LanguageExample />)).toContain("Loading...");
  });

  it("ignores invalid stored values without changing storage or the default UI", () => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "fr");
    render(<LanguageExample />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "en");
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("fr");
  });

  it("remains usable when browser storage throws on both reads and writes", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("Blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Blocked", "SecurityError");
    });
    render(<LanguageExample />);
    await selectChinese();
    expect(screen.getByText("加载中…")).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "zh-CN");
  });
});
