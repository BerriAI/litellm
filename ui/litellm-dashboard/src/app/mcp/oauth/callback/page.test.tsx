import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import McpOAuthCallbackPage from "./page";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("code=oauth-code&state=oauth-state"),
}));

vi.mock("@/utils/secureStorage", () => ({
  getSecureItem: vi.fn(() => null),
  setSecureItem: vi.fn(),
}));

vi.mock("@/components/LanguageSelector/LanguageSelector", () => ({
  default: () => <button aria-label="Language selector">RU</button>,
}));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources.ru.auth) ?? key,
    }),
  };
});

describe("MCP OAuth callback", () => {
  it("shows the localized completion state and language selector", () => {
    render(<McpOAuthCallbackPage />);

    expect(screen.getByText("Авторизация завершена")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Language selector" })).toBeInTheDocument();
  });
});
