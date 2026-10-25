import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import ViewUserSpend from "./view_user_spend";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: null, userRole: null, userId: null }),
}));
vi.mock("./networking", () => ({ modelAvailableCall: vi.fn() }));

describe("用户费用汉化", () => {
  beforeEach(() => localStorage.clear());

  it("汉化费用表头和预算后缀，切回英文保持原文", () => {
    const content = (language: "en" | "zh-CN") => (
      <LanguageProvider defaultLanguage={language} userId="spend-test" storageScope="spend-test">
        <ViewUserSpend userSpend={1234.5} userMaxBudget={2000} selectedTeam={null} />
      </LanguageProvider>
    );
    const { rerender } = renderWithProviders(content("zh-CN"));
    expect(screen.getByText("总费用")).toBeInTheDocument();
    expect(screen.getByText("预算上限")).toBeInTheDocument();
    expect(screen.getByText("上限 $2,000.0000")).toBeInTheDocument();
    rerender(content("en"));
    expect(screen.getByText("Total Spend")).toBeInTheDocument();
    expect(screen.getByText("$2,000.0000 limit")).toBeInTheDocument();
  });
});
