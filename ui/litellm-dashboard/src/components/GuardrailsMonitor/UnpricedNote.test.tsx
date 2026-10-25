import { describe, expect, it } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { UnpricedNote } from "./UnpricedNote";
import { pricingIssueUrl } from "./usageUnits";

describe("未定价用量提示汉化", () => {
  it.each([1, 1000])("完整翻译 %i 个用量单位，不改变 GitHub 请求参数", (count) => {
    localStorage.clear();
    const unpriced = { customCounterUnits: count };
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="unpriced-language" storageScope="unpriced-language">
        <UnpricedNote unpriced={unpriced} provider="Home" />
      </LanguageProvider>,
    );

    expect(screen.getByText(`有 ${count.toLocaleString()} 个用量单位因价格未知而未计入费用。`)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "在 GitHub 请求添加定价" })).toHaveAttribute(
      "href",
      pricingIssueUrl(unpriced, "Home"),
    );
  });

  it.each([1, 2])("英文保留 %i 个用量单位的单复数", (count) => {
    renderWithProviders(<UnpricedNote unpriced={{ customCounterUnits: count }} />);
    const sentence =
      count === 1
        ? "1 unit with no known price is left out of the cost."
        : "2 units with no known price are left out of the cost.";
    expect(screen.getByText(sentence)).toBeInTheDocument();
  });

  it("零用量不显示提示", () => {
    const { container } = renderWithProviders(<UnpricedNote unpriced={{ customCounterUnits: 0 }} />);
    expect(container).toBeEmptyDOMElement();
  });
});
