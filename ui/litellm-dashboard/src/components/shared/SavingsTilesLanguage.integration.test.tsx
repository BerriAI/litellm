import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import type { DailyData } from "@/components/UsagePage/types";
import SavingsTiles from "./SavingsTiles";

const results: DailyData[] = [
  {
    date: "2026-10-25",
    metrics: {
      spend: 10,
      prompt_tokens: 3000,
      completion_tokens: 500,
      total_tokens: 3500,
      api_requests: 4,
      successful_requests: 4,
      failed_requests: 0,
      cache_read_input_tokens: 1000,
      cache_creation_input_tokens: 500,
      compression_savings_spend: 1.5,
      prompt_caching_savings_spend: 2,
      gateway_injected_caching_savings_spend: 0.5,
      autorouter_savings_spend: 3,
      compression_saved_tokens: 1234,
    },
    breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities: {} },
  },
];

function renderLocalized(isLoading = false) {
  return render(
    <LanguageProvider defaultLanguage="zh-CN" userId="savings-language" storageScope="test">
      <SavingsTiles results={results} isLoading={isLoading} />
      <LanguageSelector />
    </LanguageProvider>,
  );
}

// 保留真实卡片和弹出说明，验证语言变化不会改变金额或 Token 数量。
describe("节省费用卡片语言", () => {
  beforeEach(() => localStorage.clear());

  it("翻译提示与次要标签，并支持切回英文且保持金额不变", () => {
    renderLocalized();
    expect(screen.getByText("压缩 + 提示缓存 + 自动路由")).toBeVisible();
    expect(screen.getByText("已压缩 1,234 个 Token")).toBeVisible();
    expect(screen.getByText("LiteLLM 注入")).toBeVisible();
    expect(screen.getByText("已记录估算小计")).toBeVisible();
    expect(screen.getByText("总计")).toBeVisible();
    expect(screen.getByTestId("summary-card-已记录的总节省费用")).toHaveTextContent("$5.00");
    const cachingCard = screen.getByTestId("summary-card-提示词缓存节省的费用");
    expect(within(cachingCard).getByText("$0.5000")).toBeVisible();
    expect(within(cachingCard).getByText("$2.00")).toBeVisible();

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByText("Compression + prompt caching + auto-router")).toBeVisible();
    expect(screen.getByText("1,234 tokens compressed")).toBeVisible();
    expect(screen.getByText("Recorded estimates subtotal")).toBeVisible();
    expect(screen.getByText("Total")).toBeVisible();
    expect(screen.getByTestId("summary-card-total-recorded-savings")).toHaveTextContent("$5.00");
  });

  it.each([
    {
      label: "已记录的总节省费用",
      englishLabel: "total recorded savings",
      chinese: /^旁边三张卡片所记录节省金额的总和/,
      english: /^The sum of recorded savings in the three tiles beside it\./,
    },
    {
      label: "压缩节省的费用",
      englishLabel: "compression savings",
      chinese: /^Headroom 在请求前移除的 Token/,
      english: /^Tokens Headroom removed before the call/,
    },
    {
      label: "提示词缓存节省的费用",
      englishLabel: "prompt caching savings",
      chinese: /^相比所有 Token 均按输入价格计费/,
      english: /^What caching saved against paying the input rate for every token/,
    },
    {
      label: "自动路由节省的费用",
      englishLabel: "auto-router savings",
      chinese: /^每个请求相对其路由器最高档模型基准的可用节省估算之和/,
      english: /^Sum of available per-request savings estimates/,
    },
  ])("翻译 $label 的真实弹出说明并支持英文回退", async ({ label, englishLabel, chinese, english }) => {
    renderLocalized();
    fireEvent.click(screen.getByRole("button", { name: `${label}的计算方式` }));
    const details = await screen.findByRole("dialog");
    expect(within(details).getByText(chinese)).toBeVisible();
    fireEvent.keyDown(document, { key: "Escape" });
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    fireEvent.click(screen.getByRole("button", { name: `How ${englishLabel} is calculated` }));
    expect(await screen.findByText(english)).toBeVisible();
  });

  it("翻译加载状态", () => {
    renderLocalized(true);
    expect(screen.getByText("加载中…")).toBeVisible();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
  });

  it("未挂载语言上下文时仍保持原有英文回退", () => {
    render(<SavingsTiles results={results} isLoading={false} />);
    expect(screen.getByText("Compression + prompt caching + auto-router")).toBeVisible();
    expect(screen.getByText("1,234 tokens compressed")).toBeVisible();
    expect(screen.getByText("LiteLLM injected")).toBeVisible();
    expect(screen.getByText("Total")).toBeVisible();
  });
});
