import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import userEvent from "@testing-library/user-event";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { TooltipProvider } from "@/components/ui/tooltip";
import ResponseMetrics, { type TokenUsage } from "./ResponseMetrics";

const baseUsage: TokenUsage = { promptTokens: 5000, completionTokens: 12, totalTokens: 5012 };

describe("ResponseMetrics prompt cache chips", () => {
  it.each([
    { label: "输入: 5000", tooltip: "输入 Token" },
    { label: "输出: 12", tooltip: "输出 Token" },
    { label: "推理: 3", tooltip: "推理 Token" },
    { label: "工具: Home", tooltip: "使用的工具" },
    {
      label: "缓存读取: 4",
      tooltip: "从模型提供商（例如 Anthropic 或 OpenAI）的提示词缓存中读取的输入 Token，按折扣费率计费，由提供商上报。",
    },
    {
      label: "缓存写入: 5",
      tooltip: "写入模型提供商提示词缓存、供后续请求复用的输入 Token。",
    },
    {
      label: "响应缓存: 命中",
      tooltip: "此响应来自 LiteLLM 的响应缓存。请求未发送到模型提供商，因此没有读取或写入提供商的提示词缓存。",
      responseCache: true,
    },
  ])("中文显示 $label 的说明，工具名称不翻译", async ({ label, tooltip, responseCache }) => {
    localStorage.clear();
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="response-metrics-language" storageScope="response-metrics-test">
        <TooltipProvider>
          <ResponseMetrics
            usage={{
              ...baseUsage,
              reasoningTokens: 3,
              cacheReadTokens: 4,
              cacheCreationTokens: 5,
              servedFromResponseCache: responseCache,
            }}
            toolName="Home"
          />
        </TooltipProvider>
      </LanguageProvider>,
    );

    expect(screen.getByText("工具: Home")).toBeInTheDocument();
    await user.hover(screen.getByLabelText(label));
    expect(await screen.findByText(tooltip)).toBeVisible();
    if (responseCache) {
      expect(screen.queryByText(/缓存读取|缓存写入/)).not.toBeInTheDocument();
    }
  });

  it("renders both cache chips when the provider reports reads and writes", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 4695, cacheCreationTokens: 1234 }} />);

    expect(screen.getByText("Cache Read: 4695")).toBeInTheDocument();
    expect(screen.getByText("Cache Write: 1234")).toBeInTheDocument();
  });

  it("renders only the read chip when the provider reports reads alone", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 4695 }} />);

    expect(screen.getByText("Cache Read: 4695")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("renders no cache chips for a provider that reports no cache fields", () => {
    render(<ResponseMetrics usage={baseUsage} />);

    expect(screen.getByText("In: 5000")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("renders no cache chips when the provider reports zero cache tokens", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 0, cacheCreationTokens: 0 }} />);

    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("shows the response cache indicator instead of the provider cache chips on a response-cache hit", () => {
    render(
      <ResponseMetrics
        usage={{ ...baseUsage, cacheReadTokens: 4695, cacheCreationTokens: 1234, servedFromResponseCache: true }}
      />,
    );

    expect(screen.getByText("Response Cache: Hit")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("does not show the response cache indicator when the flag is absent", () => {
    render(<ResponseMetrics usage={baseUsage} />);

    expect(screen.queryByText(/Response Cache/)).not.toBeInTheDocument();
  });

  it("does not render the Cost chip when a persisted cost is null", () => {
    render(<ResponseMetrics usage={{ promptTokens: 1, cost: null as unknown as number }} />);

    expect(screen.queryByText(/Cost:/)).not.toBeInTheDocument();
    expect(screen.getByText("In: 1")).toBeInTheDocument();
  });

  it("does not render the Cost chip for NaN", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cost: Number.NaN }} />);

    expect(screen.queryByText(/Cost:/)).not.toBeInTheDocument();
  });

  it("renders the Cost chip for a finite cost", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cost: 0.000063 }} />);

    expect(screen.getByText("Cost: $0.000063")).toBeInTheDocument();
  });
});
