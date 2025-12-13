import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ActivityMetrics } from "./activity_metrics";
import { ModelActivityData } from "./UsagePage/types";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("@tremor/react", () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Grid: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Text: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Title: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  AreaChart: () => <div>AreaChart</div>,
  BarChart: () => <div>BarChart</div>,
}));

vi.mock("antd", () => {
  const CollapseComponent = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  const PanelComponent = ({ children, header }: { children: React.ReactNode; header: React.ReactNode }) => (
    <div>
      <div>{header}</div>
      <div>{children}</div>
    </div>
  );
  PanelComponent.displayName = "Collapse.Panel";
  CollapseComponent.Panel = PanelComponent;
  return {
    Collapse: CollapseComponent,
  };
});

vi.mock("./UsagePage/utils/value_formatters", () => ({
  valueFormatter: (value: number) => value.toString(),
}));

vi.mock("./common_components/chartUtils", () => ({
  CustomTooltip: () => null,
  CustomLegend: () => <div>Legend</div>,
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (value: number, decimals?: number) => {
    return value.toFixed(decimals || 0);
  },
}));

describe("ActivityMetrics", () => {
  const mockModelMetrics: Record<string, ModelActivityData> = {
    "gpt-4": {
      label: "GPT-4",
      total_requests: 100,
      total_successful_requests: 95,
      total_failed_requests: 5,
      total_tokens: 50000,
      prompt_tokens: 30000,
      completion_tokens: 20000,
      total_spend: 100.5,
      total_cache_read_input_tokens: 1000,
      total_cache_creation_input_tokens: 500,
      top_api_keys: [],
      daily_data: [
        {
          date: "2025-01-01",
          metrics: {
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            spend: 100.5,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
        },
      ],
    },
  };

  it("should render", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
  });

  it("should display prompt caching metrics when hidePromptCachingMetrics is false", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={false} />);
    expect(screen.getByText("Prompt Caching Metrics")).toBeInTheDocument();
  });

  it("should hide prompt caching metrics when hidePromptCachingMetrics is true", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={true} />);
    expect(screen.queryByText("Prompt Caching Metrics")).not.toBeInTheDocument();
  });
});
