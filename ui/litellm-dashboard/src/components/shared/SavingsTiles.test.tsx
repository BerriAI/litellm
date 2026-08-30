import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import SavingsTiles from "./SavingsTiles";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

const metrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({
  spend: 6,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const day = (overrides: Partial<SpendMetrics>): DailyData => ({
  date: "2026-08-29",
  metrics: metrics(overrides),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    entities: {},
    api_keys: {},
  },
});

describe("SavingsTiles", () => {
  it("does not add caching and auto-router cards into one Total saved headline", () => {
    const overlappingDrivers: Partial<SpendMetrics> = {
      compression_savings_spend: 0,
      prompt_caching_savings_spend: 4,
      gateway_injected_caching_savings_spend: 4,
      autorouter_savings_spend: 6,
    };
    const { queryByTestId, getByTestId, queryByText } = render(
      <SavingsTiles isLoading={false} results={[day(overlappingDrivers)]} />,
    );

    expect(queryByTestId("summary-card-total-saved")).not.toBeInTheDocument();
    expect(queryByText("$10.00")).not.toBeInTheDocument();
    expect(getByTestId("summary-card-prompt-caching-savings")).toHaveTextContent("$4.00");
    expect(getByTestId("summary-card-auto-router-savings")).toHaveTextContent("$6.00");
  });
});
