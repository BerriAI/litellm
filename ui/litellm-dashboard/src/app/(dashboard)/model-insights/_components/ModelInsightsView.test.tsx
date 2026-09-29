import { render, screen } from "@testing-library/react";
import type React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ModelInsightsView from "./ModelInsightsView";
import { apiClient } from "@/components/networking";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn() } }));
vi.mock("@/components/ui/chart", () => ({
  ChartContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ChartTooltip: () => null,
  ChartTooltipContent: () => null,
}));
vi.mock("recharts", () => ({
  Bar: () => null,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
  Treemap: () => null,
  XAxis: () => null,
  YAxis: () => null,
}));

const metrics = {
  model_group: "fast-chat",
  model: "openai/gpt-5.4-mini",
  provider: "openai",
  spend: 2.5,
  prompt_tokens: 1000,
  completion_tokens: 2000,
  requests: 12,
  successful_requests: 12,
  failed_requests: 0,
};

const response = {
  start_date: "2025-09-29",
  end_date: "2026-09-28",
  top_models: [metrics],
  daily: [{ ...metrics, date: "2026-09-28" }],
  by_task: [{ ...metrics, task_type: "code_generation" }],
};

describe("ModelInsightsView", () => {
  beforeEach(() => {
    vi.mocked(apiClient.get).mockResolvedValue(response);
  });

  it("shows the ranking with share and the task legend from the API response", async () => {
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("fast-chat")).toBeInTheDocument();
    expect(screen.getByText("by openai")).toBeInTheDocument();
    expect(screen.getByText("Code")).toBeInTheDocument();
    expect(screen.getAllByText("100.0%")).toHaveLength(2);
    expect(screen.getByRole("tab", { name: "tokens" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "log" })).toBeInTheDocument();
    expect(apiClient.get).toHaveBeenCalledWith("/model-insights", { accessToken: "token" });
  });
});
