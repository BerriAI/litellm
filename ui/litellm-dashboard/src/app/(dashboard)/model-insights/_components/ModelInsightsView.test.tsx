import { render, screen, waitFor } from "@testing-library/react";
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
  Area: () => null,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
}));

const response = {
  start_date: "2026-09-09",
  end_date: "2026-09-28",
  top_models: [
    {
      model_group: "fast-chat",
      model: "openai/gpt-5.4-mini",
      provider: "openai",
      spend: 2.5,
      prompt_tokens: 1000,
      completion_tokens: 2000,
      requests: 12,
      successful_requests: 12,
      failed_requests: 0,
    },
  ],
  daily: [
    {
      model_group: "fast-chat",
      model: "openai/gpt-5.4-mini",
      provider: "openai",
      date: "2026-09-28",
      spend: 2.5,
      prompt_tokens: 1000,
      completion_tokens: 2000,
      requests: 12,
      successful_requests: 12,
      failed_requests: 0,
    },
  ],
  by_task: [
    {
      model_group: "fast-chat",
      model: "openai/gpt-5.4-mini",
      provider: "openai",
      task_type: "chat",
      spend: 2.5,
      prompt_tokens: 1000,
      completion_tokens: 2000,
      requests: 12,
      successful_requests: 12,
      failed_requests: 0,
    },
  ],
};

describe("ModelInsightsView", () => {
  beforeEach(() => {
    vi.mocked(apiClient.get).mockResolvedValue(response);
  });

  it("renders actual model rankings, task usage, and provider logo", async () => {
    render(<ModelInsightsView accessToken="token" />);

    await waitFor(() => expect(screen.getAllByText("fast-chat").length).toBeGreaterThan(0));
    expect(screen.getByText("Top model usage")).toBeInTheDocument();
    expect(screen.getByText("Top models by task")).toBeInTheDocument();
    expect(screen.getByText("chat")).toBeInTheDocument();
    expect(screen.getByText("Cost per session")).toBeInTheDocument();
    expect(apiClient.get).toHaveBeenCalledWith("/model-insights", { accessToken: "token" });
  });
});
