import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
  tasks: [{ task_type: "code_generation", label: "Code Generation", category: "Code" }],
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
    expect(apiClient.get).toHaveBeenCalledWith("/model-insights", {
      accessToken: "token",
      query: { metric: "tokens" },
    });
  });

  it("refetches with the selected metric so top models are ranked by it", async () => {
    render(<ModelInsightsView accessToken="token" />);
    await screen.findByText("fast-chat");

    await userEvent.click(screen.getByRole("tab", { name: "requests" }));

    await waitFor(() =>
      expect(apiClient.get).toHaveBeenLastCalledWith("/model-insights", {
        accessToken: "token",
        query: { metric: "requests" },
      }),
    );
  });

  it("shows the API error instead of loading forever", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("Only proxy admins can view deployment-wide model insights"));
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("Could not load model insights")).toBeInTheDocument();
    expect(screen.getByText("Only proxy admins can view deployment-wide model insights")).toBeInTheDocument();
  });
});
