import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ModelInsightsView from "./ModelInsightsView";
import { apiClient } from "@/components/networking";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock("@/components/ui/chart", () => ({
  ChartContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ChartTooltip: () => null,
  ChartTooltipContent: () => null,
}));
vi.mock("recharts", () => ({
  Bar: () => null,
  BarChart: ({ children, data }: { children: React.ReactNode; data: { date: string }[] }) => (
    <div data-testid="usage-chart" data-first={data[0]?.date} data-buckets={data.length}>
      {children}
    </div>
  ),
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
  daily_totals: [{ date: "2026-09-28", spend: 2.5, prompt_tokens: 1000, completion_tokens: 2000, requests: 12 }],
};

const taskResponse = {
  start_date: "2025-09-29",
  end_date: "2026-09-28",
  tasks: [
    {
      task_type: "code_generation",
      label: "Code Generation",
      category: "Code",
      value: 2.5,
      share: 100,
      leader: "fast-chat",
      provider: "openai",
    },
  ],
};

const providers = [
  { provider: "jev", label: "Jev (TypeSafe)", models: ["jev-latest"], ready: true, missing_env: [] },
  { provider: "laya", label: "Laya", models: ["english"], ready: false, missing_env: ["LAYA_API_BASE"] },
];
const unconfigured = { configured: null, providers };
const configured = { configured: { provider: "jev", model: "jev-latest" }, providers };

const mockApi = (tasks: unknown = taskResponse, classifier: unknown = configured) =>
  vi.mocked(apiClient.get).mockImplementation((path: string) => {
    if (path === "/model-insights/tasks") return tasks as Promise<unknown>;
    if (path === "/model-insights/task-classifier") return Promise.resolve(classifier);
    return Promise.resolve(response);
  });

describe("ModelInsightsView", () => {
  beforeEach(() => {
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.put).mockReset();
    vi.mocked(apiClient.delete).mockReset();
    mockApi(Promise.resolve(taskResponse));
  });

  it("shows the ranking with share and the task legend from the API response", async () => {
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("fast-chat")).toBeInTheDocument();
    expect(screen.getByText("by openai")).toBeInTheDocument();
    expect(await screen.findByText("Code")).toBeInTheDocument();
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
      expect(apiClient.get).toHaveBeenCalledWith("/model-insights", {
        accessToken: "token",
        query: { metric: "requests" },
      }),
    );
  });

  it("does not refetch the task breakdown when the chart metric changes", async () => {
    render(<ModelInsightsView accessToken="token" />);
    await screen.findByText("Code");
    const taskCalls = () =>
      vi.mocked(apiClient.get).mock.calls.filter(([path]) => path === "/model-insights/tasks").length;
    const before = taskCalls();

    await userEvent.click(screen.getByRole("tab", { name: "requests" }));
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith("/model-insights", {
        accessToken: "token",
        query: { metric: "requests" },
      }),
    );

    expect(taskCalls()).toBe(before);
  });

  it("shows the API error instead of loading forever", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("Only proxy admins can view deployment-wide model insights"));
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("Could not load model insights")).toBeInTheDocument();
    expect(screen.getByText("Only proxy admins can view deployment-wide model insights")).toBeInTheDocument();
  });

  it("keeps the previous ranking, dimmed, until the new metric's data arrives", async () => {
    render(<ModelInsightsView accessToken="token" />);
    await screen.findByText("fast-chat");
    let resolve: (value: typeof response) => void = () => {};
    vi.mocked(apiClient.get).mockImplementation((path: string) =>
      path === "/model-insights/tasks"
        ? Promise.resolve(taskResponse)
        : new Promise((done) => (resolve = done as typeof resolve)),
    );

    await userEvent.click(screen.getByRole("tab", { name: "spend" }));

    expect(
      screen.getByText("Share of tokens, with the change between the first and second half of the period"),
    ).toBeInTheDocument();

    resolve(response);
    expect(
      await screen.findByText("Share of spend, with the change between the first and second half of the period"),
    ).toBeInTheDocument();
  });

  it("charts one bar per day by default and switches to weekly bars", async () => {
    render(<ModelInsightsView accessToken="token" />);
    await screen.findByText("fast-chat");
    const chart = screen.getByTestId("usage-chart");

    expect(screen.getByRole("tab", { name: "Daily" })).toHaveAttribute("aria-selected", "true");
    expect(chart).toHaveAttribute("data-buckets", "30");
    expect(chart).toHaveAttribute("data-first", "2026-08-30");
    expect(screen.getByText("Daily tokens across your gateway")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Weekly" }));

    expect(chart).toHaveAttribute("data-buckets", "12");
    expect(chart).toHaveAttribute("data-first", "2026-07-13");
    expect(screen.getByText("Weekly tokens across your gateway")).toBeInTheDocument();
  });

  it("tells the admin the task classifier is not set up and enables the first ready System One model", async () => {
    mockApi(Promise.resolve({ ...taskResponse, tasks: [] }), unconfigured);
    vi.mocked(apiClient.put).mockResolvedValue(configured);
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("Task classifier is not set up")).toBeInTheDocument();
    expect(screen.getByText("No task data yet")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Enable classifier" }));

    expect(apiClient.put).toHaveBeenCalledWith("/model-insights/task-classifier", {
      accessToken: "token",
      body: { provider: "jev", model: "jev-latest" },
    });
    expect(await screen.findByText("jev-latest")).toBeInTheDocument();
    expect(screen.queryByText("Task classifier is not set up")).not.toBeInTheDocument();
    expect(
      screen.getByText("No requests classified yet. New requests show up here after the next background batch"),
    ).toBeInTheDocument();
  });

  it("keeps the setup prompt and shows the server error when enabling fails", async () => {
    mockApi(Promise.resolve(taskResponse), unconfigured);
    vi.mocked(apiClient.put).mockRejectedValue(new Error("TYPESAFE_API_KEY is not set"));
    render(<ModelInsightsView accessToken="token" />);

    await userEvent.click(await screen.findByRole("button", { name: "Enable classifier" }));

    expect(await screen.findByText("TYPESAFE_API_KEY is not set")).toBeInTheDocument();
    expect(screen.getByText("Task classifier is not set up")).toBeInTheDocument();
  });

  it("shows the configured classifier and turns it off", async () => {
    vi.mocked(apiClient.delete).mockResolvedValue(unconfigured);
    render(<ModelInsightsView accessToken="token" />);

    expect(await screen.findByText("jev-latest")).toBeInTheDocument();
    expect(screen.queryByText("Task classifier is not set up")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Turn off" }));

    expect(apiClient.delete).toHaveBeenCalledWith("/model-insights/task-classifier", { accessToken: "token" });
    expect(await screen.findByText("Task classifier is not set up")).toBeInTheDocument();
  });
});
