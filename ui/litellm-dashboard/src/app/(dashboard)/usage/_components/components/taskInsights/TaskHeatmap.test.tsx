import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import TaskHeatmap from "./TaskHeatmap";
import type { TaskMatrix } from "./taskInsightsData";

const matrix: TaskMatrix = {
  measurement_window: { start_date: "2026-11-01", end_date: "2026-11-11", team_id: null },
  last_refreshed_at: null,
  tasks: [
    { task_id: "debug", name: "Debug failing tests", kind: "task", required_capabilities: ["function_calling"] },
    { task_id: "explain", name: "Explain code", kind: "task", required_capabilities: [] },
  ],
  model_configs: [
    {
      model_config_id: "opus",
      model_name: "claude-opus",
      provider_model: "anthropic/claude-opus",
      configured: true,
      max_input_tokens: null,
    },
    {
      model_config_id: "idle",
      model_name: "idle-model",
      provider_model: "anthropic/idle",
      configured: true,
      max_input_tokens: null,
    },
  ],
  entries: [
    {
      model_config_id: "opus",
      task_id: "debug",
      request_count: 30,
      avg_cost_usd: null,
      cost_observations: 0,
      p95_latency_ms: 4200,
      latency_observations: 28,
      eval_success_rate: null,
      eval_sample_count: 0,
    },
    {
      model_config_id: "opus",
      task_id: "explain",
      request_count: 10,
      avg_cost_usd: 0.004,
      cost_observations: 10,
      p95_latency_ms: 900,
      latency_observations: 10,
      eval_success_rate: null,
      eval_sample_count: 0,
    },
  ],
  compatibility: [{ model_config_id: "idle", task_id: "debug", status: "incompatible", missing: ["function_calling"] }],
  unattributed_request_count: 3,
};

const renderHeatmap = (overrides: Partial<ComponentProps<typeof TaskHeatmap>> = {}) => {
  const props = {
    matrix,
    metric: "share" as const,
    onMetricChange: vi.fn(),
    selectedTaskId: null,
    selectedModelId: null,
    onSelectCell: vi.fn(),
    ...overrides,
  };
  renderWithProviders(<TaskHeatmap {...props} />);
  return props;
};

describe("TaskHeatmap", () => {
  it("shows each deployment's request count with its share of the task", () => {
    renderHeatmap();

    const cell = screen.getByRole("button", { name: "Debug failing tests on claude-opus: 30" });
    expect(cell).toHaveTextContent("100% of task");
  });

  it("greys out a cell without cost evidence instead of showing a zero cost", () => {
    renderHeatmap({ metric: "cost" });

    expect(screen.getByRole("button", { name: "Debug failing tests on claude-opus: No data" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Explain code on claude-opus: $0.004" })).toHaveTextContent(
      "10 observed",
    );
  });

  it("keeps a configured deployment without traffic as an empty column and flags missing capabilities", () => {
    renderHeatmap();

    const idle = screen.getByRole("button", { name: "Debug failing tests on idle-model: No data" });
    expect(within(idle).getByLabelText("Model metadata reports it lacks function_calling")).toBeInTheDocument();
  });

  it("opens the cell's task and deployment when a cell is clicked", async () => {
    const props = renderHeatmap({ metric: "latency" });

    await userEvent.click(screen.getByRole("button", { name: "Debug failing tests on claude-opus: 4.2 s" }));

    expect(props.onSelectCell).toHaveBeenCalledWith("debug", "opus");
  });

  it("switches the metric from the selector", async () => {
    const props = renderHeatmap();

    await userEvent.click(screen.getByRole("radio", { name: "p95 latency" }));

    expect(props.onMetricChange).toHaveBeenCalledWith("latency");
  });

  it("says how many requests could not be tied to a deployment", () => {
    renderHeatmap();

    expect(screen.getByText(/3 requests could not be tied to a configured deployment/)).toBeInTheDocument();
  });
});
