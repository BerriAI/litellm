import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import TaskBreakdown from "./TaskBreakdown";
import { taskColors, type TaskSummary } from "./taskInsightsData";

const task = (overrides: Partial<TaskSummary>): TaskSummary => ({
  task_id: "t",
  name: "Task",
  description: "",
  kind: "task",
  request_count: 0,
  delivery_count: 0,
  session_count: 0,
  share: 0,
  spend: null,
  cost_observations: 0,
  avg_cost_usd: null,
  map_x: null,
  map_y: null,
  ...overrides,
});

const tasks = [
  task({ task_id: "busy", name: "Busy cheap task", request_count: 80, share: 0.8, spend: 1, cost_observations: 80 }),
  task({ task_id: "costly", name: "Costly task", request_count: 15, share: 0.15, spend: 9, cost_observations: 12 }),
  task({ task_id: "unknown", name: "Unknown", kind: "unknown", request_count: 5, share: 0.05 }),
];

const rowNames = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0].textContent);

describe("TaskBreakdown", () => {
  it("ranks by request volume, then by spend when asked", async () => {
    renderWithProviders(
      <TaskBreakdown tasks={tasks} colors={taskColors(tasks)} selectedTaskId={null} onSelect={vi.fn()} />,
    );

    expect(rowNames()).toEqual(["Busy cheap task", "Costly task", "Unknown"]);

    await userEvent.click(screen.getByRole("radio", { name: "Spend" }));

    expect(rowNames()).toEqual(["Costly task", "Busy cheap task", "Unknown"]);
  });

  it("says how many requests the spend rests on and never shows missing spend as zero", () => {
    renderWithProviders(
      <TaskBreakdown tasks={tasks} colors={taskColors(tasks)} selectedTaskId={null} onSelect={vi.fn()} />,
    );

    const costly = screen.getByRole("row", { name: /Costly task/ });
    expect(costly).toHaveTextContent("Cost known for 12 of 15");
    const unknown = screen.getByRole("row", { name: /Unknown/ });
    expect(unknown).toHaveTextContent("No data");
    expect(unknown).not.toHaveTextContent("$0");
  });

  it("selects a task from its row", async () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <TaskBreakdown tasks={tasks} colors={taskColors(tasks)} selectedTaskId={null} onSelect={onSelect} />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Costly task" }));

    expect(onSelect).toHaveBeenCalledWith("costly");
  });
});
