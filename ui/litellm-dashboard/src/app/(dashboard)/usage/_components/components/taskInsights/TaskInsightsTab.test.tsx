import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import TaskInsightsTab from "./TaskInsightsTab";
import type { TaskInsightsOverview } from "./taskInsightsData";

const hooks = vi.hoisted(() => ({
  overview: vi.fn(),
  matrix: vi.fn(),
  task: vi.fn(),
  taxonomy: vi.fn(),
  discovery: { mutate: vi.fn(), isPending: false },
  freeze: { mutate: vi.fn(), isPending: false },
  edit: { mutate: vi.fn(), isPending: false },
}));

vi.mock("./useTaskInsights", () => ({
  useTaskInsightsOverview: hooks.overview,
  useTaskInsightsMatrix: hooks.matrix,
  useTaskInsightsTask: hooks.task,
  useTaskInsightsTaxonomy: hooks.taxonomy,
  useRequestDiscovery: () => hooks.discovery,
  useFreezeTaxonomy: () => hooks.freeze,
  useEditDraftTask: () => hooks.edit,
}));

vi.mock("./TaskClusterMap", () => ({ default: () => <div data-testid="task-map" /> }));

const idle = { data: undefined, isLoading: false, isError: false, error: null, refetch: vi.fn() };

const overview = (frozen: number, sample = false): TaskInsightsOverview => ({
  window: { start_date: "2026-11-04", end_date: "2026-11-11", team_id: null },
  last_refreshed_at: null,
  status: {
    discovery_status: "idle",
    discovery_error: null,
    discovery_requested_at: null,
    discovery_finished_at: null,
    taxonomy_frozen_at: null,
    last_bucketed_at: null,
    sample_data: sample,
    draft_task_count: 0,
    frozen_task_count: frozen,
  },
  totals: {
    request_count: 10,
    delivery_count: 12,
    session_count: 4,
    spend: 1.5,
    cost_observations: 9,
    unattributed_count: 0,
    classified_count: 8,
    unknown_count: 1,
    unclassified_count: 1,
    analysis_request_count: 0,
    analysis_spend: null,
  },
  tasks: [
    {
      task_id: "debug",
      name: "Debug failing tests",
      description: "",
      kind: "task",
      request_count: 8,
      delivery_count: 10,
      session_count: 3,
      share: 0.8,
      spend: 1.2,
      cost_observations: 8,
      avg_cost_usd: 0.15,
      map_x: 20,
      map_y: 30,
    },
  ],
  points: [],
});

const renderTab = (canEdit = true, searchParams = "?tab=tasks") => {
  const onUrlUpdate = vi.fn();
  renderWithProviders(
    <TaskInsightsTab startTime={new Date(2026, 10, 4)} endTime={new Date(2026, 10, 11)} teams={[]} canEdit={canEdit} active />,
    { searchParams, onUrlUpdate },
  );
  return onUrlUpdate;
};

beforeEach(() => {
  hooks.matrix.mockReturnValue(idle);
  hooks.task.mockReturnValue(idle);
  hooks.taxonomy.mockReturnValue(idle);
});

describe("TaskInsightsTab", () => {
  it("explains how to start when there is no frozen taxonomy, and offers discovery to admins", () => {
    hooks.overview.mockReturnValue({ ...idle, data: overview(0) });

    renderTab();

    expect(screen.getByText("No task taxonomy yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Discover tasks" })).toBeInTheDocument();
    expect(screen.queryByTestId("task-map")).not.toBeInTheDocument();
  });

  it("hides discovery from an admin viewer", () => {
    hooks.overview.mockReturnValue({ ...idle, data: overview(0) });

    renderTab(false);

    expect(screen.queryByRole("button", { name: "Discover tasks" })).not.toBeInTheDocument();
  });

  it("shows a retryable error when the overview fails", async () => {
    const refetch = vi.fn();
    hooks.overview.mockReturnValue({ ...idle, isError: true, error: new Error("boom"), refetch });

    renderTab();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(refetch).toHaveBeenCalled();
  });

  it("labels sample data and counts requests separately from ledger deliveries", () => {
    hooks.overview.mockReturnValue({ ...idle, data: overview(1, true) });

    renderTab();

    expect(screen.getAllByText("Sample data").length).toBeGreaterThan(0);
    expect(screen.getByText("12 ledger deliveries")).toBeInTheDocument();
    expect(screen.getByText("Task coverage").parentElement).toHaveTextContent("80%");
  });

  it("opens a task from the breakdown by writing it to the URL", async () => {
    hooks.overview.mockReturnValue({ ...idle, data: overview(1) });
    const onUrlUpdate = renderTab();

    await userEvent.click(screen.getByRole("button", { name: "Debug failing tests" }));

    expect(onUrlUpdate).toHaveBeenCalledWith(expect.objectContaining({ queryString: expect.stringContaining("ti_task=debug") }));
  });
});
