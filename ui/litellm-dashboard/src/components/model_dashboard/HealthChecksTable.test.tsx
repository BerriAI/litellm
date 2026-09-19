/* @vitest-environment jsdom */
import type { PaginationState, RowSelectionState } from "@tanstack/react-table";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { useState } from "react";
import { describe, expect, it, type Mock, vi } from "vitest";

import { renderWithProviders, screen } from "../../../tests/test-utils";

import { HealthChecksTable } from "./HealthChecksTable";
import type { HealthCheckData, HealthStatus } from "./HealthChecksTableColumns";

const makeRow = (overrides: Partial<HealthCheckData> & { id: string }): HealthCheckData => {
  const { id, ...rest } = overrides;
  return {
    model_name: `model-${id}`,
    model_info: { id },
    health_status: "none",
    last_check: "None",
    last_success: "None",
    health_loading: false,
    ...rest,
  };
};

interface HarnessProps {
  data: HealthCheckData[];
  modelHealthStatuses?: Record<string, HealthStatus>;
  onRunHealthCheck?: (modelId: string) => void;
  onSelectModel?: (modelId: string) => void;
}

function Harness({ data, modelHealthStatuses = {}, onRunHealthCheck = vi.fn(), onSelectModel }: HarnessProps) {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  return (
    <HealthChecksTable
      data={data}
      rowCount={data.length}
      isLoading={false}
      pagination={pagination}
      onPaginationChange={setPagination}
      rowSelection={rowSelection}
      onRowSelectionChange={setRowSelection}
      modelHealthStatuses={modelHealthStatuses}
      getDisplayModelName={(model) => model.model_name}
      onRunHealthCheck={onRunHealthCheck}
      onShowError={vi.fn()}
      onShowSuccess={vi.fn()}
      onSelectModel={onSelectModel}
    />
  );
}

/** Row order by model id, read off the per-row selection checkbox (keyed by getRowId). */
const rowIds = (): string[] =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => row.querySelector('[data-testid^="datatable-select-row-"]'))
    .filter((node): node is Element => node !== null)
    .map((node) => (node.getAttribute("data-testid") ?? "").replace("datatable-select-row-", ""));

describe("HealthChecksTable client sorting", () => {
  it("orders health status healthy > checking > unknown > unhealthy", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Harness
        data={[
          makeRow({ id: "unhealthy-row", health_status: "unhealthy" }),
          makeRow({ id: "healthy-row", health_status: "healthy" }),
          makeRow({ id: "weird-row", health_status: "some-other-status" }),
          makeRow({ id: "checking-row", health_status: "checking" }),
        ]}
      />,
    );

    await user.click(screen.getByTestId("sort-header-health_status"));

    expect(rowIds()).toEqual(["healthy-row", "checking-row", "unhealthy-row", "weird-row"]);
  });

  it("floats in-progress checks to the top, sinks never-checked, and sorts real checks most-recent-first", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Harness
        data={[
          makeRow({ id: "never", last_check: "Never checked" }),
          makeRow({ id: "older", last_check: "2024-01-01T10:00:00Z" }),
          makeRow({ id: "in-progress", last_check: "Check in progress..." }),
          makeRow({ id: "newer", last_check: "2024-06-01T10:00:00Z" }),
        ]}
      />,
    );

    await user.click(screen.getByTestId("sort-header-last_check"));

    expect(rowIds()).toEqual(["in-progress", "newer", "older", "never"]);
  });

  // "Never succeeded" is ranked below "None" -- both sink, but not to the same slot.
  it("sinks None below real successes and Never succeeded below None", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Harness
        data={[
          makeRow({ id: "never", last_success: "Never succeeded" }),
          makeRow({ id: "none", last_success: "None" }),
          makeRow({ id: "older", last_success: "2024-01-01T10:00:00Z" }),
          makeRow({ id: "newer", last_success: "2024-06-01T10:00:00Z" }),
        ]}
      />,
    );

    await user.click(screen.getByTestId("sort-header-last_success"));

    expect(rowIds()).toEqual(["newer", "older", "none", "never"]);
  });
});

const statusRows = [
  makeRow({ id: "unhealthy-row", health_status: "unhealthy" }),
  makeRow({ id: "healthy-row", health_status: "healthy" }),
  makeRow({ id: "weird-row", health_status: "some-other-status" }),
  makeRow({ id: "checking-row", health_status: "checking" }),
];

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>): URLSearchParams => {
  const lastCall = onUrlUpdate.mock.calls.at(-1);
  if (!lastCall) throw new Error("expected a URL update");
  return lastCall[0].searchParams;
};

describe("HealthChecksTable URL sort state", () => {
  it("keeps the fetched row order when the URL names no sort column", () => {
    renderWithProviders(<Harness data={statusRows} />);

    expect(rowIds()).toEqual(["unhealthy-row", "healthy-row", "weird-row", "checking-row"]);
  });

  it("sorts by health_sort_by and health_sort_order from the URL", () => {
    renderWithProviders(<Harness data={statusRows} />, {
      searchParams: "?health_sort_by=health_status&health_sort_order=desc",
    });

    expect(rowIds()).toEqual(["weird-row", "unhealthy-row", "checking-row", "healthy-row"]);
  });

  it("writes header clicks to the URL and toggles direction from the URL value", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<Harness data={statusRows} />, { onUrlUpdate });

    await user.click(screen.getByTestId("sort-header-health_status"));

    expect(lastUrl(onUrlUpdate).get("health_sort_by")).toBe("health_status");
    expect(lastUrl(onUrlUpdate).get("health_sort_order")).toBeNull();
    expect(rowIds()).toEqual(["healthy-row", "checking-row", "unhealthy-row", "weird-row"]);

    await user.click(screen.getByTestId("sort-header-health_status"));

    expect(lastUrl(onUrlUpdate).get("health_sort_by")).toBe("health_status");
    expect(lastUrl(onUrlUpdate).get("health_sort_order")).toBe("desc");
    expect(rowIds()).toEqual(["weird-row", "unhealthy-row", "checking-row", "healthy-row"]);
  });

  it("keeps the fetched page when sorting, since the sort only reorders that page", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<Harness data={statusRows} />, {
      searchParams: "?health_page=3&health_page_size=25",
      onUrlUpdate,
    });

    await user.click(screen.getByTestId("sort-header-model_name"));

    expect(lastUrl(onUrlUpdate).get("health_sort_by")).toBe("model_name");
    expect(lastUrl(onUrlUpdate).get("health_page")).toBe("3");
    expect(lastUrl(onUrlUpdate).get("health_page_size")).toBe("25");
    expect(rowIds()).toEqual(["checking-row", "healthy-row", "unhealthy-row", "weird-row"]);
  });

  const sortableRow = (suffix: string, healthStatus: string, checkedAt: string): HealthCheckData => {
    const fields = {
      id: `row-${suffix}`,
      model_name: `model-${suffix}`,
      model_info: { id: `row-${suffix}`, team_id: `team-${suffix}` },
      health_status: healthStatus,
      last_check: checkedAt,
      last_success: checkedAt,
    };
    return makeRow(fields);
  };
  const newerRow = sortableRow("a", "healthy", "2024-06-01T10:00:00Z");
  const olderRow = sortableRow("b", "unhealthy", "2024-01-01T10:00:00Z");

  it.each(["model_id", "model_name", "team_id", "health_status", "last_check", "last_success"])(
    "sorts by %s when the URL names it",
    (column) => {
      renderWithProviders(<Harness data={[olderRow, newerRow]} />, { searchParams: `?health_sort_by=${column}` });

      expect(rowIds()).toEqual(["row-a", "row-b"]);
    },
  );
});

describe("HealthChecksTable rows", () => {
  it("renders the live checking cell while a row is loading and disables its run button", () => {
    renderWithProviders(<Harness data={[makeRow({ id: "busy", health_loading: true, health_status: "checking" })]} />);

    expect(screen.getByText("Checking...")).toBeInTheDocument();
    expect(screen.getByTestId("run-health-check-btn")).toBeDisabled();
  });

  it("runs a health check for the row's model id", async () => {
    const user = userEvent.setup();
    const onRunHealthCheck = vi.fn();
    renderWithProviders(<Harness data={[makeRow({ id: "deployment-9" })]} onRunHealthCheck={onRunHealthCheck} />);

    await user.click(screen.getByTestId("run-health-check-btn"));

    expect(onRunHealthCheck).toHaveBeenCalledWith("deployment-9");
  });

  it("opens the model detail from the identity cell", async () => {
    const user = userEvent.setup();
    const onSelectModel = vi.fn();
    renderWithProviders(<Harness data={[makeRow({ id: "deployment-9" })]} onSelectModel={onSelectModel} />);

    await user.click(screen.getByRole("button", { name: /deployment-9/ }));

    expect(onSelectModel).toHaveBeenCalledWith("deployment-9");
  });

  it("surfaces the error detail button only when a fuller error exists", () => {
    const { rerender } = renderWithProviders(
      <Harness
        data={[makeRow({ id: "m1", health_status: "unhealthy" })]}
        modelHealthStatuses={{
          m1: { status: "unhealthy", lastCheck: "x", loading: false, error: "same", fullError: "same" },
        }}
      />,
    );
    expect(screen.queryByTestId("view-health-error-btn")).not.toBeInTheDocument();

    rerender(
      <Harness
        data={[makeRow({ id: "m1", health_status: "unhealthy" })]}
        modelHealthStatuses={{
          m1: { status: "unhealthy", lastCheck: "x", loading: false, error: "short", fullError: "the long form" },
        }}
      />,
    );
    expect(screen.getByTestId("view-health-error-btn")).toBeInTheDocument();
  });

  it("renders the empty state when the page has no models", () => {
    renderWithProviders(<Harness data={[]} />);

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });
});
