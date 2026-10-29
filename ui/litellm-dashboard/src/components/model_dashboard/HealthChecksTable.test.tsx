/* @vitest-environment jsdom */
import type { PaginationState, RowSelectionState } from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

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
    render(
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
    render(
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
    render(
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

describe("HealthChecksTable rows", () => {
  it("renders the live checking cell while a row is loading and disables its run button", () => {
    render(<Harness data={[makeRow({ id: "busy", health_loading: true, health_status: "checking" })]} />);

    expect(screen.getByText("Checking...")).toBeInTheDocument();
    expect(screen.getByTestId("run-health-check-btn")).toBeDisabled();
  });

  it("runs a health check for the row's model id", async () => {
    const user = userEvent.setup();
    const onRunHealthCheck = vi.fn();
    render(<Harness data={[makeRow({ id: "deployment-9" })]} onRunHealthCheck={onRunHealthCheck} />);

    await user.click(screen.getByTestId("run-health-check-btn"));

    expect(onRunHealthCheck).toHaveBeenCalledWith("deployment-9");
  });

  it("opens the model detail from the identity cell", async () => {
    const user = userEvent.setup();
    const onSelectModel = vi.fn();
    render(<Harness data={[makeRow({ id: "deployment-9" })]} onSelectModel={onSelectModel} />);

    await user.click(screen.getByRole("button", { name: /deployment-9/ }));

    expect(onSelectModel).toHaveBeenCalledWith("deployment-9");
  });

  it("surfaces the error detail button only when a fuller error exists", () => {
    const { rerender } = render(
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
    render(<Harness data={[]} />);

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });
});
