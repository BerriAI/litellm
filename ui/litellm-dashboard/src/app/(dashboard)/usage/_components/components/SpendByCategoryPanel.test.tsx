import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ColumnDef } from "@tanstack/react-table";

import { SpendByCategoryPanel } from "./SpendByCategoryPanel";

vi.mock("@/components/shared/chart_loader", () => ({
  ChartLoader: ({ isDateChanging }: { isDateChanging: boolean }) => (
    <div data-testid="chart-loader">{isDateChanging ? "Processing date selection..." : "Loading chart data..."}</div>
  ),
}));

interface TestRow extends Record<string, unknown> {
  key: string;
  spend: number;
}

const columns: ColumnDef<TestRow>[] = [{ header: "Key", accessorKey: "key" }];

const rows: TestRow[] = [{ key: "row-1", spend: 5 }];

describe("SpendByCategoryPanel", () => {
  it("displays the given title", () => {
    render(
      <SpendByCategoryPanel
        title="Spend by Widget"
        loading={false}
        isDateChanging={false}
        data={[]}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data"
      />,
    );
    expect(screen.getByText("Spend by Widget")).toBeInTheDocument();
  });

  it("shows the loader instead of chart content while loading", () => {
    render(
      <SpendByCategoryPanel
        title="Spend by Widget"
        loading
        isDateChanging={false}
        data={rows}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data"
      />,
    );
    expect(screen.getByTestId("chart-loader")).toBeInTheDocument();
    expect(screen.queryByText("row-1")).not.toBeInTheDocument();
  });

  it("renders the table rows once loaded", () => {
    render(
      <SpendByCategoryPanel
        title="Spend by Widget"
        loading={false}
        isDateChanging={false}
        data={rows}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data"
      />,
    );
    expect(screen.getAllByText("row-1").length).toBeGreaterThan(0);
  });

  it("shows the empty message when there is no data", () => {
    render(
      <SpendByCategoryPanel
        title="Spend by Widget"
        loading={false}
        isDateChanging={false}
        data={[]}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data for this range"
      />,
    );
    expect(screen.getByText("No data for this range")).toBeInTheDocument();
  });

  it("renders a header action only when one is given", () => {
    const { rerender } = render(
      <SpendByCategoryPanel
        title="Spend by Widget"
        loading={false}
        isDateChanging={false}
        data={[]}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data"
      />,
    );
    expect(screen.queryByText("Toggle")).not.toBeInTheDocument();

    rerender(
      <SpendByCategoryPanel
        title="Spend by Widget"
        headerAction={<button type="button">Toggle</button>}
        loading={false}
        isDateChanging={false}
        data={[]}
        indexKey="key"
        columns={columns}
        getRowId={(row) => row.key}
        noDataMessage="No data"
      />,
    );
    expect(screen.getByText("Toggle")).toBeInTheDocument();
  });
});
