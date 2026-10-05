import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import { Inspector } from "@/components/shared/Inspector";
import traceList from "../__fixtures__/trace_list.json";
import { AgentTracesTable } from "./AgentTracesTable";
import { traceKey } from "../routing";
import type { TracePage, TraceSummary } from "../types";

const inList = (table: React.ReactElement) => (
  <Inspector.Root
    items={[]}
    itemKey={traceKey}
    selected={null}
    onSelectedChange={vi.fn()}
    noun="trace"
    storageKey="test"
  >
    {table}
  </Inspector.Root>
);

const renderEmpty = (rangeEmpty: boolean) => {
  const onSetUpTracing = vi.fn();
  render(
    inList(
      <AgentTracesTable
        traces={[]}
        isLoading={false}
        error={null}
        hasMore={false}
        onLoadMore={vi.fn()}
        rangeEmpty={rangeEmpty}
        onSetUpTracing={onSetUpTracing}
      />,
    ),
  );
  return onSetUpTracing;
};

describe("AgentTracesTable empty state", () => {
  it.each([
    { rangeEmpty: true, message: "No runs in this time range" },
    { rangeEmpty: false, message: "No runs match these filters." },
  ])("offers tracing setup when rangeEmpty is $rangeEmpty", ({ rangeEmpty, message }) => {
    const onSetUpTracing = renderEmpty(rangeEmpty);
    expect(screen.getByText(message)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Set up tracing" }));
    expect(onSetUpTracing).toHaveBeenCalledOnce();
  });
});

describe("AgentTracesTable loading state", () => {
  it("fills the first page load with skeleton rows instead of an empty table", () => {
    render(
      inList(
        <AgentTracesTable
          traces={[]}
          isLoading
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          rangeEmpty
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
    expect(screen.getByRole("status")).toHaveTextContent("Loading runs…");
    const placeholders = screen.getAllByTestId("runs-placeholder");
    expect(placeholders.length).toBeGreaterThanOrEqual(8);
    const columnCount = screen.getAllByRole("columnheader").length;
    expect(within(placeholders[0]).getAllByRole("cell", { hidden: true })).toHaveLength(columnCount);
    expect(screen.queryByText(/No runs/)).not.toBeInTheDocument();
  });
});

describe("AgentTracesTable virtualization", () => {
  const template = (traceList as TracePage).data[0] as TraceSummary;
  const manyRuns: TraceSummary[] = Array.from({ length: 500 }, (_, i) => ({
    ...template,
    trace_id: `trace-${i}`,
    input_preview: `question ${i}`,
  }));

  it("renders only the rows near the viewport as the list scrolls", () => {
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={manyRuns}
          isLoading={false}
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
    expect(screen.getAllByTestId("agent-trace-row").length).toBeLessThan(50);
    expect(screen.getByText("question 0")).toBeInTheDocument();

    const scroller = screen.getByTestId("runs-table");
    scroller.scrollTop = 36 * 480;
    fireEvent.scroll(scroller);

    expect(screen.getAllByTestId("agent-trace-row").length).toBeLessThan(50);
    expect(screen.getByText("question 499")).toBeInTheDocument();
    expect(screen.queryByText("question 0")).not.toBeInTheDocument();
  });
});

describe("AgentTracesTable column picker", () => {
  const runs = (traceList as TracePage).data as TraceSummary[];
  const renderRuns = () =>
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={runs}
          isLoading={false}
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
  const headers = () => screen.getAllByRole("columnheader").map((header) => header.textContent);

  beforeEach(() => localStorage.clear());

  it("hides a column from the header picker and keeps it hidden after a remount", async () => {
    const user = userEvent.setup();
    const { unmount } = renderRuns();
    const before = headers().length;
    expect(headers()).toContain("Cost");

    await user.click(screen.getByRole("button", { name: "Columns" }));
    expect(screen.queryByTestId("view-option-time")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-option-agent")).not.toBeInTheDocument();
    expect(screen.queryByTestId("view-option-open")).not.toBeInTheDocument();
    await user.click(await screen.findByTestId("view-option-cost"));

    expect(headers()).not.toContain("Cost");
    expect(headers()).toHaveLength(before - 1);
    const [firstRow] = screen.getAllByTestId("agent-trace-row");
    expect(within(firstRow).getAllByRole("cell")).toHaveLength(before - 1);

    unmount();
    renderRuns();
    expect(headers()).not.toContain("Cost");
    expect(headers()).toHaveLength(before - 1);
  });

  it("shapes loading skeletons to the columns still visible", async () => {
    const user = userEvent.setup();
    const { unmount } = renderRuns();
    await user.click(screen.getByRole("button", { name: "Columns" }));
    await user.click(await screen.findByTestId("view-option-cost"));
    unmount();

    render(
      inList(
        <AgentTracesTable
          traces={[]}
          isLoading
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          rangeEmpty
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
    const columnCount = screen.getAllByRole("columnheader").length;
    const [placeholder] = screen.getAllByTestId("runs-placeholder");
    expect(within(placeholder).getAllByRole("cell", { hidden: true })).toHaveLength(columnCount);
  });
});
