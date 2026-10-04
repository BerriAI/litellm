import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../tests/test-utils";
import traceList from "./__fixtures__/trace_list.json";
import { AgentTracesTable } from "./AgentTracesTable";
import type { TracePage, TraceSummary } from "./traceTypes";

const renderEmpty = (rangeEmpty: boolean) => {
  const onSetUpTracing = vi.fn();
  render(
    <AgentTracesTable
      traces={[]}
      isLoading={false}
      error={null}
      hasMore={false}
      onLoadMore={vi.fn()}
      onOpenTrace={vi.fn()}
      rangeEmpty={rangeEmpty}
      onSetUpTracing={onSetUpTracing}
    />,
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

describe("AgentTracesTable virtualization", () => {
  const template = (traceList as TracePage).data[0] as TraceSummary;
  const manyRuns: TraceSummary[] = Array.from({ length: 500 }, (_, i) => ({
    ...template,
    trace_id: `trace-${i}`,
    input_preview: `question ${i}`,
  }));

  it("renders only the rows near the viewport as the list scrolls", () => {
    renderWithProviders(
      <AgentTracesTable
        traces={manyRuns}
        isLoading={false}
        error={null}
        hasMore={false}
        onLoadMore={vi.fn()}
        onOpenTrace={vi.fn()}
        onSetUpTracing={vi.fn()}
      />,
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
