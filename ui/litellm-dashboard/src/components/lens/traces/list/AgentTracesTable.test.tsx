import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import { Inspector } from "@/components/shared/Inspector";
import traceList from "../__fixtures__/trace_list.json";
import { AgentTracesTable, runCost } from "./AgentTracesTable";
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
        findings={new Map()}
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
          findings={new Map()}
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
          findings={new Map()}
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

describe("runCost", () => {
  it.each([
    { spend: 0.42, priced_calls: 20, llm_calls: 20, expected: { label: "$0.42", partial: null } },
    {
      spend: 0.38,
      priced_calls: 18,
      llm_calls: 20,
      expected: { label: "≥ $0.38", partial: { short: "18/20 priced", long: "18 of 20 calls priced" } },
    },
    { spend: null, priced_calls: 0, llm_calls: 20, expected: null },
    { spend: 0, priced_calls: 0, llm_calls: 0, expected: null },
  ])("prices $priced_calls of $llm_calls calls", ({ expected, ...summary }) => {
    expect(runCost(summary)).toEqual(expected);
  });
});

describe("AgentTracesTable cost cell", () => {
  const template = (traceList as TracePage).data[0] as TraceSummary;
  const costCell = (run: Partial<TraceSummary>) => {
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={[{ ...template, llm_calls: 20, ...run }]}
          findings={new Map()}
          isLoading={false}
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
    const costColumn = screen.getAllByRole("columnheader").findIndex((header) => header.textContent === "Cost");
    return within(screen.getByTestId("agent-trace-row")).getAllByRole("cell")[costColumn];
  };

  it("shows an exact cost when every call is priced", () => {
    expect(costCell({ spend: 0.42, priced_calls: 20 })).toHaveTextContent("$0.42");
  });

  it("marks a partial cost as a lower bound and says how many calls were priced", () => {
    const cell = costCell({ spend: 0.38, priced_calls: 18 });
    expect(cell).toHaveTextContent("≥ $0.38");
    expect(cell).toHaveTextContent("18/20 priced");
    expect(within(cell).getByTitle("18 of 20 calls priced")).toBeInTheDocument();
  });

  it("shows a dash when no call is priced", () => {
    expect(costCell({ spend: null, priced_calls: 0 })).toHaveTextContent("—");
  });
});

describe("AgentTracesTable input cell", () => {
  it("shows the whole preview on one line instead of only its first line", () => {
    const template = (traceList as TracePage).data[0] as TraceSummary;
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={[{ ...template, input_preview: "CURRENT USER REQUEST:\n\n- add feedback to Lens" }]}
          findings={new Map()}
          isLoading={false}
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          onSetUpTracing={vi.fn()}
        />,
      ),
    );
    expect(screen.getByText("CURRENT USER REQUEST: - add feedback to Lens")).toBeInTheDocument();
  });
});

describe("AgentTracesTable column picker", () => {
  const runs = (traceList as TracePage).data as TraceSummary[];
  const renderRuns = () =>
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={runs}
          findings={new Map()}
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
          findings={new Map()}
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

describe("AgentTracesTable signals", () => {
  const [flaggedRun, cleanRun, queuedRun] = ((traceList as TracePage).data as TraceSummary[]).slice(0, 3);
  const key = (run: TraceSummary) => run.trace_ref || run.trace_id;
  const result = (
    run: TraceSummary,
    status: "classified" | "unclassified",
    flags: { signal_id: string; name: string; score: number }[] = [],
  ) => ({
    status: "ready" as const,
    signals: {
      trace_id: run.trace_id,
      trace_ref: run.trace_ref ?? "",
      status,
      flags,
      model: "jev",
      classified_at: null,
    },
  });
  const renderTable = (showSignals: boolean, onSetUpSignals?: () => void) =>
    renderWithProviders(
      inList(
        <AgentTracesTable
          traces={[flaggedRun, cleanRun, queuedRun]}
          findings={new Map()}
          signals={
            new Map([
              [
                key(flaggedRun),
                result(flaggedRun, "classified", [
                  { signal_id: "user_frustration", name: "User frustration", score: 0.92 },
                  { signal_id: "repeated_request", name: "Repeated request", score: 0.71 },
                ]),
              ],
              [key(cleanRun), result(cleanRun, "classified")],
              [key(queuedRun), result(queuedRun, "unclassified")],
            ])
          }
          showSignals={showSignals}
          signalsColumn
          onSetUpSignals={onSetUpSignals}
          isLoading={false}
          error={null}
          hasMore={false}
          onLoadMore={vi.fn()}
          rangeEmpty={false}
          onSetUpTracing={vi.fn()}
        />,
      ),
    );

  it("flags matching runs in red and names every detected signal", () => {
    renderTable(true);
    const rows = screen.getAllByTestId("agent-trace-row");
    expect(rows.map((row) => row.hasAttribute("data-flagged"))).toEqual([true, false, false]);
    const flagged = within(rows[0]).getByRole("list", { name: "Signals" });
    expect(
      within(flagged)
        .getAllByRole("listitem")
        .map((item) => item.textContent),
    ).toEqual(["User frustration", "Repeated request"]);
    expect(flagged).toHaveAttribute("title", "Signals: User frustration (92%), Repeated request (71%)");
    expect(within(rows[1]).getByTitle("No signals detected")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Checking")).toBeInTheDocument();
  });

  it("keeps the signals column with a setup link until signals are configured", async () => {
    const onSetUpSignals = vi.fn();
    renderTable(false, onSetUpSignals);
    const header = screen.getByRole("columnheader", { name: /Signals/ });
    await userEvent.click(within(header).getByRole("button", { name: "Set up signals" }));
    expect(onSetUpSignals).toHaveBeenCalledOnce();
    expect(screen.getAllByTitle("Signals are not set up")).toHaveLength(3);
    expect(screen.getAllByTestId("agent-trace-row").some((row) => row.hasAttribute("data-flagged"))).toBe(false);
  });
});
