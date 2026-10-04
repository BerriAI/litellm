import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { focusManager, onlineManager } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import researchTrace from "./__fixtures__/research_trace.json";
import swarmTrace from "./__fixtures__/swarm_trace.json";
import type { ComponentProps } from "react";
import { initialRunSelection, RunView } from "./TraceDrawer";
import { useOpenTraceRouting } from "./traceRouting";
import { agentHandoffText } from "./tracesApi";
import type { Span } from "./traceTypes";
import type { Trace } from "./traceTypes";
import { traceDisplayName } from "./traceUtils";

vi.mock("../../networking", () => ({
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test/",
}));

// DetailPane is built separately; render a stub that exposes which row is selected.
vi.mock("./DetailPane", () => ({
  DetailPane: ({ row, onClose }: { row?: { id: string }; onClose: () => void }) => (
    <div data-testid="detail-pane" data-row-id={row?.id}>
      <button type="button" onClick={onClose}>
        close detail
      </button>
    </div>
  ),
}));

vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceCall } from "../../networking";

const swarm = swarmTrace as Trace;
const research = researchTrace as Trace;

function RoutedRunView(props: Omit<ComponentProps<typeof RunView>, "selection">) {
  const { selection } = useOpenTraceRouting();
  return <RunView {...props} selection={selection} />;
}

const renderRun = (trace: Trace) => {
  vi.mocked(agentTraceCall).mockResolvedValue(trace);
  return renderWithProviders(<RoutedRunView traceId={trace.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} />);
};

const rootSpanId = (trace: Trace): string => trace.spans.find((s) => s.parent_span_id === null)?.span_id ?? "";

describe("RunView", () => {
  beforeEach(() => {
    testQueryClient.clear();
    testQueryClient.setQueryDefaults(["agentTrace"], {});
    vi.mocked(copyToClipboard).mockClear();
  });

  it("shows the run name, copyable ID and run totals", async () => {
    renderRun(research);

    const header = await screen.findByRole("banner");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(traceDisplayName(research.summary));
    expect(within(header).getByRole("button", { name: "Copy trace ID" })).toHaveAttribute(
      "title",
      research.summary.trace_id,
    );
    expect(header).toHaveTextContent("Duration 40.20s");
    expect(header).toHaveTextContent(`Steps ${research.summary.span_count}`);
    expect(header).not.toHaveTextContent("failed");
  });

  it("shows the agent name with the SDK logo in the run header instead of the generic agent icon", async () => {
    renderRun({
      ...research,
      summary: { ...research.summary, agent_names: ["research-bot"], frameworks: ["claude-agent-sdk", "claude-code"] },
    });

    const header = await screen.findByRole("banner");
    expect(within(header).getByTestId("run-framework")).toHaveTextContent(/^research-bot$/);
    expect(within(header).getByTestId("run-framework")).toHaveAttribute("title", "Claude Agent SDK");
    expect(within(header).getByRole("img", { name: "Claude Agent SDK logo", hidden: true })).toBeInTheDocument();
    expect(within(header).queryByTestId("span-icon")).not.toBeInTheDocument();
  });

  it("keeps the generic agent icon when the trace has no known SDK", async () => {
    renderRun({ ...research, summary: { ...research.summary, frameworks: ["some-other-sdk"] } });

    const header = await screen.findByRole("banner");
    expect(within(header).getByTestId("span-icon")).toBeInTheDocument();
    expect(within(header).queryByTestId("run-framework")).not.toBeInTheDocument();
  });

  it("folds researcher ×12 in the span tree", async () => {
    renderRun(swarm);

    const tree = await screen.findByRole("tree", { name: "Spans in time order" });
    expect(tree).toHaveTextContent("researcher×12");
    expect(screen.getByRole("banner")).toHaveTextContent(`Step errors ${swarm.summary.error_count}`);
  });

  it("opens a failed run on its first failed span", async () => {
    renderRun(swarm);

    const pane = await screen.findByTestId("detail-pane");
    const { selectedId } = initialRunSelection(swarm);
    expect(selectedId).not.toBe(rootSpanId(swarm));
    expect(pane).toHaveAttribute("data-row-id", selectedId);
    expect(swarm.spans.find((s) => s.span_id === selectedId)?.status).toBe("error");
  });

  it("opens a healthy run on the root span", async () => {
    renderRun(research);
    expect(await screen.findByTestId("detail-pane")).toHaveAttribute("data-row-id", rootSpanId(research));
  });

  it("inside the drawer moves spans with the arrow keys and leaves J / K to switch runs", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceCall).mockResolvedValue(research);
    renderWithProviders(
      <RoutedRunView traceId={research.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} embedded />,
    );

    const root = rootSpanId(research);
    expect(await screen.findByTestId("detail-pane")).toHaveAttribute("data-row-id", root);
    await user.keyboard("j");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", root);
    await user.keyboard("{ArrowDown}");
    expect(screen.getByTestId("detail-pane").getAttribute("data-row-id")).not.toBe(root);
    expect(screen.queryByRole("button", { name: "Back to runs" })).not.toBeInTheDocument();
  });

  it("keeps the current run on screen, inert, while an unvisited run loads in the drawer", async () => {
    const user = userEvent.setup();
    let resolveSwarm: (trace: Trace) => void = () => {};
    vi.mocked(agentTraceCall).mockImplementation((_token, traceId) =>
      traceId === swarm.summary.trace_id
        ? new Promise<Trace>((resolve) => {
            resolveSwarm = resolve;
          })
        : Promise.resolve(research),
    );
    const props = { accessToken: "sk-test", onBack: vi.fn(), embedded: true };
    const { rerender } = renderWithProviders(<RoutedRunView traceId={research.summary.trace_id} {...props} />);
    const root = rootSpanId(research);
    expect(await screen.findByTestId("detail-pane")).toHaveAttribute("data-row-id", root);

    rerender(<RoutedRunView traceId={swarm.summary.trace_id} {...props} />);
    await waitFor(() => expect(screen.getByTestId("run-view")).toHaveAttribute("aria-busy", "true"));
    expect(screen.queryByRole("status", { name: "Loading trace" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(traceDisplayName(research.summary));
    await user.keyboard("{ArrowDown}");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", root);

    act(() => resolveSwarm(swarm));
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(traceDisplayName(swarm.summary)),
    );
    expect(screen.getByTestId("run-view")).toHaveAttribute("aria-busy", "false");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", initialRunSelection(swarm).selectedId);
  });

  it("moves the selection with J / K and closes the detail pane with Esc", async () => {
    const user = userEvent.setup();
    renderRun(research);

    const pane = await screen.findByTestId("detail-pane");
    const root = rootSpanId(research);
    expect(pane).toHaveAttribute("data-row-id", root);
    await user.keyboard("{Meta>}j{/Meta}{Control>}j{/Control}");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", root);
    await user.keyboard("j");
    expect(screen.getByTestId("detail-pane").getAttribute("data-row-id")).not.toBe(root);
    await user.keyboard("k");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", root);
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("detail-pane")).not.toBeInTheDocument();
  });

  it.each(["button", "Escape"])("reopens the selected step after closing details with %s", async (method) => {
    const user = userEvent.setup();
    renderRun(research);
    await screen.findByTestId("detail-pane");
    await user.keyboard("j");
    const selectedId = screen.getByTestId("detail-pane").getAttribute("data-row-id");
    expect(selectedId).not.toBe(rootSpanId(research));
    if (method === "button") await user.click(screen.getByRole("button", { name: "close detail" }));
    else await user.keyboard("{Escape}");
    expect(screen.queryByTestId("detail-pane")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", selectedId);
    expect(screen.queryByRole("button", { name: "Show details" })).not.toBeInTheDocument();
  });

  it("keeps loaded steps and totals after a page fails, then retries the same cursor", async () => {
    const user = userEvent.setup();
    const summary = { ...research.summary, span_count: 2 };
    const first: Trace = { ...research, summary, spans: research.spans.slice(0, 1), next_cursor: "next-page" };
    const second: Trace = {
      ...research,
      summary,
      spans: [
        { ...research.spans[1], type: "tool", name: "later-page-tool", parent_span_id: research.spans[0].span_id },
      ],
      next_cursor: null,
    };
    vi.mocked(agentTraceCall).mockReset();
    vi.mocked(agentTraceCall)
      .mockResolvedValueOnce(first)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(second);
    renderWithProviders(<RoutedRunView traceId={research.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} />);

    expect(await screen.findByText("Showing 1 of 2 steps")).toBeVisible();
    const before = screen.getByRole("banner").textContent;
    await user.click(screen.getByRole("button", { name: "Load more steps" }));
    expect(await screen.findByText("Could not load more steps. Your loaded steps are still available.")).toBeVisible();
    expect(screen.getByRole("tree", { name: "Spans in time order" })).toHaveTextContent(research.spans[0].name);
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("later-page-tool")).toBeVisible();
    expect(screen.getByRole("tree", { name: "Spans in time order" })).toHaveTextContent(research.spans[0].name);
    expect(screen.getAllByRole("treeitem")).toHaveLength(2);
    expect(screen.getByRole("banner")).toHaveTextContent(before ?? "");
    expect(screen.queryByRole("button", { name: "Load more steps" })).not.toBeInTheDocument();
    expect(vi.mocked(agentTraceCall).mock.calls.map((call) => call[3])).toEqual([null, "next-page", "next-page"]);
  });

  it("refreshes a failed later page from one new snapshot", async () => {
    const user = userEvent.setup();
    const summary = { ...research.summary, span_count: 3 };
    const first: Trace = { ...research, summary, spans: research.spans.slice(0, 1), next_cursor: "old-second" };
    const second: Trace = {
      ...research,
      summary,
      spans: [
        { ...research.spans[1], type: "tool", name: "old-snapshot-tool", parent_span_id: research.spans[0].span_id },
      ],
      next_cursor: "old-third",
    };
    const fresh: Trace = {
      ...first,
      summary: { ...summary, span_count: 2 },
      spans: [{ ...research.spans[0], name: "fresh-root" }],
      next_cursor: "fresh-second",
    };
    const freshSecond: Trace = { ...fresh, spans: [{ ...second.spans[0], name: "fresh-tool" }], next_cursor: null };
    vi.mocked(agentTraceCall).mockReset();
    vi.mocked(agentTraceCall)
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second)
      .mockRejectedValueOnce(new Error("Trace changed while paging; refresh the trace"))
      .mockResolvedValueOnce(fresh)
      .mockResolvedValueOnce(freshSecond);
    renderWithProviders(<RoutedRunView traceId={research.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "Load more steps" }));
    expect(await screen.findByText("old-snapshot-tool")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Load more steps" }));
    await user.click(await screen.findByRole("button", { name: "Refresh trace" }));
    expect(await screen.findByText("Showing 1 of 2 steps")).toBeVisible();
    expect(screen.queryByText("old-snapshot-tool")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load more steps" }));
    expect(await screen.findByText("fresh-tool")).toBeVisible();
    expect(screen.getAllByRole("treeitem")).toHaveLength(2);
    expect(vi.mocked(agentTraceCall).mock.calls.map((call) => call[3])).toEqual([
      null,
      "old-second",
      "old-third",
      null,
      "fresh-second",
    ]);
  });

  it("keeps a loaded snapshot on focus and reconnect", async () => {
    testQueryClient.setQueryDefaults(["agentTrace"], { refetchOnWindowFocus: true, refetchOnReconnect: true });
    vi.mocked(agentTraceCall).mockReset();
    renderRun(research);
    await screen.findByTestId("detail-pane");
    await testQueryClient.invalidateQueries({ queryKey: ["agentTrace"], refetchType: "none" });
    await act(async () => {
      focusManager.setFocused(false);
      onlineManager.setOnline(false);
      focusManager.setFocused(true);
      onlineManager.setOnline(true);
    });
    await waitFor(() => expect(testQueryClient.isFetching()).toBe(0));
    expect(vi.mocked(agentTraceCall)).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("tree", { name: "Spans in time order" })).toHaveTextContent(research.spans[0].name);
  });

  it("keeps a way back to the runs table when a run fails to load", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    vi.mocked(agentTraceCall).mockRejectedValue(new Error("Traces are temporarily unavailable"));
    renderWithProviders(<RoutedRunView traceId="big" accessToken="sk-test" onBack={onBack} />);

    expect(await screen.findByText("Traces are temporarily unavailable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /back to traces/i }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("loads the run on retry after a failed load", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceCall).mockRejectedValueOnce(new Error("Traces are temporarily unavailable"));
    renderRun(research);

    await user.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent(traceDisplayName(research.summary));
  });

  it("finds a step beyond a folded group's first page and reveals it after search clears", async () => {
    const user = userEvent.setup();
    const root = research.spans.find((span) => span.parent_span_id === null)!;
    const children = Array.from(
      { length: 45 },
      (_, index): Span => ({
        ...root,
        span_id: `case-${index}`,
        parent_span_id: root.span_id,
        type: "tool",
        name: "check_case",
        input_preview: `Case ${index}`,
        start_offset_ms: index + 1,
        status: index === 44 ? "error" : "ok",
      }),
    );
    renderRun({ ...research, spans: [root, ...children] });
    const search = await screen.findByRole("textbox", { name: "Search steps" });
    fireEvent.change(search, { target: { value: "case 43" } });
    expect(screen.getAllByRole("treeitem")).toHaveLength(1);
    await user.click(screen.getByRole("treeitem"));
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", "case-43");
    await user.click(screen.getByRole("button", { name: "Clear step search" }));
    expect(screen.getByRole("treeitem", { selected: true })).toHaveAttribute("data-row-id", "case-43");
    await user.click(screen.getByRole("button", { name: /^Errors/ }));
    expect(screen.getAllByRole("treeitem")).toHaveLength(1);
    expect(screen.getByRole("treeitem")).toHaveAttribute("data-row-id", "case-44");
    fireEvent.change(search, { target: { value: "not present" } });
    expect(screen.getByText("No matching steps")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getAllByRole("treeitem").length).toBeGreaterThan(1);
  });

  it("restores the step search and errors filter from a shared link", async () => {
    vi.mocked(agentTraceCall).mockResolvedValue(research);
    renderWithProviders(<RoutedRunView traceId={research.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} />, {
      searchParams: "?steps_q=no+such+step&errors=true",
    });
    expect(await screen.findByRole("textbox", { name: "Search steps" })).toHaveValue("no such step");
    expect(screen.getByText("No matching steps")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getAllByRole("treeitem").length).toBeGreaterThan(1);
  });

  it("does not navigate steps while typing or moving the search cursor", async () => {
    renderRun(research);
    const search = await screen.findByRole("textbox", { name: "Search steps" });
    const selected = screen.getByTestId("detail-pane").getAttribute("data-row-id");
    fireEvent.change(search, { target: { value: "jk" } });
    expect(search).toHaveValue("jk");
    for (const key of ["j", "k", "ArrowDown", "ArrowUp"]) {
      fireEvent.keyDown(search, { key });
    }
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", selected);
  });

  it("distinguishes a completed run with recovered step errors from a failed run", async () => {
    renderRun({ ...research, summary: { ...research.summary, status: "ok", error_count: 2 } });
    const header = await screen.findByRole("banner");
    expect(header).toHaveTextContent("Completed");
    expect(header).toHaveTextContent("Step errors 2");
    expect(header).not.toHaveTextContent("Failed");
  });

  it("copies a curl one-liner for Claude / Codex", async () => {
    const user = userEvent.setup();
    renderRun(research);

    await user.click(await screen.findByRole("button", { name: /copy for agent/i }));
    expect(copyToClipboard).toHaveBeenCalledWith(agentHandoffText(research.summary.trace_id), "Command copied");
    expect(agentHandoffText("t1")).toContain('"http://proxy.test/v1/traces/t1?format=md"');
    expect(agentHandoffText("t1", "s1")).toContain("&span_id=s1");
  });
});

describe("initialRunSelection", () => {
  const base = research.spans.find((s) => s.parent_span_id === null) as Span;
  const child = (over: Partial<Span>): Span => {
    const defaults: Partial<Span> = { parent_span_id: base.span_id, status: "ok" };
    return { ...base, ...defaults, ...over };
  };

  it("lands on a visible failure, never on a framework span the tree hides", () => {
    const hiddenFields: Partial<Span> = { span_id: "mw", type: "framework", status: "error", start_offset_ms: 1 };
    const toolFields: Partial<Span> = { span_id: "tool", type: "tool", status: "error", start_offset_ms: 5 };
    const hiddenFailure = child(hiddenFields);
    const toolFailure = child(toolFields);
    const trace = { ...research, spans: [base, hiddenFailure, toolFailure] };
    expect(initialRunSelection(trace).selectedId).toBe("tool");
  });

  it("folds other agent branches while revealing the failed step", () => {
    const first = child({ span_id: "first", type: "agent" });
    const second = child({ span_id: "second", type: "agent" });
    const failure = child({ span_id: "failed", parent_span_id: "second", type: "tool", status: "error" });
    const { selectedId, state } = initialRunSelection({ ...research, spans: [base, first, second, failure] });
    expect(selectedId).toBe("failed");
    expect(state.collapsedSpanIds.has("first")).toBe(true);
    expect(state.collapsedSpanIds.has("second")).toBe(false);
  });

  it("falls back to the nearest visible ancestor when only a hidden span failed", () => {
    const agent = child({ span_id: "agent", type: "agent", name: "researcher" });
    const hiddenFields: Partial<Span> = { span_id: "mw", parent_span_id: "agent", type: "framework", status: "error" };
    const hiddenFailure = child(hiddenFields);
    const trace = { ...research, spans: [base, agent, hiddenFailure] };
    expect(initialRunSelection(trace).selectedId).toBe("agent");
  });
});

it("opens a cited span instead of the default failed span", () => {
  const cited = research.spans.find((span) => span.parent_span_id !== null)!;
  expect(initialRunSelection(research, cited.span_id).selectedId).toBe(cited.span_id);
  expect(initialRunSelection(research, "missing")).toEqual(initialRunSelection(research));
});
