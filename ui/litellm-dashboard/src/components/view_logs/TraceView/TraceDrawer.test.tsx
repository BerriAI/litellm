import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import researchTrace from "./__fixtures__/research_trace.json";
import swarmTrace from "./__fixtures__/swarm_trace.json";
import { agentHandoffText, initialRunSelection, RunView } from "./TraceDrawer";
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

const renderRun = (trace: Trace) => {
  vi.mocked(agentTraceCall).mockResolvedValue(trace);
  return renderWithProviders(<RunView traceId={trace.summary.trace_id} accessToken="sk-test" onBack={vi.fn()} />);
};

const rootSpanId = (trace: Trace): string => trace.spans.find((s) => s.parent_span_id === null)?.span_id ?? "";

describe("RunView", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(copyToClipboard).mockClear();
  });

  it("shows a one-line run header: agent name, trace id, duration and steps", async () => {
    renderRun(research);

    const header = await screen.findByRole("banner");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(traceDisplayName(research.summary));
    expect(header).toHaveTextContent(research.summary.trace_id);
    expect(header).toHaveTextContent("duration 40.20s");
    expect(header).toHaveTextContent(`steps ${research.summary.span_count}`);
    expect(header).not.toHaveTextContent("failed");
  });

  it("folds researcher ×12 in the span tree", async () => {
    renderRun(swarm);

    const tree = await screen.findByRole("tree", { name: "Spans in time order" });
    expect(tree).toHaveTextContent("researcher×12");
    expect(screen.getByRole("banner")).toHaveTextContent(`failed ${swarm.summary.error_count}`);
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

  it("moves the selection with J / K and closes the detail pane with Esc", async () => {
    const user = userEvent.setup();
    renderRun(research);

    const pane = await screen.findByTestId("detail-pane");
    const root = rootSpanId(research);
    expect(pane).toHaveAttribute("data-row-id", root);
    await user.keyboard("j");
    expect(screen.getByTestId("detail-pane").getAttribute("data-row-id")).not.toBe(root);
    await user.keyboard("k");
    expect(screen.getByTestId("detail-pane")).toHaveAttribute("data-row-id", root);
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("detail-pane")).not.toBeInTheDocument();
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
