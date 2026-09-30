import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import researchTrace from "./__fixtures__/research_trace.json";
import swarmTrace from "./__fixtures__/swarm_trace.json";
import { initialTraceView, TraceDrawer } from "./TraceDrawer";
import type { SpanDetail, Trace } from "./traceTypes";

vi.mock("../../networking", () => ({
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
}));

import { agentTraceCall, agentTraceSpanCall } from "../../networking";

const swarm = swarmTrace as Trace;
const research = researchTrace as Trace;

const spanDetail = (spanId: string): SpanDetail => ({
  span_id: spanId,
  input: JSON.stringify([{ role: "user", content: "Compare ClickHouse and Postgres ingest" }]),
  output: JSON.stringify({
    role: "assistant",
    content: "Calling tools",
    tool_calls: [{ name: "lookup_benchmark", args: { db: "clickhouse" } }],
  }),
  attributes: { "gen_ai.system": "openai" },
});

const renderDrawer = (trace: Trace, props: Partial<React.ComponentProps<typeof TraceDrawer>> = {}) => {
  vi.mocked(agentTraceCall).mockResolvedValue(trace);
  return renderWithProviders(
    <TraceDrawer open traceId={trace.summary.trace_id} accessToken="sk-test" onClose={vi.fn()} {...props} />,
  );
};

describe("TraceDrawer", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, spanId) => spanDetail(spanId));
  });

  it("opens a healthy trace on the Steps view with the stats strip and per-agent cost split", async () => {
    renderDrawer(research);

    expect(await screen.findByRole("heading", { name: "research_lead" })).toBeInTheDocument();
    const stats = screen.getByLabelText("Trace stats");
    expect(stats).toHaveTextContent("40.20s");
    expect(stats).toHaveTextContent("LLM calls21");
    expect(stats).toHaveTextContent("Tool calls25");
    expect(stats).toHaveTextContent("$0.1283");
    expect(screen.getByLabelText("Cost by agent")).toHaveTextContent("researcher $0.0892");
    expect(screen.getByRole("button", { name: "Steps" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("list", { name: "Trace steps" })).toBeInTheDocument();
  });

  it("opens a failed trace on the Tree view with the first failed span selected", async () => {
    renderDrawer(swarm);

    const tree = await screen.findByRole("tree", { name: "Span tree" });
    expect(screen.getByRole("button", { name: "Tree" })).toHaveAttribute("aria-pressed", "true");
    const selected = within(tree).getByRole("treeitem", { selected: true });
    expect(selected).toHaveTextContent("lookup_benchmark");
    // the 12 researcher invocations collapse into one group row
    expect(within(tree).getByText(/researcher ×12 · \$0\.0635 · p50/)).toBeInTheDocument();
  });

  it("shows the LiteLLM request card with model group and deployment for an LLM span", async () => {
    const user = userEvent.setup();
    const onOpenRequestLog = vi.fn();
    const llm = research.spans.find((s) => s.type === "llm" && s.litellm) as Trace["spans"][number];
    renderDrawer(research, { initialSpanId: llm.span_id, onOpenRequestLog });

    const card = await screen.findByRole("region", { name: "LiteLLM request" });
    expect(card).toHaveTextContent(llm.litellm?.request_id ?? "");
    expect(card).toHaveTextContent("claude-sonnet-4-5");
    expect(card).toHaveTextContent("openai/claude-sonnet-4-5 (openai)");
    await user.click(within(card).getByRole("button", { name: "Open request log →" }));
    expect(onOpenRequestLog).toHaveBeenCalledWith(llm.litellm?.request_id);
    expect(await screen.findByText(/lookup_benchmark/, { selector: "b" })).toBeInTheDocument();
  });

  it("hides framework spans until the toggle is checked", async () => {
    const user = userEvent.setup();
    renderDrawer(research);
    await user.click(await screen.findByRole("button", { name: "Tree" }));

    const total = research.spans.length;
    expect(screen.getByText(new RegExp(`^\\d+ of ${total} spans$`))).not.toHaveTextContent(new RegExp(`^${total} of`));
    await user.click(screen.getByRole("checkbox"));
    expect(screen.getByText(`${total} of ${total} spans`)).toBeInTheDocument();
  });

  it("moves the selection with J and K", async () => {
    const user = userEvent.setup();
    renderDrawer(research);
    const steps = await screen.findByRole("list", { name: "Trace steps" });
    const items = within(steps).getAllByRole("listitem");
    expect(items[0]).toHaveAttribute("aria-current", "step");
    await user.keyboard("j");
    expect(items[1]).toHaveAttribute("aria-current", "step");
    await user.keyboard("k");
    expect(items[0]).toHaveAttribute("aria-current", "step");
  });

  it("jumps from an agent graph invocation to the tree focused on that span", async () => {
    const user = userEvent.setup();
    renderDrawer(swarm);
    await user.click(await screen.findByRole("button", { name: "Graph" }));
    await user.click(screen.getByRole("button", { name: "Agent critic" }));
    await user.click(within(screen.getByRole("list", { name: "critic invocations" })).getByRole("button"));

    const tree = screen.getByRole("tree", { name: "Span tree" });
    expect(within(tree).getByRole("treeitem", { selected: true })).toHaveTextContent("critic");
  });
});

describe("initialTraceView", () => {
  it("prefers an explicitly requested span", () => {
    const llm = swarm.spans.find((s) => s.type === "llm") as Trace["spans"][number];
    expect(initialTraceView(swarm, llm.span_id)).toEqual({ mode: "steps", selectedId: llm.span_id });
  });
});
