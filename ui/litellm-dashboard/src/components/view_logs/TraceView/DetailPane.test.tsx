import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { DetailPane } from "./DetailPane";
import type { GroupRowData, SpanRowData } from "./traceTree";
import type { Span, SpanDetail, Trace } from "./traceTypes";

vi.mock("../../networking", () => ({
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test/",
}));

import { agentTraceSpanCall } from "../../networking";

const span = (overrides: Partial<Span> & Pick<Span, "span_id">): Span => ({
  parent_span_id: "root",
  name: overrides.span_id,
  type: "chain",
  agent: "support_triage_agent",
  start_offset_ms: 0,
  duration_ms: 1300,
  status: "ok",
  error: null,
  input_preview: "",
  model: null,
  input_tokens: 0,
  output_tokens: 0,
  litellm_request_id: null,
  ...overrides,
});

const rootFields: SpanFields = { span_id: "root", parent_span_id: null, name: "support_triage_agent", type: "agent" };
const llmFields: SpanFields = {
  span_id: "llm1",
  name: "ChatOpenAI",
  type: "llm",
  model: "claude-sonnet-4-5",
  input_tokens: 659,
  output_tokens: 60,
  litellm_request_id: "chatcmpl-abc",
};
const failedToolFields: SpanFields = {
  span_id: "tool1",
  name: "get_customer_plan",
  type: "tool",
  status: "error",
  error:
    "ValueError('customer acme-404 not found in billing DB')Traceback (most recent call last):\n  File \"x.py\", line 1",
};
const root = span(rootFields);
const llm = span(llmFields);
const failedTool = span(failedToolFields);

const trace: Trace = {
  summary: {
    trace_id: "t1",
    name: "support_triage_agent",
    service: "research-agent",
    input_preview: '[{"role": "user", "content": "Customer acme-404 says billing is wrong."}]',
    start_time: "2026-09-30T06:43:52.928000+00:00",
    duration_ms: 1310,
    status: "ok",
    span_count: 3,
    agent_count: 1,
    agent_invocations: 1,
    llm_calls: 1,
    tool_calls: 1,
    error_count: 1,
    input_tokens: 659,
    output_tokens: 60,
    models: ["claude-sonnet-4-5"],
  } as Trace["summary"],
  agents: [],
  spans: [root, llm, failedTool],
};

const LONG_NOTE = "Escalated twice already. ".repeat(6).trim();

const details: Record<string, SpanDetail> = {
  llm1: {
    span_id: "llm1",
    input: JSON.stringify([
      { role: "system", content: "You are a LiteLLM support agent." },
      { role: "user", content: "Customer acme-404 says billing is wrong." },
    ]),
    output: JSON.stringify({
      role: "assistant",
      content: "",
      tool_calls: [{ name: "get_customer_plan", args: { customer_id: "acme-404", note: LONG_NOTE } }],
    }),
    attributes: { "gen_ai.request.model": "claude-sonnet-4-5" },
  },
  tool1: { span_id: "tool1", input: '{"customer_id":"acme-404"}', output: "", attributes: {} },
  root: {
    span_id: "root",
    input: JSON.stringify([{ role: "user", content: "Customer acme-404 says billing is wrong." }]),
    output: JSON.stringify({ role: "assistant", content: "Customer acme-404 is on the Enterprise plan." }),
    attributes: {},
  },
};

const spanRow = (s: Span): SpanRowData => ({
  kind: "span",
  id: s.span_id,
  span: s,
  depth: 1,
  hasChildren: false,
  collapsed: false,
});

const renderPane = (row: SpanRowData | GroupRowData) =>
  renderWithProviders(<DetailPane trace={trace} row={row} accessToken="sk-test" onClose={vi.fn()} />);

describe("DetailPane", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceSpanCall).mockReset();
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, spanId) => details[spanId]);
  });

  it("renders the span tabs and the fetched LLM conversation with its tool call", async () => {
    renderPane(spanRow(llm));
    expect(screen.getByRole("tab", { name: "Content" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Request" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Attributes" })).toBeInTheDocument();
    expect(await screen.findByText("You are a LiteLLM support agent.")).toBeInTheDocument();
    expect(screen.getAllByText("get_customer_plan").length).toBeGreaterThan(0);
    expect(vi.mocked(agentTraceSpanCall)).toHaveBeenCalledWith("sk-test", "t1", "llm1", undefined);
  });

  it("shows a tool failure as 'Tool · <reason>' with the exception line and no traceback", async () => {
    renderPane(spanRow(failedTool));
    const error = screen.getByRole("region", { name: "Error" });
    expect(error).toHaveTextContent("Tool · ValueError");
    expect(error).toHaveTextContent("ValueError('customer acme-404 not found in billing DB')");
    expect(error).not.toHaveTextContent("Traceback");
    const input = await screen.findByRole("region", { name: "Input" });
    expect(input).toHaveTextContent("customer_id");
    expect(input).toHaveTextContent("acme-404");
  });

  it("shows the LiteLLM request facts on the Request tab", async () => {
    const user = userEvent.setup();
    renderPane(spanRow(llm));
    await user.click(screen.getByRole("tab", { name: "Request" }));
    expect(await screen.findByText("chatcmpl-abc")).toBeInTheDocument();
    expect(screen.getByText("659")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Open request log/ })).toBeInTheDocument();
  });

  it("summarizes a ×N group with its failure pattern", () => {
    const members = Array.from({ length: 12 }, (_, i) => {
      const timedOut: SpanFields = {
        span_id: `f${i}`,
        name: "lookup_benchmark",
        type: "tool",
        status: "error",
        error: "TimeoutError('slow')",
      };
      return span(timedOut);
    });
    const groupRow: GroupRowData = {
      kind: "group",
      id: "grp",
      depth: 1,
      name: "lookup_benchmark",
      type: "tool",
      agent: "researcher",
      members,
      failedCount: 12,
      p50Duration: 640,
      isFailureGroup: true,
      expanded: false,
    };
    renderPane(groupRow);
    const pane = screen.getByRole("complementary", { name: "Group details" });
    expect(pane).toHaveTextContent("lookup_benchmark ×12");
    expect(pane).toHaveTextContent("Invocations12");
    expect(pane).toHaveTextContent("Failed12");
    expect(pane).toHaveTextContent("TimeoutError('slow')");
  });

  it("'Copy step' copies a curl for just this span as Markdown", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    renderPane(spanRow(llm));
    await user.click(screen.getByRole("button", { name: "Copy step" }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0][0]).toContain("http://proxy.test/v1/traces/t1?format=md&span_id=llm1");
  });

  it("renders the AI tool call as a card and expands a long argument on click", async () => {
    const user = userEvent.setup();
    renderPane(spanRow(llm));
    const output = await screen.findByRole("region", { name: "Output" });
    expect(output).toHaveTextContent("AI");
    expect(output).toHaveTextContent("get_customer_plan");
    const expand = within(output).getAllByRole("button", { name: "Expand note" })[0];
    expect(within(output).queryAllByText(LONG_NOTE, { selector: "pre", ignore: "[inert] *" })).toHaveLength(0);
    await user.click(expand);
    expect(within(output).getAllByRole("button", { name: "Collapse note" })[0]).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(within(output).getAllByText(LONG_NOTE, { selector: "pre", ignore: "[inert] *" })).not.toHaveLength(0);
  });

  it("collapses the Input section without touching Output", async () => {
    const user = userEvent.setup();
    renderPane(spanRow(llm));
    const input = await screen.findByRole("region", { name: "Input" });
    const systemText = "You are a LiteLLM support agent.";
    expect(within(input).getByText(systemText, { ignore: "[inert] *" })).toBeInTheDocument();
    await user.click(within(input).getByRole("button", { name: "Input" }));
    expect(within(input).getByRole("button", { name: "Input" })).toHaveAttribute("aria-expanded", "false");
    expect(within(input).queryByText(systemText, { ignore: "[inert] *" })).not.toBeInTheDocument();
    const output = screen.getByRole("region", { name: "Output" });
    expect(within(output).getAllByText("get_customer_plan", { ignore: "[inert] *" })).not.toHaveLength(0);
  });
});
