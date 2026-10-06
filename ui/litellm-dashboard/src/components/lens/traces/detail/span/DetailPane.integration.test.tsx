import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { type ComponentProps, useState } from "react";

import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
import { DetailPane } from "./DetailPane";
import type { SpanTab } from "../../routing";
import { absoluteTime, SpanHoverCard, spanFacts } from "../tree/SpanHoverCard";
import type { GroupRowData, SpanRowData } from "../../tree";
import type { Span, SpanDetail, SpanErrorPage, Trace } from "../../types";

vi.mock("../../../../networking", () => ({
  agentTraceSpanCall: vi.fn(),
  agentTraceSpanErrorCall: vi.fn(),
  uiSpendLogsCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test/",
}));

vi.mock("../../../../logs/detail", () => ({
  LogDetailsDrawer: ({
    open,
    logEntry,
    onClose,
    backTo,
  }: {
    open: boolean;
    logEntry: { request_id: string } | null;
    onClose: () => void;
    backTo?: { label: string };
  }) =>
    open && logEntry ? (
      <div role="dialog" aria-label="Request log">
        {logEntry.request_id}
        {backTo && <button onClick={onClose}>Back to {backTo.label}</button>}
      </div>
    ) : null,
}));

import { agentTraceSpanCall, agentTraceSpanErrorCall, uiSpendLogsCall } from "../../../../networking";

type SpanFields = Partial<Span> & Pick<Span, "span_id">;

const span = (overrides: SpanFields): Span => ({
  parent_span_id: "root",
  name: overrides.span_id,
  type: "chain",
  agent: "support_triage_agent",
  framework: "",
  start_offset_ms: 0,
  duration_ms: 1300,
  status: "ok",
  error: null,
  error_truncated: false,
  input_preview: "",
  model: null,
  input_tokens: 0,
  output_tokens: 0,
  litellm_request_id: null,
  spend: null,
  spend_log_request_id: null,
  spend_match: null,
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
  spend: 0.0002,
  spend_log_request_id: "req_7f3a9c2e1b44",
  spend_match: "matched",
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

const standardDetail: SpanDetail = {
  span_id: "llm1",
  input: "raw input left unparsed",
  output: "raw output left unparsed",
  input_ui: { kind: "fields", fields: [{ key: "ticket_id", value: "T-981" }] },
  output_ui: {
    kind: "messages",
    messages: [
      {
        role: "assistant",
        content: "Refund approved for T-981.",
        name: null,
        tool_calls: [{ name: "issue_refund", arguments: '{"amount_usd": 40}' }],
      },
    ],
  },
  attributes: {},
};

const textDetail: SpanDetail = {
  span_id: "llm1",
  input: "",
  output: '{"answer": "all done"}',
  output_ui: { kind: "text", text: "all done" },
  attributes: {},
};

const workflowDetail: SpanDetail = {
  span_id: "root",
  input: '{"init_state": {"config": {"timeout": null}}}',
  input_ui: {
    kind: "fields",
    fields: [
      { key: "init_state", value: '{"config": {"timeout": null, "workers": 4}}' },
      { key: "start_event", value: "AgentWorkflowStartEvent()" },
    ],
  },
  output: "StopEvent(result='done')",
  output_ui: { kind: "text", text: "An **agent trace** records each step" },
  attributes: {},
};

const langchainDetail: SpanDetail = {
  span_id: "root",
  input: "",
  output: '{"messages": []}',
  output_ui: {
    kind: "fields",
    fields: [
      {
        key: "messages",
        value: JSON.stringify([
          { type: "human", data: { content: "What is an agent trace?" } },
          { type: "ai", data: { content: "A record of every step an agent took." } },
        ]),
      },
    ],
  },
  attributes: {},
};

const failedToolMessageDetail: SpanDetail = {
  span_id: "tool1",
  input: '{"customer_id":"acme-404"}',
  output: "raw tool output",
  output_ui: { kind: "messages", messages: [{ role: "tool", content: "permission denied: /etc/shadow" }] },
  attributes: {},
};

const spanRow = (s: Span): SpanRowData => ({
  kind: "span",
  id: s.span_id,
  span: s,
  depth: 1,
  hasChildren: false,
  collapsed: false,
});

function LocalDetailPane(props: Omit<ComponentProps<typeof DetailPane>, "spanTab" | "onSpanTabChange">) {
  const [spanTab, setSpanTab] = useState<SpanTab>("content");
  return <DetailPane {...props} spanTab={spanTab} onSpanTabChange={setSpanTab} />;
}

const renderPane = (row: SpanRowData | GroupRowData) =>
  renderWithProviders(<LocalDetailPane trace={trace} row={row} accessToken="sk-test" onClose={vi.fn()} />);

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
    expect(await screen.findByText("Customer acme-404 says billing is wrong.")).toBeVisible();
    expect(screen.getAllByText("get_customer_plan").length).toBeGreaterThan(0);
    expect(vi.mocked(agentTraceSpanCall)).toHaveBeenCalledWith("sk-test", "t1", "llm1", undefined);
  });

  it("keeps the output visible when a step contains a long input conversation", async () => {
    const user = userEvent.setup();
    const messages = Array.from({ length: 120 }, (_, index) => ({
      role: "user" as const,
      content: `Review case ${index}`,
    }));
    vi.mocked(agentTraceSpanCall).mockResolvedValue({
      ...details.root,
      input: JSON.stringify(messages),
      input_ui: { kind: "messages", messages },
    });
    renderPane(spanRow(root));
    const input = await screen.findByRole("button", { name: "Input 120 messages" });
    expect(input).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("Customer acme-404 is on the Enterprise plan.")).toBeVisible();
    expect(screen.getByText("Review case 119")).not.toBeVisible();
    await user.click(input);
    expect(screen.getByText("Review case 119")).toBeVisible();
  });

  it("shows a tool failure as 'Tool · <reason>' with the exception line and no traceback", async () => {
    renderPane(spanRow(failedTool));
    const error = screen.getByRole("region", { name: "Error" });
    expect(error).toHaveTextContent("Tool · ValueError");
    expect(error).toHaveTextContent("ValueError('customer acme-404 not found in billing DB')");
    expect(error).not.toHaveTextContent("Traceback");
    const input = await screen.findByRole("region", { name: /^Input/ });
    expect(input).toHaveTextContent("customer_id");
    expect(input).toHaveTextContent("acme-404");
  });

  it("links a priced LLM step to its spend log in the header and opens it by that request id", async () => {
    const user = userEvent.setup();
    vi.mocked(uiSpendLogsCall).mockResolvedValue({ data: [{ request_id: "req_7f3a9c2e1b44" }] });
    renderPane(spanRow(llm));
    const link = screen.getByRole("button", { name: "Open LiteLLM spend log req_7f3a9c2e1b44" });
    expect(link).toHaveTextContent("LiteLLM Spend Log$0.0002");
    await user.click(link);
    expect(await screen.findByRole("dialog", { name: "Request log" })).toHaveTextContent("req_7f3a9c2e1b44");
    expect(vi.mocked(uiSpendLogsCall).mock.calls[0][0].params).toEqual({ request_id: "req_7f3a9c2e1b44" });
  });

  it("returns from the spend log to the same agent trace step", async () => {
    const user = userEvent.setup();
    vi.mocked(uiSpendLogsCall).mockResolvedValue({ data: [{ request_id: "req_7f3a9c2e1b44" }] });
    renderPane(spanRow(llm));
    await user.click(screen.getByRole("button", { name: "Open LiteLLM spend log req_7f3a9c2e1b44" }));
    const drawer = await screen.findByRole("dialog", { name: "Request log" });
    await user.click(within(drawer).getByRole("button", { name: "Back to Lens trace" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Request log" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Open LiteLLM spend log req_7f3a9c2e1b44" })).toBeVisible();
  });

  it.each([
    { spend_match: "no_call_id" as const, reason: "This step records no usable call identifier or transport context" },
    { spend_match: "no_spend_log" as const, reason: "No matching spend log belongs to this run" },
    { spend_match: "incomplete_evidence" as const, reason: "This step does not account for every request in the call" },
  ])("says why an unpriced LLM step has no cost ($spend_match)", async ({ spend_match, reason }) => {
    const user = userEvent.setup();
    const unpriced: SpanFields = { ...llmFields, spend: null, spend_log_request_id: null, spend_match };
    renderPane(spanRow(span(unpriced)));
    expect(screen.getByTitle(reason)).toHaveTextContent("Cost not matched");
    expect(screen.queryByRole("button", { name: /Open LiteLLM spend log/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Request" }));
    expect(await screen.findByText(reason, { selector: "p" })).toBeInTheDocument();
  });

  it("keeps the Cost fact for a priced step that has no single spend log to link", () => {
    const pricedFromRetries: SpanFields = { ...llmFields, spend: 0.0005, spend_log_request_id: null };
    renderPane(spanRow(span(pricedFromRetries)));
    expect(screen.queryByRole("button", { name: /Open LiteLLM spend log/ })).not.toBeInTheDocument();
    expect(screen.getByText("Cost").parentElement).toHaveTextContent("Cost$0.0005");
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

  it("renders the assistant tool call as a card and expands a long argument on click", async () => {
    const user = userEvent.setup();
    renderPane(spanRow(llm));
    const output = await screen.findByRole("region", { name: "Output" });
    expect(output).toHaveTextContent("Assistant");
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
    const input = await screen.findByRole("region", { name: /^Input/ });
    const systemText = "You are a LiteLLM support agent.";
    expect(within(input).queryByText(systemText)).not.toBeInTheDocument();
    await user.click(within(input).getByRole("button", { name: "Expand System" }));
    expect(within(input).getByText(systemText, { ignore: "[inert] *" })).toBeVisible();
    await user.click(within(input).getByRole("button", { name: /^Input/ }));
    expect(within(input).getByRole("button", { name: /^Input/ })).toHaveAttribute("aria-expanded", "false");
    expect(within(input).queryByText(systemText, { ignore: "[inert] *" })).not.toBeInTheDocument();
    const output = screen.getByRole("region", { name: "Output" });
    expect(within(output).getAllByText("get_customer_plan", { ignore: "[inert] *" })).not.toHaveLength(0);
  });

  it("renders the standard input_ui / output_ui instead of re-parsing the raw payload", async () => {
    vi.mocked(agentTraceSpanCall).mockResolvedValue(standardDetail);
    renderPane(spanRow(llm));
    const input = await screen.findByRole("region", { name: /^Input/ });
    expect(input).toHaveTextContent("ticket_id");
    expect(input).toHaveTextContent("T-981");
    expect(input).not.toHaveTextContent("raw input left unparsed");
    const output = screen.getByRole("region", { name: "Output" });
    expect(output).toHaveTextContent("Assistant");
    expect(output).toHaveTextContent("Refund approved for T-981.");
    expect(output).toHaveTextContent("issue_refund");
    expect(output).toHaveTextContent("amount_usd");
    expect(output).not.toHaveTextContent("raw output left unparsed");
  });

  it("identifies and styles a failed tool result when its output arrives as a single message", async () => {
    vi.mocked(agentTraceSpanCall).mockResolvedValue(failedToolMessageDetail);
    renderPane(spanRow(failedTool));
    const output = await screen.findByRole("region", { name: "Output" });
    const result = within(output).getByRole("group", { name: "Failed tool result" });
    expect(result).toBeVisible();
    expect(result).toHaveClass("text-destructive");
    expect(result).toHaveTextContent("permission denied: /etc/shadow");
    expect(output).not.toHaveTextContent("Assistant");
  });

  it("shows a text output_ui as its plain text", async () => {
    vi.mocked(agentTraceSpanCall).mockResolvedValue(textDetail);
    renderPane(spanRow(llm));
    const output = await screen.findByRole("region", { name: "Output" });
    expect(within(output).getByText("all done", { selector: "pre" })).toBeInTheDocument();
    expect(output).not.toHaveTextContent("answer");
  });

  it("unfolds JSON-encoded field values into a nested tree and shows the raw payload on request", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceSpanCall).mockResolvedValue(workflowDetail);
    renderPane(spanRow(root));
    const input = await screen.findByRole("region", { name: /^Input/ });
    expect(within(input).getByText("AgentWorkflowStartEvent()")).toBeVisible();
    expect(within(input).getByRole("button", { name: "Collapse init_state" })).toHaveAttribute("aria-expanded", "true");
    expect(within(input).queryByText("workers")).not.toBeInTheDocument();
    await user.click(within(input).getByRole("button", { name: "Expand config" }));
    expect(within(input).getByText("workers")).toBeVisible();
    expect(within(input).getByText("4")).toBeVisible();

    const output = screen.getByRole("region", { name: "Output" });
    expect(within(output).getByText("agent trace", { selector: "strong" })).toBeVisible();
    await user.click(within(output).getByRole("radio", { name: "Raw" }));
    expect(within(output).getByText("StopEvent(result='done')", { selector: "pre" })).toBeVisible();
    expect(within(output).queryByText("agent trace", { selector: "strong" })).not.toBeInTheDocument();
    expect(within(input).getByText("AgentWorkflowStartEvent()")).toBeVisible();
  });

  it("renders LangChain's serialized messages as conversation cards", async () => {
    vi.mocked(agentTraceSpanCall).mockResolvedValue(langchainDetail);
    renderPane(spanRow(root));
    const output = await screen.findByRole("region", { name: "Output" });
    expect(within(output).getByRole("button", { name: "Collapse messages" })).toHaveAttribute("aria-expanded", "true");
    expect(within(output).getByRole("button", { name: "Collapse User" })).toBeVisible();
    expect(within(output).getByText("What is an agent trace?")).toBeVisible();
    expect(within(output).getByText("A record of every step an agent took.")).toBeVisible();
    expect(output).not.toHaveTextContent('"type": "human"');
  });

  it("groups ids and OTEL attributes into separate key / value sections on the Attributes tab", async () => {
    const user = userEvent.setup();
    renderPane(spanRow(llm));
    await user.click(screen.getByRole("tab", { name: "Attributes" }));
    const ids = screen.getByRole("region", { name: "Identifiers" });
    expect(ids).toHaveTextContent("span_idllm1");
    expect(ids).toHaveTextContent("parent_span_idroot");
    const attributes = await screen.findByRole("region", { name: "Attributes" });
    expect(attributes).toHaveTextContent("gen_ai.request.model");
    expect(attributes).not.toHaveTextContent("span_id");
  });
});

describe("SpanHoverCard", () => {
  it.each([
    ["retriever", "Retriever"],
    ["embedding", "Embedding"],
    ["reranker", "Reranker"],
    ["guardrail", "Guardrail"],
    ["evaluator", "Evaluator"],
    ["prompt", "Prompt"],
    ["decision", "Decision"],
  ] as const)("renders the %s operation", async (type, label) => {
    const user = userEvent.setup();
    renderWithProviders(
      <SpanHoverCard
        facts={spanFacts(span({ span_id: "specialized", type }))}
        traceStartMs={Date.parse(trace.summary.start_time)}
      >
        <button type="button">row</button>
      </SpanHoverCard>,
    );
    await user.hover(screen.getByRole("button", { name: "row" }));
    const card = await screen.findByTestId("span-hover-card", {}, { timeout: 2000 });
    expect(within(card).getByText(label)).toBeInTheDocument();
  });

  it("shows absolute Start / End times and the agent tag after hovering the row", async () => {
    const user = userEvent.setup();
    const traceStartMs = Date.parse(trace.summary.start_time);
    const timed = span({ span_id: "timed", start_offset_ms: 2000, duration_ms: 3000 });
    renderWithProviders(
      <SpanHoverCard facts={spanFacts(timed)} traceStartMs={traceStartMs}>
        <button type="button">row</button>
      </SpanHoverCard>,
    );
    expect(screen.queryByTestId("span-hover-card")).not.toBeInTheDocument();
    await user.hover(screen.getByRole("button", { name: "row" }));
    const card = await screen.findByTestId("span-hover-card", {}, { timeout: 2000 });
    const time = within(card).getByRole("region", { name: "Time" });
    expect(time).toHaveTextContent(`Start${absoluteTime(traceStartMs, 2000)}`);
    expect(time).toHaveTextContent(`End${absoluteTime(traceStartMs, 5000)}`);
    expect(within(card).getByRole("region", { name: "Tags" })).toHaveTextContent("agent:support_triage_agent");
  });
});

it("retrieves the retained diagnostic one section at a time", async () => {
  const firstPage: SpanErrorPage = {
    span_id: "tool1",
    message: "First diagnostic section",
    total_chars: 100,
    next_cursor: "next-section",
  };
  const lastPage: SpanErrorPage = {
    span_id: "tool1",
    message: "Last diagnostic section",
    total_chars: 100,
    next_cursor: null,
  };
  vi.mocked(agentTraceSpanErrorCall).mockResolvedValueOnce(firstPage).mockResolvedValueOnce(lastPage);
  renderPane(spanRow({ ...failedTool, error_truncated: true }));
  expect(screen.getByText("Error preview truncated")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "View stored diagnostic" }));
  expect(await screen.findByText("First diagnostic section")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Next section" }));
  expect(await screen.findByText("Last diagnostic section")).toBeInTheDocument();
  expect(screen.queryByText("First diagnostic section")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Next section" })).not.toBeInTheDocument();
  expect(agentTraceSpanErrorCall).toHaveBeenLastCalledWith("sk-test", "t1", "tool1", {
    traceRef: undefined,
    cursor: "next-section",
  });
});
