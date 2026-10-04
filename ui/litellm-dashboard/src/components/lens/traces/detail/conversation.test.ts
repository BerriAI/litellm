import { describe, expect, it } from "vitest";
import { buildConversation, conversationSteps, newConversationMessages } from "./conversation";
import type { Span, SpanDetail, TraceMessage } from "../types";
import research from "../__fixtures__/research_trace.json";

const root = { ...research.spans[0], span_id: "root", parent_span_id: null, type: "agent" } as Span;
const user: TraceMessage = { role: "user", content: "Find my order" };
const call: TraceMessage = {
  role: "assistant",
  content: "Checking the order",
  tool_calls: [{ name: "lookup", args: { order: 42 } }],
};
const result: TraceMessage = { role: "tool", content: "Shipped" };
const answer: TraceMessage = { role: "assistant", content: "Your order shipped" };
const detail = (span_id: string, input: unknown, output: unknown): SpanDetail => ({
  span_id,
  input: JSON.stringify(input),
  output: JSON.stringify(output),
  attributes: {},
});

describe("trace conversation", () => {
  it("removes repeated prefixes and trimmed context, but preserves a genuinely repeated question", () => {
    expect(newConversationMessages([user, call, result], [user, call, result, answer])).toEqual([answer]);
    expect(newConversationMessages([user, call, result], [call, result, answer])).toEqual([answer]);
    expect(newConversationMessages([user, answer], [user, answer, user])).toEqual([user]);
    expect(newConversationMessages([user, call, result], [user])).toEqual([]);
  });

  it("reads a tool exchange once, without repeating the root's final answer", () => {
    const first = { ...root, span_id: "first", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const tool = { ...first, span_id: "tool", name: "lookup", type: "tool", start_offset_ms: 2 } as Span;
    const last = { ...first, span_id: "last", start_offset_ms: 3 };
    const details = new Map([
      ["root", detail("root", [user], [answer])],
      ["first", detail("first", [user], [call])],
      ["tool", { ...detail("tool", { order: 42 }, "Shipped"), output: "Shipped" }],
      ["last", detail("last", [user, call, result], [answer])],
    ]);
    const items = buildConversation([root, first, tool, last], details, true);
    expect(items.flatMap((item) => item.messages)).toEqual([user, { ...call, tool_calls: [] }, answer]);
    expect(items.find((item) => item.span.span_id === "tool")).toMatchObject({
      toolCall: { name: "lookup", args: { order: 42 } },
      toolResult: "Shipped",
    });
  });

  it("uses normalized message content and structured tool arguments from the gateway", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm" } as Span;
    const normalized: SpanDetail = {
      span_id: "model",
      input: "unparsed input",
      output: "unparsed output",
      attributes: {},
      input_ui: { kind: "messages", messages: [{ role: "user", content: user.content }] },
      output_ui: {
        kind: "messages",
        messages: [
          { role: "assistant", content: "Checking", tool_calls: [{ name: "lookup", arguments: '{"order":42}' }] },
        ],
      },
    };
    const details = new Map([
      ["root", detail("root", [], [])],
      ["model", normalized],
    ]);
    const items = buildConversation([root, model], details, true);
    expect(items[0].messages).toEqual([
      user,
      { role: "assistant", content: "Checking", tool_calls: [{ name: "lookup", args: { order: 42 } }] },
    ]);
  });

  it("keeps parallel agent histories separate and excludes framework scaffolding", () => {
    const agent = { ...root, span_id: "agent", parent_span_id: "root" };
    const otherAgent = { ...agent, span_id: "other" };
    const first = { ...root, span_id: "first", parent_span_id: "agent", type: "llm", start_offset_ms: 1 } as Span;
    const second = { ...first, span_id: "second", parent_span_id: "other", start_offset_ms: 2 };
    const framework = { ...first, span_id: "framework", type: "framework" } as Span;
    const spans = [root, agent, otherAgent, first, second, framework];
    expect(conversationSteps(spans).map((span) => span.span_id)).toEqual(["root", "agent", "other", "first", "second"]);
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [], [])],
      ["other", detail("other", [], [])],
      ["first", detail("first", [user], [answer])],
      ["second", detail("second", [user], [answer])],
    ]);
    expect(buildConversation(spans, details, true).flatMap((item) => item.messages)).toEqual([
      user,
      answer,
      user,
      answer,
    ]);
  });

  it("defers the root's final answer until all sections have been loaded", () => {
    const details = new Map([["root", detail("root", [user], [answer])]]);
    expect(buildConversation([root], details, false).flatMap((item) => item.messages)).toEqual([user]);
    expect(buildConversation([root], details, true).flatMap((item) => item.messages)).toEqual([user, answer]);
  });

  it("pairs typed tool arguments without coercing strings that resemble JSON", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const tool = { ...model, span_id: "tool", name: "lookup", type: "tool", start_offset_ms: 2 } as Span;
    const args = { order: 42, active: true, filters: { tags: ["paid"] }, empty: null, label: "42", text: "true" };
    const typedCall = { ...call, tool_calls: [{ name: "lookup", args }] };
    const toolDetail: SpanDetail = {
      ...detail("tool", args, "Shipped"),
      input_ui: {
        kind: "fields",
        fields: Object.entries(args).map(([key, value]) => ({
          key,
          value: typeof value === "string" ? value : JSON.stringify(value),
        })),
      },
    };
    const details = new Map([
      ["root", detail("root", [user], [])],
      ["model", detail("model", [user], [typedCall])],
      ["tool", toolDetail],
    ]);
    const items = buildConversation([root, model, tool], details, true);
    expect(items.flatMap((item) => item.messages.flatMap((message) => message.tool_calls ?? []))).toEqual([]);
    expect(items.find((item) => item.toolCall)?.toolCall?.args).toEqual(args);

    const differentType = { ...typedCall, tool_calls: [{ name: "lookup", args: { ...args, order: "42" } }] };
    details.set("model", detail("model", [user], [differentType]));
    const unmatched = buildConversation([root, model, tool], details, true);
    expect(unmatched.flatMap((item) => item.messages.flatMap((message) => message.tool_calls ?? []))).toEqual(
      differentType.tool_calls,
    );
  });

  it("loads child agents and places their own answers after their tools", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const tool = {
      ...agent,
      span_id: "tool",
      parent_span_id: "agent",
      type: "tool",
      start_offset_ms: 2,
      duration_ms: 2,
    } as Span;
    const next = { ...tool, span_id: "next", parent_span_id: "root", type: "llm", start_offset_ms: 30 } as Span;
    const final = { role: "assistant", content: "Finished the whole run" };
    const details = new Map([
      ["root", detail("root", [], [final])],
      ["agent", detail("agent", [user], [answer])],
      ["tool", detail("tool", { order: 42 }, "Shipped")],
      ["next", detail("next", [], [final])],
    ]);
    const spans = [parent, agent, tool, next];
    expect(conversationSteps(spans)).toContain(agent);
    const items = buildConversation(spans, details, true);
    expect(items.map((item) => item.id)).toEqual(["agent", "tool", "agent-output", "next"]);
    expect(items.flatMap((item) => item.messages)).toEqual([user, answer, final]);

    details.delete("next");
    expect(buildConversation(spans, details, false).flatMap((item) => item.messages)).toEqual([user, answer]);
    details.delete("tool");
    expect(buildConversation(spans, details, false).flatMap((item) => item.messages)).toEqual([user]);
  });

  it("does not repeat a child agent answer already recorded by its model", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const model = {
      ...agent,
      span_id: "model",
      parent_span_id: "agent",
      type: "llm",
      start_offset_ms: 2,
      duration_ms: 2,
    } as Span;
    const spans = [parent, agent, model];
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [user], [answer])],
      ["model", detail("model", [user], [answer])],
    ]);
    expect(buildConversation(spans, details, true).flatMap((item) => item.messages)).toEqual([user, answer]);
  });

  it("stops at a missing step even if later details and completion are supplied", () => {
    const first = { ...root, span_id: "first", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const missing = { ...first, span_id: "missing", start_offset_ms: 2 };
    const last = { ...first, span_id: "last", start_offset_ms: 3 };
    const details = new Map([
      ["root", detail("root", [user], [answer])],
      ["first", detail("first", [user], [call])],
      ["last", detail("last", [user, call, result], [answer])],
    ]);
    expect(buildConversation([root, first, missing, last], details, true).flatMap((item) => item.messages)).toEqual([
      user,
      call,
    ]);
  });

  it("keeps nested and parallel agent answers in their own histories", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const nested = { ...agent, span_id: "nested", parent_span_id: "agent", start_offset_ms: 2, duration_ms: 5 };
    const other = { ...agent, span_id: "other", start_offset_ms: 4, duration_ms: 30 };
    const tool = {
      ...nested,
      span_id: "tool",
      parent_span_id: "nested",
      type: "tool",
      start_offset_ms: 3,
      duration_ms: 1,
    } as Span;
    const otherTool = { ...tool, span_id: "other-tool", parent_span_id: "other", start_offset_ms: 5 };
    const summary = { role: "assistant", content: "Nested work complete" };
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [], [summary])],
      ["nested", detail("nested", [], [answer])],
      ["other", detail("other", [], [answer])],
      ["tool", detail("tool", {}, "First result")],
      ["other-tool", detail("other-tool", {}, "Second result")],
    ]);
    const items = buildConversation([parent, agent, nested, tool, other, otherTool], details, true);
    expect(items.map((item) => item.id)).toEqual([
      "tool",
      "other-tool",
      "nested-output",
      "agent-output",
      "other-output",
    ]);
    expect(items.flatMap((item) => item.messages)).toEqual([answer, summary, answer]);
  });

  it("renders a tool-only trace as one exchange", () => {
    const tool = { ...root, type: "tool" } as Span;
    const items = buildConversation([tool], new Map([["root", detail("root", { order: 42 }, "Shipped")]]), true);
    expect(items).toHaveLength(1);
    expect(items[0].toolResult).toBe("Shipped");
    expect(items[0].messages).toEqual([]);
  });

  it("removes forwarded child answers from ancestors while preserving identical answers in sibling branches", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const nested = { ...agent, span_id: "nested", parent_span_id: "agent", start_offset_ms: 2, duration_ms: 5 };
    const other = { ...agent, span_id: "other", start_offset_ms: 4, duration_ms: 30 };
    const spans = [parent, agent, nested, other];
    const details = new Map(spans.map((span) => [span.span_id, detail(span.span_id, [], [answer])]));
    const items = buildConversation(spans, details, true);
    expect(items.map((item) => item.id)).toEqual(["nested-output", "other-output"]);
    expect(items.flatMap((item) => item.messages)).toEqual([answer, answer]);
  });
});
