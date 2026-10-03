import { describe, expect, it } from "vitest";
import { buildConversation, conversationSteps, newConversationMessages } from "./conversation";
import type { Span, SpanDetail, TraceMessage } from "./traceTypes";
import research from "./__fixtures__/research_trace.json";

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
    const items = buildConversation([root, model], new Map([["model", normalized]]), true);
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
    expect(conversationSteps(spans).map((span) => span.span_id)).toEqual(["root", "first", "second"]);
    const details = new Map([
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
});
