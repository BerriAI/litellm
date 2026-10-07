import { describe, expect, it } from "vitest";

import {
  caseConversation,
  caseInput,
  caseOutput,
  caseReplyText,
  changedCount,
  toolCallCount,
  withEdits,
} from "./caseView";
import type { DatasetCase } from "./types";

const source = { trace_id: "t", trace_ref: "", span_id: "", finding_id: "", lens_id: "" };
const makeCase = (id: string, overrides: Partial<DatasetCase> = {}): DatasetCase => ({
  id,
  messages: [{ role: "user", content: `question ${id}`, name: "", tool_calls: [] }],
  reply: `reply ${id}`,
  tool_calls: [],
  expected: "",
  included: true,
  source,
  agent_version: "",
  ...overrides,
});

describe("withEdits / changedCount", () => {
  it("applies only the edited fields and counts a case once however many of its fields changed", () => {
    const cases = [makeCase("a", { expected: "old" }), makeCase("b"), makeCase("c")];
    const edits = { a: { expected: "new", included: false }, b: { included: true }, c: { expected: "" } };
    expect(withEdits(cases, edits).map((item) => [item.id, item.expected, item.included])).toEqual([
      ["a", "new", false],
      ["b", "", true],
      ["c", "", true],
    ]);
    expect(changedCount(cases, edits)).toBe(1);
  });

  it("does not count an edit that matches the saved value", () => {
    expect(changedCount([makeCase("a")], { a: { included: true, expected: "" } })).toBe(0);
  });
});

describe("caseInput", () => {
  it("picks the first user turn over a system prompt that comes before it", () => {
    const item = makeCase("a", {
      messages: [
        { role: "system", content: "Be helpful", name: "", tool_calls: [] },
        { role: "user", content: "first ask", name: "", tool_calls: [] },
        { role: "user", content: "second ask", name: "", tool_calls: [] },
      ],
    });
    expect(caseInput(item)).toEqual({ role: "user", text: "first ask" });
  });

  it("falls back to the first message with text when there is no user turn", () => {
    const item = makeCase("a", { messages: [{ role: "system", content: "Be helpful", name: "", tool_calls: [] }] });
    expect(caseInput(item)).toEqual({ role: "system", text: "Be helpful" });
    expect(caseInput(makeCase("b", { messages: [] }))).toBeNull();
  });
});

describe("toolCallCount", () => {
  it("counts calls made during the conversation and in the reply", () => {
    const call = { name: "lookup", arguments: "{}" };
    const item = makeCase("a", {
      messages: [{ role: "assistant", content: "", name: "", tool_calls: [call, call] }],
      tool_calls: [call],
    });
    expect(toolCallCount(item)).toBe(3);
  });
});

describe("caseConversation / caseOutput", () => {
  it("parses tool call arguments into objects and keeps unparseable ones as text", () => {
    const item = makeCase("a", {
      messages: [
        { role: "assistant", content: "", name: "", tool_calls: [{ name: "lookup", arguments: '{"id": 42}' }] },
      ],
      reply: "",
      tool_calls: [{ name: "refund", arguments: "not json" }],
    });
    expect(caseConversation(item)[0].tool_calls).toEqual([{ name: "lookup", args: { id: 42 } }]);
    const reply = { role: "assistant", content: "", name: "", tool_calls: [{ name: "refund", args: "not json" }] };
    expect(caseOutput(item)).toEqual([reply]);
  });

  it("has no output when the agent neither replied nor called a tool", () => {
    expect(caseOutput(makeCase("a", { reply: "" }))).toEqual([]);
  });
});

describe("JSON text from older revisions", () => {
  const summaryReply = '[{"content":"The result is 5050.","tool_names":[]}]';

  it("reads an assistant summary reply as an assistant message in the panel and as plain text in the table", () => {
    const item = makeCase("a", { reply: summaryReply });
    expect(caseOutput(item)).toEqual([{ role: "assistant", content: "The result is 5050.", tool_calls: [] }]);
    expect(caseReplyText(item)).toBe("The result is 5050.");
  });

  it("keeps the recorded tool calls after a decoded reply", () => {
    const item = makeCase("a", { reply: summaryReply, tool_calls: [{ name: "sum", arguments: '{"n":100}' }] });
    expect(caseOutput(item).map((message) => [message.content, message.tool_calls])).toEqual([
      ["The result is 5050.", []],
      ["", [{ name: "sum", args: { n: 100 } }]],
    ]);
  });

  it("expands a user turn whose text is a JSON message list into those messages", () => {
    const content = JSON.stringify([{ role: "user", content: "Add 1 to 100" }]);
    const item = makeCase("a", { messages: [{ role: "user", content, name: "", tool_calls: [] }] });
    expect(caseConversation(item).map((message) => [message.role, message.content])).toEqual([
      ["user", "Add 1 to 100"],
    ]);
    expect(caseInput(item)).toEqual({ role: "user", text: "Add 1 to 100" });
  });

  it("leaves ordinary text untouched", () => {
    const item = makeCase("a", { reply: "Plain [not json] reply" });
    expect(caseReplyText(item)).toBe("Plain [not json] reply");
    expect(caseOutput(item)[0].content).toBe("Plain [not json] reply");
  });
});
