import { describe, expect, it } from "vitest";

import { fieldNode, payloadView, textFormat } from "./payload";

describe("textFormat", () => {
  it.each([
    ["StopEvent(result=AgentOutput(response='hi'))", "code"],
    ["<workflows.context.Context object at 0x1>", "code"],
    ["Traceback (most recent call last):\n  File x", "code"],
    ["An **agent trace** is a record", "markdown"],
    ["# Findings\nnone", "markdown"],
    ["Steps:\n- read\n- write", "markdown"],
    ["see `answer.md`", "markdown"],
    ["read [the post](https://x.test)", "markdown"],
    ["What is an agent trace?", "plain"],
    ["calls f (twice)", "plain"],
  ])("reads %j as %s", (text, format) => {
    expect(textFormat(text)).toBe(format);
  });
});

describe("fieldNode", () => {
  it("unfolds JSON-encoded strings into a nested tree", () => {
    expect(fieldNode('{"config": {"steps": [1, null]}, "ok": true}')).toEqual({
      kind: "object",
      entries: [
        [
          "config",
          {
            kind: "object",
            entries: [
              [
                "steps",
                {
                  kind: "array",
                  items: [
                    { kind: "scalar", text: "1" },
                    { kind: "scalar", text: "null" },
                  ],
                },
              ],
            ],
          },
        ],
        ["ok", { kind: "scalar", text: "true" }],
      ],
    });
  });

  it("keeps a string that only starts like JSON as text", () => {
    expect(fieldNode("[not json")).toEqual({ kind: "text", text: "[not json", format: "plain" });
  });

  it("reads a nested message list as a conversation, but not an empty list", () => {
    const node = fieldNode([{ type: "human", data: { content: "What is an agent trace?" } }]);
    expect(node).toEqual({ kind: "messages", messages: [{ role: "user", content: "What is an agent trace?" }] });
    expect(fieldNode([])).toEqual({ kind: "array", items: [] });
  });
});

describe("payloadView", () => {
  it("renders JSON fields as a tree with nested values unfolded", () => {
    const raw = JSON.stringify({ start_event: "AgentWorkflowStartEvent()", tags: { run_id: "r1" } });
    expect(payloadView(raw, false)).toEqual({
      kind: "fields",
      entries: [
        ["start_event", { kind: "text", text: "AgentWorkflowStartEvent()", format: "code" }],
        ["tags", { kind: "object", entries: [["run_id", { kind: "text", text: "r1", format: "plain" }]] }],
      ],
    });
  });

  it("reads raw text by its format", () => {
    expect(payloadView("An **answer**", false)).toEqual({ kind: "text", text: "An **answer**", format: "markdown" });
    expect(payloadView("raw text", false)).toEqual({ kind: "text", text: "raw text", format: "plain" });
  });

  it("shows a tool's output as its result", () => {
    expect(payloadView(JSON.stringify([{ role: "tool", content: "denied" }]), true)).toEqual({
      kind: "tool-result",
      text: "denied",
    });
    expect(payloadView("42", true)).toEqual({ kind: "tool-result", text: "42" });
    expect(payloadView('{"a":1}', true)).toEqual({ kind: "tool-result", text: '{"a":1}' });
  });

  it("keeps a tool's message with calls as a conversation", () => {
    const raw = JSON.stringify({
      role: "assistant",
      content: "",
      tool_calls: [{ function: { name: "f", arguments: '{"x":1}' } }],
    });
    expect(payloadView(raw, true)).toEqual({
      kind: "messages",
      messages: [{ role: "assistant", content: "", tool_calls: [{ name: "f", args: { x: 1 } }] }],
    });
  });

  it("classifies raw messages, objects, lists and code", () => {
    expect(payloadView('[{"role":"user","content":"hi"}]', false)).toEqual({
      kind: "messages",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(payloadView('{"file":"/tmp/x"}', false)).toEqual({
      kind: "fields",
      entries: [["file", { kind: "text", text: "/tmp/x", format: "plain" }]],
    });
    expect(payloadView("[1,2]", false)).toEqual({
      kind: "fields",
      entries: [
        [
          "value",
          {
            kind: "array",
            items: [
              { kind: "scalar", text: "1" },
              { kind: "scalar", text: "2" },
            ],
          },
        ],
      ],
    });
    expect(payloadView("{}", false)).toEqual({ kind: "text", text: "{}", format: "plain" });
    expect(payloadView("StopEvent(result=1)", false)).toEqual({
      kind: "text",
      text: "StopEvent(result=1)",
      format: "code",
    });
  });
});
