import { describe, expect, it } from "vitest";

import { fieldNode, payloadView, textFormat, toolInput, toolSummary, toolResult } from "./payload";

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
  it("renders standard fields as a tree with JSON values unfolded", () => {
    const view = payloadView(
      "raw",
      {
        kind: "fields",
        fields: [
          { key: "start_event", value: "AgentWorkflowStartEvent()" },
          { key: "tags", value: '{"run_id": "r1"}' },
        ],
      },
      false,
    );
    expect(view).toEqual({
      kind: "fields",
      entries: [
        ["start_event", { kind: "text", text: "AgentWorkflowStartEvent()", format: "code" }],
        ["tags", { kind: "object", entries: [["run_id", { kind: "text", text: "r1", format: "plain" }]] }],
      ],
    });
  });

  it("reads standard text by its format and falls back to the raw text for empty fields", () => {
    expect(payloadView("", { kind: "text", text: "An **answer**" }, false)).toEqual({
      kind: "text",
      text: "An **answer**",
      format: "markdown",
    });
    expect(payloadView("raw text", { kind: "fields", fields: [] }, false)).toEqual({
      kind: "text",
      text: "raw text",
      format: "plain",
    });
  });

  it("shows a tool's output as its result whatever shape the store sent", () => {
    expect(payloadView("raw", { kind: "messages", messages: [{ role: "tool", content: "denied" }] }, true)).toEqual({
      kind: "tool-result",
      text: "denied",
    });
    expect(payloadView("raw", { kind: "text", text: "42" }, true)).toEqual({ kind: "tool-result", text: "42" });
    expect(payloadView('{"a": 1}', { kind: "fields", fields: [{ key: "a", value: "1" }] }, true)).toEqual({
      kind: "tool-result",
      text: '{"a": 1}',
    });
    expect(payloadView("plain output", undefined, true)).toEqual({ kind: "tool-result", text: "plain output" });
  });

  it("keeps a tool's message with calls as a conversation", () => {
    const view = payloadView(
      "raw",
      {
        kind: "messages",
        messages: [{ role: "assistant", content: "", tool_calls: [{ name: "f", arguments: '{"x": 1}' }] }],
      },
      true,
    );
    expect(view).toEqual({
      kind: "messages",
      messages: [{ role: "assistant", content: "", tool_calls: [{ name: "f", args: { x: 1 } }] }],
    });
  });

  it("classifies raw payloads: messages, objects, lists and text", () => {
    expect(payloadView('[{"role":"user","content":"hi"}]', undefined, false)).toEqual({
      kind: "messages",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(payloadView('{"file": "/tmp/x"}', undefined, false)).toEqual({
      kind: "fields",
      entries: [["file", { kind: "text", text: "/tmp/x", format: "plain" }]],
    });
    expect(payloadView("[1, 2]", undefined, false)).toEqual({
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
    expect(payloadView("{}", undefined, false)).toEqual({ kind: "text", text: "{}", format: "plain" });
    expect(payloadView("StopEvent(result=1)", undefined, false)).toEqual({
      kind: "text",
      text: "StopEvent(result=1)",
      format: "code",
    });
  });
});

describe("tool payloads", () => {
  it("decodes transport escapes once while preserving literal escapes inside a command", () => {
    const args = { command: "printf 'first\\nsecond'\nls src", workdir: "/workspace" };
    expect(toolInput(JSON.stringify(args))).toEqual(args);
    expect(toolSummary(JSON.stringify(args))).toBe("printf 'first\\nsecond' ls src");
  });

  it("extracts a useful action from a truncated tree preview without showing JSON scaffolding", () => {
    expect(toolSummary('{"command":"npm test\\n-- --run')).toBe("npm test -- --run");
    expect(toolSummary('{"file_path":"/workspace/src/page.tsx","offset":10}')).toBe("/workspace/src/page.tsx");
    expect(toolSummary('{"unknown":42}')).toBe("");
    expect(toolSummary({ command: "[ -f package.json ] && npm test" })).toBe("[ -f package.json ] && npm test");
    expect(toolSummary(JSON.stringify({ command: "{ npm test; }" }))).toBe("{ npm test; }");
  });

  it("preserves command failure metadata and renders structured results as fields", () => {
    expect(toolResult(JSON.stringify({ output: "first\nsecond", exit_code: 1, error: null }))).toEqual({
      body: { kind: "text", text: "first\nsecond", format: "plain" },
      metadata: [
        ["exit_code", { kind: "scalar", text: "1" }],
        ["error", { kind: "scalar", text: "null" }],
      ],
    });
    expect(toolResult('{"result":{"number":42,"state":"open"}}').body).toEqual(
      fieldNode({ result: { number: 42, state: "open" } }),
    );
  });

  it("unwraps MCP text while preserving unknown blocks, annotations and error flags", () => {
    expect(toolResult('{"content":[{"type":"text","text":"Permission denied"}],"isError":true}')).toEqual({
      body: fieldNode("Permission denied"),
      metadata: [["isError", { kind: "scalar", text: "true" }]],
    });
    const unknown = {
      content: [
        { type: "image", data: "sample" },
        { type: "text", text: "caption", annotations: { audience: ["user"] } },
      ],
    };
    expect(toolResult(JSON.stringify(unknown)).body).toEqual(fieldNode(unknown));
    expect(toolResult("{malformed output").body).toEqual(fieldNode("{malformed output"));
  });

  it("recognizes message arrays inside normalized text without rewriting ordinary agent content", () => {
    const raw = JSON.stringify([{ role: "user", content: "SAVED TASK RESUMED: Continue this task" }]);
    expect(payloadView(raw, { kind: "text", text: raw }, false)).toEqual({
      kind: "messages",
      messages: [{ role: "user", content: "SAVED TASK RESUMED: Continue this task" }],
    });
    const summary = JSON.stringify([{ content: "Checking files", tool_names: ["read_file"] }]);
    expect(payloadView(summary, { kind: "text", text: summary }, false, true)).toMatchObject({
      kind: "messages",
      messages: [
        { role: "assistant", content: "Checking files", tool_calls: [{ name: "read_file", args: undefined }] },
      ],
    });
    expect(payloadView(summary, { kind: "text", text: summary }, false).kind).toBe("text");
  });
});
