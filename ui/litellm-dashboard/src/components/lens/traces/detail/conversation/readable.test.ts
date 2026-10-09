import { describe, expect, it } from "vitest";

import type { Span, TraceMessage } from "../../types";
import type { ConversationItem } from "./conversation";
import { threadDigest, validReadableTurns } from "./readable";
import type { ThreadTurn } from "./thread";

const span = (id: string, type: Span["type"], status: Span["status"] = "ok") =>
  ({ span_id: id, name: id, type, status, start_offset_ms: 0, duration_ms: 1 }) as Span;

const user: TraceMessage = {
  role: "user",
  content: `<system-reminder>noise</system-reminder> Fix it ${"x".repeat(3_000)}`,
};
const toolStep: ConversationItem = {
  id: "tool",
  span: span("tool", "tool", "error"),
  messages: [],
  toolCall: { name: "bash", args: { command: "npm test" } },
  toolResult: "2 failed",
};
const modelStep: ConversationItem = {
  id: "llm",
  span: span("llm", "llm"),
  model: "gpt-x",
  messages: [{ role: "assistant", content: "Running tests", tool_calls: [{ name: "bash", args: {} }] }],
};
const turn: ThreadTurn = {
  id: "turn-1",
  prompt: [user],
  context: [],
  work: [
    { kind: "step", item: modelStep },
    { kind: "step", item: toolStep },
  ],
  reply: { role: "assistant", content: "Done" },
  replyItem: null,
  startMs: 0,
  endMs: 1,
  llmCalls: 1,
  toolCalls: 1,
  failed: true,
};

describe("threadDigest", () => {
  it("summarizes each step by span id and clips long prompts", () => {
    const [digest] = threadDigest([turn]);
    expect(digest.turn_id).toBe("turn-1");
    expect(digest.user.length).toBeLessThan(2_100);
    expect(digest.user).toContain("more chars, use read_step");
    expect(digest.steps).toEqual([
      { span_id: "llm", kind: "model", name: "gpt-x", detail: "Running tests | calls: bash" },
      { span_id: "tool", kind: "tool", name: "bash", detail: "npm test", result: "2 failed", error: true },
    ]);
    expect(digest.reply).toBe("Done");
  });
});

describe("validReadableTurns", () => {
  it("drops turns and step links that point at ids the trace does not have", () => {
    const thread = {
      title: "Fix",
      turns: [
        {
          turn_id: "turn-1",
          user: "Fix it",
          summary: "Ran the tests",
          steps: [
            { span_id: "tool", label: "Ran npm test (2 failed)", status: "error" as const },
            { span_id: "made-up", label: "Invented", status: "ok" as const },
          ],
          reply: "",
        },
        { turn_id: "turn-9", user: "", summary: "", steps: [], reply: "" },
      ],
    };
    const turns = validReadableTurns(thread, [turn], new Set(["llm", "tool"]));
    expect([...turns.keys()]).toEqual(["turn-1"]);
    expect(turns.get("turn-1")?.steps.map((step) => step.label)).toEqual(["Ran npm test (2 failed)"]);
  });
});
