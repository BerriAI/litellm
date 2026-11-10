import { describe, expect, it } from "vitest";
import type { ChatCompletionChunk } from "openai/resources/chat/completions";
import type { ResponseStreamEvent } from "openai/resources/responses/responses";
import { responseParts, type ResponsePart } from "../src/stream";

type ToolCallDelta = NonNullable<ChatCompletionChunk.Choice.Delta["tool_calls"]>[number];

const chunk = (delta: ChatCompletionChunk.Choice.Delta, finishReason: ChatCompletionChunk.Choice["finish_reason"] = null): ChatCompletionChunk => ({
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 0,
  model: "gpt-5.6",
  choices: [{ index: 0, delta, finish_reason: finishReason }],
});

const usageChunk: ChatCompletionChunk = {
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 0,
  model: "gpt-5.6",
  choices: [],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};

async function* stream(chunks: readonly ChatCompletionChunk[]): AsyncGenerator<ChatCompletionChunk> {
  yield* chunks;
}

const collect = async (chunks: readonly ChatCompletionChunk[]): Promise<readonly ResponsePart[]> => {
  const parts: ResponsePart[] = [];
  for await (const part of responseParts(stream(chunks))) {
    parts.push(part);
  }
  return parts;
};

describe("responseParts", () => {
  it("yields text deltas as they arrive and ignores empty and usage-only chunks", async () => {
    expect(await collect([chunk({ role: "assistant", content: "" }), chunk({ content: "Hel" }), chunk({ content: "lo" }), usageChunk])).toEqual([
      { kind: "text", value: "Hel" },
      { kind: "text", value: "lo" },
    ]);
  });

  it("assembles tool calls split across chunks and emits them after the text, in index order", async () => {
    expect(
      await collect([
        chunk({ content: "Looking" }),
        chunk({ tool_calls: [{ index: 1, id: "call_b", type: "function", function: { name: "grep", arguments: "" } }] }),
        chunk({ tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: "read_file", arguments: '{"pa' } }] }),
        chunk({ tool_calls: [{ index: 0, function: { name: "read_file", arguments: 'th":"a"}' } }] }),
        chunk({ tool_calls: [{ index: 1, function: { arguments: '{"q":"x"}' } }] }, "tool_calls"),
      ]),
    ).toEqual([
      { kind: "text", value: "Looking" },
      { kind: "tool_call", callId: "call_a", name: "read_file", input: { path: "a" } },
      { kind: "tool_call", callId: "call_b", name: "grep", input: { q: "x" } },
    ]);
  });

  it("starts a new call when a fresh id reuses an index and appends index-less deltas to the last call", async () => {
    expect(
      await collect([
        chunk({ tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: "grep", arguments: '{"q":' } }] }),
        chunk({ tool_calls: [{ function: { arguments: '"a"}' } } as ToolCallDelta] }),
        chunk({ tool_calls: [{ index: 0, id: "call_b", type: "function", function: { name: "grep", arguments: '{"q":"b"}' } }] }),
      ]),
    ).toEqual([
      { kind: "tool_call", callId: "call_a", name: "grep", input: { q: "a" } },
      { kind: "tool_call", callId: "call_b", name: "grep", input: { q: "b" } },
    ]);
  });

  it("flags a response cut off at the output token limit after the text it did produce", async () => {
    expect(await collect([chunk({ content: "half" }), chunk({}, "length"), usageChunk])).toEqual([
      { kind: "text", value: "half" },
      { kind: "truncated" },
    ]);
  });

  it("treats empty arguments as an empty object and flags malformed JSON", async () => {
    expect(
      await collect([
        chunk({ tool_calls: [{ index: 0, id: "call_0", type: "function", function: { name: "noop", arguments: "" } }] }),
        chunk({ tool_calls: [{ index: 1, id: "call_1", type: "function", function: { name: "bad", arguments: "{oops" } }] }),
        chunk({ tool_calls: [{ index: 2, id: "call_2", type: "function", function: { name: "scalar", arguments: "42" } }] }),
      ]),
    ).toEqual([
      { kind: "tool_call", callId: "call_0", name: "noop", input: {} },
      { kind: "invalid_tool_call", callId: "call_1", name: "bad", arguments: "{oops" },
      { kind: "invalid_tool_call", callId: "call_2", name: "scalar", arguments: "42" },
    ]);
  });
});

describe("responseParts with Responses API events", () => {
  const collectEvents = async (events: readonly object[]): Promise<readonly ResponsePart[]> => {
    async function* source(): AsyncGenerator<ResponseStreamEvent> {
      yield* events as readonly ResponseStreamEvent[];
    }
    const parts: ResponsePart[] = [];
    for await (const part of responseParts(source())) {
      parts.push(part);
    }
    return parts;
  };

  it("reports a function call under its call_id, not the output item id", async () => {
    expect(
      await collectEvents([
        { type: "response.output_item.added", output_index: 0, item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "read_file", arguments: "" } },
        { type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: 0, delta: '{"pa' },
        { type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: 0, delta: 'th":"a"}' },
        { type: "response.function_call_arguments.done", item_id: "fc_1", output_index: 0, arguments: '{"path":"a"}' },
        { type: "response.completed", response: {} },
      ]),
    ).toEqual([{ kind: "tool_call", callId: "call_1", name: "read_file", input: { path: "a" } }]);
  });

  it("keeps parallel calls apart and orders them by output index", async () => {
    expect(
      await collectEvents([
        { type: "response.output_item.added", output_index: 1, item: { type: "function_call", id: "fc_b", call_id: "call_b", name: "grep", arguments: "" } },
        { type: "response.output_item.added", output_index: 0, item: { type: "function_call", id: "fc_a", call_id: "call_a", name: "read_file", arguments: "" } },
        { type: "response.function_call_arguments.delta", item_id: "fc_a", output_index: 0, delta: '{"p":1}' },
        { type: "response.function_call_arguments.delta", item_id: "fc_b", output_index: 1, delta: '{"q":2}' },
      ]),
    ).toEqual([
      { kind: "tool_call", callId: "call_a", name: "read_file", input: { p: 1 } },
      { kind: "tool_call", callId: "call_b", name: "grep", input: { q: 2 } },
    ]);
  });

  it("streams output text and flags a response cut off at max_output_tokens", async () => {
    expect(
      await collectEvents([
        { type: "response.output_text.delta", delta: "half" },
        { type: "response.incomplete", response: { incomplete_details: { reason: "max_output_tokens" } } },
      ]),
    ).toEqual([{ kind: "text", value: "half" }, { kind: "truncated" }]);
  });

  it("does not flag an incomplete response that stopped for another reason", async () => {
    expect(await collectEvents([{ type: "response.incomplete", response: { incomplete_details: { reason: "content_filter" } } }])).toEqual([]);
  });
});
