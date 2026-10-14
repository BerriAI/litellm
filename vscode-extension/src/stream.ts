import type { ChatCompletionChunk } from "openai/resources/chat/completions";
import type { ResponseStreamEvent } from "openai/resources/responses/responses";

export type ResponsePart =
  | { readonly kind: "text"; readonly value: string }
  | { readonly kind: "tool_call"; readonly callId: string; readonly name: string; readonly input: object }
  | { readonly kind: "invalid_tool_call"; readonly callId: string; readonly name: string; readonly arguments: string }
  | { readonly kind: "truncated" };

interface PendingToolCall {
  readonly index: number;
  readonly callId: string;
  readonly itemId: string;
  readonly name: string;
  readonly arguments: string;
}

export type PendingToolCalls = readonly PendingToolCall[];

export interface ChunkOutcome {
  readonly pending: PendingToolCalls;
  readonly parts: readonly ResponsePart[];
}

export const NO_PENDING_TOOL_CALLS: PendingToolCalls = [];

type ChatToolCallDelta = NonNullable<ChatCompletionChunk.Choice.Delta["tool_calls"]>[number];

const nonEmpty = (value: string | undefined): string | undefined => (value === undefined || value === "" ? undefined : value);

const targetOfChat = (pending: PendingToolCalls, delta: ChatToolCallDelta): PendingToolCall | undefined => {
  const id = nonEmpty(delta.id);
  if (id !== undefined) {
    return pending.find((call) => call.callId === id);
  }
  const sameIndex = pending.filter((call) => call.index === delta.index);
  return sameIndex.at(-1) ?? (delta.index === undefined ? pending.at(-1) : undefined);
};

const mergeChatToolCallDelta = (pending: PendingToolCalls, delta: ChatToolCallDelta): PendingToolCalls => {
  const target = targetOfChat(pending, delta);
  const base: PendingToolCall = target ?? { index: delta.index ?? pending.length, callId: nonEmpty(delta.id) ?? "", itemId: "", name: "", arguments: "" };
  const merged: PendingToolCall = {
    ...base,
    name: nonEmpty(delta.function?.name) ?? base.name,
    arguments: base.arguments + (delta.function?.arguments ?? ""),
  };
  return target === undefined ? [...pending, merged] : pending.map((call) => (call === target ? merged : call));
};

const targetOfResponses = (pending: PendingToolCalls, itemId: string): PendingToolCall | undefined =>
  pending.find((call) => call.itemId === itemId);

const startResponsesToolCall = (pending: PendingToolCalls, item: { id?: string; call_id: string; name: string }, index: number): PendingToolCalls => [
  ...pending.filter((call) => call.itemId !== (item.id ?? item.call_id)),
  { index, callId: item.call_id, itemId: item.id ?? item.call_id, name: item.name, arguments: "" },
];

const updateResponsesToolCallArguments = (
  pending: PendingToolCalls,
  itemId: string,
  update: (current: string) => string,
): PendingToolCalls => {
  const target = targetOfResponses(pending, itemId);
  return target === undefined ? pending : pending.map((call) => (call === target ? { ...call, arguments: update(call.arguments) } : call));
};

const parseArguments = (raw: string): object | undefined => {
  if (raw.trim() === "") {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? parsed : undefined;
  } catch {
    return undefined;
  }
};

const finishToolCall = (call: PendingToolCall): ResponsePart => {
  const input = parseArguments(call.arguments);
  return input === undefined
    ? { kind: "invalid_tool_call", callId: call.callId, name: call.name, arguments: call.arguments }
    : { kind: "tool_call", callId: call.callId, name: call.name, input };
};

export const flushToolCalls = (pending: PendingToolCalls): readonly ResponsePart[] =>
  [...pending].sort((left, right) => left.index - right.index).map(finishToolCall);

export async function* responseParts(chunks: AsyncIterable<ChatCompletionChunk | ResponseStreamEvent>): AsyncGenerator<ResponsePart> {
  let pending: PendingToolCalls = NO_PENDING_TOOL_CALLS;
  for await (const chunk of chunks) {
    const outcome = applyChunk(pending, chunk);
    pending = outcome.pending;
    yield* outcome.parts;
  }
  yield* flushToolCalls(pending);
}

const applyChatChunk = (pending: PendingToolCalls, chunk: ChatCompletionChunk): ChunkOutcome => {
  const choice = chunk.choices?.[0];
  if (choice === undefined) {
    return { pending, parts: [] as const };
  }
  const text = typeof choice.delta.content === "string" && choice.delta.content !== "" ? [{ kind: "text" as const, value: choice.delta.content }] : [];
  const truncated = choice.finish_reason === "length" ? [{ kind: "truncated" as const }] : [];
  const nextPending = (choice.delta.tool_calls ?? []).reduce(mergeChatToolCallDelta, pending);
  return { pending: nextPending, parts: [...text, ...truncated] as const };
};

const applyResponsesEvent = (pending: PendingToolCalls, event: ResponseStreamEvent): ChunkOutcome => {
  const eventType = event.type;

  if (eventType === "response.output_text.delta") {
    const text = nonEmpty(event.delta);
    if (text !== undefined) {
      return { pending, parts: [{ kind: "text", value: text }] as const };
    }
    return { pending, parts: [] as const };
  }

  if (eventType === "response.output_text.done") {
    return { pending, parts: [] as const };
  }

  if (eventType === "response.function_call_arguments.delta") {
    return { pending: updateResponsesToolCallArguments(pending, event.item_id, (current) => current + event.delta), parts: [] as const };
  }

  if (eventType === "response.function_call_arguments.done") {
    return { pending: updateResponsesToolCallArguments(pending, event.item_id, () => event.arguments), parts: [] as const };
  }

  if (eventType === "response.output_item.added") {
    if (event.item.type === "function_call") {
      return { pending: startResponsesToolCall(pending, event.item, event.output_index), parts: [] as const };
    }
    return { pending, parts: [] as const };
  }

  if (eventType === "response.incomplete") {
    const truncated = event.response.incomplete_details?.reason === "max_output_tokens";
    return { pending, parts: truncated ? ([{ kind: "truncated" }] as const) : ([] as const) };
  }

  return { pending, parts: [] as const };
};

export const applyChunk = (pending: PendingToolCalls, chunk: ChatCompletionChunk | ResponseStreamEvent): ChunkOutcome => {
  if ("type" in chunk) {
    return applyResponsesEvent(pending, chunk as ResponseStreamEvent);
  }
  return applyChatChunk(pending, chunk as ChatCompletionChunk);
};
