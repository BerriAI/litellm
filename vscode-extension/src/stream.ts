import type { ChatCompletionChunk } from "openai/resources/chat/completions";

export type ResponsePart =
  | { readonly kind: "text"; readonly value: string }
  | { readonly kind: "tool_call"; readonly callId: string; readonly name: string; readonly input: object }
  | { readonly kind: "invalid_tool_call"; readonly callId: string; readonly name: string; readonly arguments: string }
  | { readonly kind: "truncated" };

interface PendingToolCall {
  readonly index: number;
  readonly callId: string;
  readonly name: string;
  readonly arguments: string;
}

export type PendingToolCalls = readonly PendingToolCall[];

export interface ChunkOutcome {
  readonly pending: PendingToolCalls;
  readonly parts: readonly ResponsePart[];
}

export const NO_PENDING_TOOL_CALLS: PendingToolCalls = [];

type ToolCallDelta = NonNullable<ChatCompletionChunk.Choice.Delta["tool_calls"]>[number];

const nonEmpty = (value: string | undefined): string | undefined => (value === undefined || value === "" ? undefined : value);

const targetOf = (pending: PendingToolCalls, delta: ToolCallDelta): PendingToolCall | undefined => {
  const id = nonEmpty(delta.id);
  if (id !== undefined) {
    return pending.find((call) => call.callId === id);
  }
  const sameIndex = pending.filter((call) => call.index === delta.index);
  return sameIndex.at(-1) ?? (delta.index === undefined ? pending.at(-1) : undefined);
};

const mergeToolCallDelta = (pending: PendingToolCalls, delta: ToolCallDelta): PendingToolCalls => {
  const target = targetOf(pending, delta);
  const base: PendingToolCall = target ?? { index: delta.index ?? pending.length, callId: nonEmpty(delta.id) ?? "", name: "", arguments: "" };
  const merged: PendingToolCall = {
    ...base,
    name: nonEmpty(delta.function?.name) ?? base.name,
    arguments: base.arguments + (delta.function?.arguments ?? ""),
  };
  return target === undefined ? [...pending, merged] : pending.map((call) => (call === target ? merged : call));
};

export const applyChunk = (pending: PendingToolCalls, chunk: ChatCompletionChunk): ChunkOutcome => {
  const choice = chunk.choices[0];
  if (choice === undefined) {
    return { pending, parts: [] };
  }
  const text = typeof choice.delta.content === "string" && choice.delta.content !== "" ? [{ kind: "text", value: choice.delta.content } as const] : [];
  const truncated = choice.finish_reason === "length" ? [{ kind: "truncated" } as const] : [];
  const nextPending = (choice.delta.tool_calls ?? []).reduce(mergeToolCallDelta, pending);
  return { pending: nextPending, parts: [...text, ...truncated] };
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

export async function* responseParts(chunks: AsyncIterable<ChatCompletionChunk>): AsyncGenerator<ResponsePart> {
  let pending: PendingToolCalls = NO_PENDING_TOOL_CALLS;
  for await (const chunk of chunks) {
    const outcome = applyChunk(pending, chunk);
    pending = outcome.pending;
    yield* outcome.parts;
  }
  yield* flushToolCalls(pending);
}
