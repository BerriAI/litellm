import type { TraceMessage } from "../../types";
import type { ConversationGroup, ConversationItem } from "./conversation";

export type ThreadWork =
  | { kind: "step"; item: ConversationItem }
  | { kind: "subagent"; id: string; name: string; groups: readonly ConversationGroup[] };

export interface ThreadTurn {
  id: string;
  prompt: readonly TraceMessage[];
  context: readonly TraceMessage[];
  work: readonly ThreadWork[];
  reply: TraceMessage | null;
  replyItem: ConversationItem | null;
  startMs: number;
  endMs: number;
  llmCalls: number;
  toolCalls: number;
  failed: boolean;
}

interface Piece {
  work: ThreadWork;
  start: number;
  end: number;
  messages: readonly TraceMessage[];
}

const isPrompt = (message: TraceMessage): boolean => message.role === "user";
const isContext = (message: TraceMessage): boolean => message.role === "system";
const isReply = (message: TraceMessage): boolean =>
  message.role === "assistant" && Boolean(message.content.trim()) && !message.tool_calls?.length;

function itemSpanMs(item: ConversationItem): { start: number; end: number } {
  const start = item.time ?? item.span.start_offset_ms;
  return { start, end: Math.max(start, item.span.start_offset_ms + item.span.duration_ms) };
}

function flatItems(groups: readonly ConversationGroup[]): ConversationItem[] {
  return groups.flatMap((group) => (group.kind === "item" ? [group.item] : flatItems(group.children)));
}

function piece(group: ConversationGroup): Piece {
  if (group.kind === "item")
    return { work: { kind: "step", item: group.item }, ...itemSpanMs(group.item), messages: group.item.messages };
  const items = flatItems(group.children);
  const times = items.map(itemSpanMs);
  return {
    work: { kind: "subagent", id: group.id, name: group.name, groups: group.children },
    start: Math.min(...times.map((t) => t.start), Infinity),
    end: Math.max(...times.map((t) => t.end), -Infinity),
    messages: [],
  };
}

function countWork(work: readonly ThreadWork[]): { llmCalls: number; toolCalls: number; failed: boolean } {
  const items = work.flatMap((w) => (w.kind === "step" ? [w.item] : flatItems(w.groups)));
  return {
    llmCalls: items.filter((item) => item.span.type === "llm").length,
    toolCalls: items.filter((item) => item.toolResult !== undefined).length,
    failed: items.some((item) => item.span.status === "error"),
  };
}

function stripMessages(item: ConversationItem, keep: (message: TraceMessage) => boolean): ConversationItem | null {
  const messages = item.messages.filter(keep);
  const empty = !messages.length && item.toolResult === undefined && !item.showError;
  return empty ? null : { ...item, messages };
}

interface TurnHead {
  id: string;
  prompt: readonly TraceMessage[];
  context: readonly TraceMessage[];
  promptStart: number;
}

function lastReply(pieces: readonly Piece[]): { item: ConversationItem; message: TraceMessage } | null {
  const direct = pieces.flatMap((p) => (p.work.kind === "step" ? [p.work.item] : []));
  const nested = pieces.flatMap((p) => (p.work.kind === "subagent" ? flatItems(p.work.groups) : []));
  const latest = (items: readonly ConversationItem[]) =>
    items
      .filter((i) => i.messages.some(isReply))
      .reduce<
        ConversationItem | undefined
      >((best, i) => (!best || itemSpanMs(i).end >= itemSpanMs(best).end ? i : best), undefined);
  const item = direct.findLast((i) => i.messages.some(isReply)) ?? latest(nested);
  const message = item?.messages.findLast(isReply);
  return item && message ? { item, message } : null;
}

function closeTurn({ id, prompt, context, promptStart }: TurnHead, pieces: readonly Piece[]): ThreadTurn {
  const found = lastReply(pieces);
  const work = pieces.flatMap((p): ThreadWork[] => {
    if (p.work.kind !== "step" || p.work.item !== found?.item) return [p.work];
    const leftover = stripMessages(p.work.item, (message) => message !== found.message);
    return leftover ? [{ kind: "step", item: leftover }] : [];
  });
  const timed = pieces.filter((p) => Number.isFinite(p.start));
  return {
    id,
    prompt,
    context,
    work,
    reply: found?.message ?? null,
    replyItem: found?.item ?? null,
    startMs: Math.min(promptStart, ...timed.map((p) => p.start)),
    endMs: Math.max(...timed.map((p) => p.end), -Infinity),
    ...countWork(work),
  };
}

export function buildThread(groups: readonly ConversationGroup[]): ThreadTurn[] {
  const pieces = groups.map(piece);
  const prompts = pieces.map((p) =>
    p.messages
      .filter(isPrompt)
      .map((m) => m.content)
      .join("\n"),
  );
  const answeredBefore = pieces.map((_, index) =>
    pieces.slice(0, index).findLastIndex((p) => p.messages.some(isReply)),
  );
  const repeatsPrompt = (index: number): boolean => {
    const previous = prompts.slice(0, index).findLastIndex(Boolean);
    return previous >= 0 && prompts[previous] === prompts[index] && answeredBefore[index] < previous;
  };
  const promptStarts = prompts.flatMap((text, index) => (text && !repeatsPrompt(index) ? [index] : []));
  const starts = promptStarts[0] === 0 ? promptStarts : [0, ...promptStarts];
  const repeats = new Set(prompts.flatMap((text, index) => (text && repeatsPrompt(index) ? [index] : [])));
  const stripRepeat = (p: Piece, index: number): Piece[] => {
    if (!repeats.has(index) || p.work.kind !== "step") return [p];
    const rest = stripMessages(p.work.item, (m) => !isPrompt(m) && !isContext(m));
    return rest ? [{ ...p, work: { kind: "step", item: rest }, messages: rest.messages }] : [];
  };
  return starts.flatMap((start, index) => {
    const end = starts[index + 1] ?? pieces.length;
    const head = pieces[start];
    if (!head) return [];
    const headItem = head.work.kind === "step" ? head.work.item : null;
    const prompt = head.messages.filter(isPrompt);
    const repeatContext = pieces
      .slice(start + 1, end)
      .flatMap((p, offset) => (repeats.has(start + 1 + offset) ? p.messages.filter(isContext) : []));
    const context = [...head.messages.filter(isContext), ...repeatContext];
    const rest = headItem ? stripMessages(headItem, (m) => !isPrompt(m) && !isContext(m)) : null;
    const body = [
      ...(head.work.kind === "subagent" ? [head] : []),
      ...(rest ? [{ ...head, work: { kind: "step" as const, item: rest }, messages: rest.messages }] : []),
      ...pieces.slice(start + 1, end).flatMap((p, offset) => stripRepeat(p, start + 1 + offset)),
    ];
    const promptStart = prompt.length ? head.start : Infinity;
    const turnHead: TurnHead = { id: headItem?.id ?? `turn-${start}`, prompt, context, promptStart };
    return [closeTurn(turnHead, body)];
  });
}

export function replyErrorSpanIds(turns: readonly ThreadTurn[]): ReadonlySet<string> {
  return new Set(turns.flatMap((turn) => (turn.replyItem?.showError ? [turn.replyItem.span.span_id] : [])));
}

export function withoutErrorsOf(turn: ThreadTurn, spanIds: ReadonlySet<string>): ThreadTurn {
  const work = turn.work.flatMap((w): ThreadWork[] => {
    if (w.kind !== "step" || !w.item.showError || !spanIds.has(w.item.span.span_id)) return [w];
    const item = { ...w.item, showError: false };
    return item.messages.length || item.toolResult !== undefined ? [{ kind: "step", item }] : [];
  });
  return { ...turn, work };
}

export function threadDurationMs(turn: ThreadTurn): number | null {
  return Number.isFinite(turn.startMs) && Number.isFinite(turn.endMs) ? turn.endMs - turn.startMs : null;
}
