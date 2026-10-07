import { z } from "zod";

import type { TraceMessage } from "../../types";
import { toolSummary } from "../content/payload";
import type { ConversationGroup, ConversationItem } from "./conversation";
import type { ThreadTurn, ThreadWork } from "./thread";

export const READABLE_MODEL = "fireworks_ai/deepseek-v4-pro";

const PROMPT_CHARS = 2_000;
const REPLY_CHARS = 3_000;
const STEP_CHARS = 240;
const STEPS_PER_TURN = 40;

export const ReadableStepSchema = z.object({
  span_id: z.string().describe("span_id of the step this label summarizes, copied from the input"),
  label: z.string().describe("Short past-tense label, e.g. 'Read CHANGELOG.md' or 'Ran the test suite (2 failed)'"),
  status: z.enum(["ok", "error"]),
});

const readableTurnShape = {
  turn_id: z.string().describe("turn_id copied from the input"),
  user: z
    .string()
    .describe("The user's request with harness noise removed, as markdown. Empty string to keep the original"),
  summary: z.string().describe("One sentence on what the agent did this turn. Empty string if it did no work"),
  steps: z.array(ReadableStepSchema).describe("At most 6 labels covering the turn's work, in order"),
  reply: z
    .string()
    .describe("The assistant reply reformatted as clean markdown. Empty string to keep the original as is"),
};

export const ReadableTurnSchema = z.object(readableTurnShape);

export const ReadableThreadSchema = z.object({
  title: z.string().describe("A short title for the whole conversation"),
  turns: z.array(ReadableTurnSchema),
});

export type ReadableStep = z.infer<typeof ReadableStepSchema>;
export type ReadableTurn = z.infer<typeof ReadableTurnSchema>;
export type ReadableThread = z.infer<typeof ReadableThreadSchema>;

interface DigestStep {
  span_id: string;
  kind: "model" | "tool" | "subagent";
  name: string;
  detail?: string;
  result?: string;
  error?: true;
}

export interface DigestTurn {
  turn_id: string;
  user: string;
  steps: DigestStep[];
  omitted_steps?: number;
  reply: string;
}

function clip(text: string, max: number): string {
  const flat = text.trim();
  return flat.length > max ? `${flat.slice(0, max)}… [${flat.length - max} more chars, use read_step]` : flat;
}

const textOf = (messages: readonly TraceMessage[]): string => messages.map((message) => message.content).join("\n\n");

function itemStep(item: ConversationItem): DigestStep {
  const error = item.span.status === "error" ? ({ error: true } as const) : {};
  if (item.toolCall)
    return {
      span_id: item.span.span_id,
      kind: "tool",
      name: item.toolCall.name,
      detail: clip(toolSummary(item.toolCall.args) || JSON.stringify(item.toolCall.args ?? ""), STEP_CHARS),
      result: clip(item.toolResult ?? "", STEP_CHARS),
      ...error,
    };
  const said = item.messages.filter((message) => message.role === "assistant");
  const calls = said.flatMap((message) => message.tool_calls?.map((call) => call.name) ?? []);
  const detail = [textOf(said), calls.length ? `calls: ${calls.join(", ")}` : ""].filter(Boolean).join(" | ");
  return {
    span_id: item.span.span_id,
    kind: "model",
    name: item.model || item.span.name,
    detail: clip(detail, STEP_CHARS),
    ...error,
  };
}

function flatItems(groups: readonly ConversationGroup[]): ConversationItem[] {
  return groups.flatMap((group) => (group.kind === "item" ? [group.item] : flatItems(group.children)));
}

function workSteps(work: ThreadWork): DigestStep[] {
  if (work.kind === "step") return [itemStep(work.item)];
  const items = flatItems(work.groups);
  const tools = items.flatMap((item) => (item.toolCall ? [item.toolCall.name] : []));
  return [
    {
      span_id: work.id,
      kind: "subagent",
      name: work.name,
      detail: clip(
        `${items.length} steps${tools.length ? `, tools: ${[...new Set(tools)].join(", ")}` : ""}`,
        STEP_CHARS,
      ),
      ...(items.some((item) => item.span.status === "error") ? { error: true as const } : {}),
    },
  ];
}

export function threadDigest(turns: readonly ThreadTurn[]): DigestTurn[] {
  return turns.map((turn) => {
    const steps = turn.work.flatMap(workSteps);
    return {
      turn_id: turn.id,
      user: clip(textOf(turn.prompt), PROMPT_CHARS),
      steps: steps.slice(0, STEPS_PER_TURN),
      ...(steps.length > STEPS_PER_TURN ? { omitted_steps: steps.length - STEPS_PER_TURN } : {}),
      reply: clip(turn.reply?.content ?? "", REPLY_CHARS),
    };
  });
}

export function validReadableTurns(
  thread: ReadableThread,
  turns: readonly ThreadTurn[],
  spanIds: ReadonlySet<string>,
): Map<string, ReadableTurn> {
  const known = new Set(turns.map((turn) => turn.id));
  return new Map(
    thread.turns
      .filter((turn) => known.has(turn.turn_id))
      .map((turn) => [turn.turn_id, { ...turn, steps: turn.steps.filter((step) => spanIds.has(step.span_id)) }]),
  );
}
