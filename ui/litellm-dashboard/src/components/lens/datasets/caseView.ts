import { toTraceMessage } from "../traces/detail/content/payload";
import type { TraceMessage } from "../traces/types";
import { parseAssistantSummary, parseMessages } from "../traces/utils";
import type { DatasetCase, DatasetMessage } from "./types";

export interface CaseEdit {
  readonly included?: boolean;
  readonly expected?: string;
}

export type CaseEdits = Readonly<Record<string, CaseEdit>>;

export const editedCase = (item: DatasetCase, edit: CaseEdit | undefined): DatasetCase =>
  edit ? { ...item, included: edit.included ?? item.included, expected: edit.expected ?? item.expected } : item;

export const withEdits = (cases: readonly DatasetCase[], edits: CaseEdits): DatasetCase[] =>
  cases.map((item) => editedCase(item, edits[item.id]));

export const changedCount = (cases: readonly DatasetCase[], edits: CaseEdits): number =>
  cases.filter((item) => {
    const next = editedCase(item, edits[item.id]);
    return next.included !== item.included || next.expected !== item.expected;
  }).length;

export const shortCaseId = (id: string): string => id.slice(0, 8);

/** Text that is really a JSON message list or an assistant summary, decoded the way the trace view reads it. */
const decodedMessages = (text: string): TraceMessage[] | null => parseAssistantSummary(text) ?? parseMessages(text);

const previewText = (text: string): string =>
  decodedMessages(text)
    ?.map((message) => message.content)
    .filter(Boolean)
    .join(" ") ?? text;

const normalizedMessages = (message: DatasetMessage): TraceMessage[] => {
  const decoded = message.content ? decodedMessages(message.content) : null;
  if (!decoded) return [toTraceMessage(message)];
  const calls = message.tool_calls.length ? [toTraceMessage({ ...message, content: "" })] : [];
  return [...decoded, ...calls];
};

export interface CaseInput {
  readonly role: string;
  readonly text: string;
}

/** The turn that opened the conversation: the first user message, else the first message with any text. */
export const caseInput = (item: DatasetCase): CaseInput | null => {
  const opening =
    item.messages.find((message) => message.role === "user" && message.content) ??
    item.messages.find((message) => message.content);
  return opening ? { role: opening.role, text: previewText(opening.content) } : null;
};

export const caseReplyText = (item: DatasetCase): string => previewText(item.reply);

export const toolCallCount = (item: DatasetCase): number =>
  item.tool_calls.length + item.messages.reduce((total, message) => total + message.tool_calls.length, 0);

export const caseConversation = (item: DatasetCase): TraceMessage[] => item.messages.flatMap(normalizedMessages);

export const caseOutput = (item: DatasetCase): TraceMessage[] => {
  const reply: DatasetMessage = { role: "assistant", content: item.reply, name: "", tool_calls: item.tool_calls };
  return item.reply || item.tool_calls.length ? normalizedMessages(reply) : [];
};
