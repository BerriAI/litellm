import type { TraceMessage, UIContent, UIMessage } from "../../types";
import { parseAssistantSummary, parseJson, parseMessages } from "../../utils";

export type TextFormat = "markdown" | "code" | "plain";

export type FieldNode =
  | { readonly kind: "object"; readonly entries: readonly FieldEntry[] }
  | { readonly kind: "array"; readonly items: readonly FieldNode[] }
  | { readonly kind: "messages"; readonly messages: readonly TraceMessage[] }
  | { readonly kind: "text"; readonly text: string; readonly format: TextFormat }
  | { readonly kind: "scalar"; readonly text: string };

export type FieldEntry = readonly [key: string, node: FieldNode];

export type PayloadView =
  | { readonly kind: "messages"; readonly messages: readonly TraceMessage[] }
  | { readonly kind: "tool-result"; readonly text: string }
  | { readonly kind: "fields"; readonly entries: readonly FieldEntry[] }
  | { readonly kind: "text"; readonly text: string; readonly format: TextFormat };

const MARKDOWN_SYNTAX = /(^|\n)\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|```)|\*\*[^*]+\*\*|`[^`\n]+`|\[[^\]]+\]\([^)]+\)/;
const CODE_LIKE = /^\s*(?:[A-Za-z_][\w.]*\(|<[\w.]+[\s>]|Traceback \(most recent call last\))/;

export function textFormat(text: string): TextFormat {
  if (CODE_LIKE.test(text)) return "code";
  if (MARKDOWN_SYNTAX.test(text)) return "markdown";
  return "plain";
}

const scalarText = (value: unknown): string => (value === null || value === undefined ? "null" : String(value));

const looksLikeJson = (text: string): boolean => /^\s*[[{]/.test(text);

export function fieldNode(value: unknown): FieldNode {
  if (typeof value === "string") {
    const parsed = looksLikeJson(value) ? parseJson(value) : null;
    if (typeof parsed === "object" && parsed !== null) return fieldNode(parsed);
    return { kind: "text", text: value, format: textFormat(value) };
  }
  if (Array.isArray(value)) {
    const messages = value.length > 0 ? parseMessages(JSON.stringify(value)) : null;
    if (messages) return { kind: "messages", messages };
    return { kind: "array", items: value.map(fieldNode) };
  }
  if (typeof value === "object" && value !== null) {
    return { kind: "object", entries: Object.entries(value).map(([key, v]): FieldEntry => [key, fieldNode(v)]) };
  }
  return { kind: "scalar", text: scalarText(value) };
}

export const fieldEntries = (pairs: readonly (readonly [string, string])[]): readonly FieldEntry[] =>
  pairs.map(([key, value]): FieldEntry => [key, fieldNode(value)]);

export const toTraceMessage = (message: UIMessage): TraceMessage => ({
  ...message,
  tool_calls: message.tool_calls?.map((call) => ({
    name: call.name,
    args: parseJson(call.arguments) ?? call.arguments,
  })),
});

const singleText = (messages: readonly UIMessage[]): string | null =>
  messages.length === 1 && !messages[0].tool_calls?.length ? messages[0].content : null;

function textView(text: string): PayloadView {
  const messages = parseMessages(text);
  if (messages?.length) return { kind: "messages", messages };
  return { kind: "text", text, format: textFormat(text) };
}

function standardView(content: UIContent, raw: string, toolOutput: boolean): PayloadView {
  switch (content.kind) {
    case "messages": {
      const toolText = toolOutput ? singleText(content.messages) : null;
      if (toolText !== null) return { kind: "tool-result", text: toolText };
      return { kind: "messages", messages: content.messages.map(toTraceMessage) };
    }
    case "fields":
      if (toolOutput) return { kind: "tool-result", text: raw };
      if (content.fields.length === 0) return textView(raw);
      return { kind: "fields", entries: fieldEntries(content.fields.map((field) => [field.key, field.value])) };
    case "text":
      return toolOutput ? { kind: "tool-result", text: content.text } : textView(content.text);
  }
}

function rawView(raw: string, toolOutput: boolean): PayloadView {
  const messages = parseMessages(raw);
  if (messages) return { kind: "messages", messages };
  if (toolOutput) return { kind: "tool-result", text: raw };
  const node = fieldNode(raw);
  if (node.kind === "object" && node.entries.length > 0) return { kind: "fields", entries: node.entries };
  if (node.kind === "array" || node.kind === "messages") return { kind: "fields", entries: [["value", node]] };
  return textView(raw);
}

/** How a span's input or output reads best: the standard UI shape when the store sent one, else the raw payload. */
export function payloadView(
  raw: string,
  content: UIContent | undefined,
  toolOutput: boolean,
  assistantOutput = false,
): PayloadView {
  if (assistantOutput && (!content || content.kind === "text")) {
    const messages = parseAssistantSummary(content?.text ?? raw);
    if (messages) return { kind: "messages", messages };
  }
  return content ? standardView(content, raw, toolOutput) : rawView(raw, toolOutput);
}

export function toolInput(raw: string, content?: UIContent): unknown {
  const parsed = parseJson(raw);
  if (parsed !== null) return parsed;
  if (content?.kind === "fields") return Object.fromEntries(content.fields.map(({ key, value }) => [key, value]));
  return content?.kind === "text" ? content.text : raw;
}

const ACTION_KEYS = ["command", "cmd", "code", "patch", "file_path", "path", "file", "query", "pattern", "url"];

export function toolAction(value: unknown): { key: string; text: string } | null {
  if (typeof value === "string") {
    const parsed = parseJson(value);
    if (parsed !== null && typeof parsed === "object") return toolAction(parsed);
    return value ? { key: "arguments", text: value } : null;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  for (const key of ACTION_KEYS) {
    const text: unknown = Reflect.get(value, key);
    if (typeof text === "string" && text) return { key, text };
  }
  return null;
}

export function toolSummary(raw: unknown): string {
  const action = toolAction(raw);
  if (action && (action.key !== "arguments" || !looksLikeJson(action.text))) return action.text.replace(/\s+/g, " ");
  if (typeof raw !== "string") return "";
  const match = /"(?:command|cmd|code|patch|file_path|path|file|query|pattern|url)"\s*:\s*"((?:[^"\\]|\\.)*)/.exec(raw);
  if (!match) return "";
  const text = match[1].replace(/\\u[0-9a-fA-F]{0,3}$|\\$/, "");
  const decoded = parseJson(`"${text}"`);
  return (typeof decoded === "string" ? decoded : text).replace(/\s+/g, " ");
}

export function toolResult(result: string): { body: FieldNode; metadata: readonly FieldEntry[] } {
  const parsed = parseJson(result);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const entries = Object.entries(parsed);
    const output = entries.find(([key, value]) => key === "output" && typeof value === "string");
    if (output)
      return {
        body: fieldNode(output[1]),
        metadata: entries.filter(([key]) => key !== "output").map(([key, value]) => [key, fieldNode(value)]),
      };
    const content: unknown = Reflect.get(parsed, "content");
    const isTextBlock = (block: unknown): block is { type: "text"; text: string } => {
      if (!block || typeof block !== "object") return false;
      const hasText = Reflect.get(block, "type") === "text" && typeof Reflect.get(block, "text") === "string";
      return hasText && Object.keys(block).every((key) => key === "type" || key === "text");
    };
    if (Array.isArray(content) && content.length && content.every(isTextBlock)) {
      return {
        body: fieldNode(content.map((block) => block.text).join("\n\n")),
        metadata: entries.filter(([key]) => key !== "content").map(([key, value]) => [key, fieldNode(value)]),
      };
    }
  }
  return { body: fieldNode(typeof parsed === "string" ? parsed : result), metadata: [] };
}
