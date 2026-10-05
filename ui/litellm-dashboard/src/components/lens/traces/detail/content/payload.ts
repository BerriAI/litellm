import type { TraceMessage, UIContent, UIMessage } from "../../types";
import { parseJson, parseMessages } from "../../utils";

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
export function payloadView(raw: string, content: UIContent | undefined, toolOutput: boolean): PayloadView {
  return content ? standardView(content, raw, toolOutput) : rawView(raw, toolOutput);
}
