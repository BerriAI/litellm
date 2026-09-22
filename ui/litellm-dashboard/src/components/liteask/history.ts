import type { LiteAskMessage } from "./api";

export function buildLiteAskHistory(messages: readonly LiteAskMessage[]): LiteAskMessage[] {
  return messages.slice(-39).reduceRight<{ remaining: number; messages: LiteAskMessage[] }>(
    (history, message) => {
      const content = message.content.slice(0, 8000);
      if (!content) return history;
      if (content.length > history.remaining) return { ...history, remaining: 0 };
      return {
        remaining: history.remaining - content.length,
        messages: [{ role: message.role, content }, ...history.messages],
      };
    },
    { remaining: 48_000, messages: [] },
  ).messages;
}
