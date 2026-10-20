import { createHash, randomUUID } from "node:crypto";
import type * as vscode from "vscode";

export const SESSION_ID_HEADER = "x-litellm-session-id";

const textOf = (part: unknown): string => {
  if (part !== null && typeof part === "object") {
    const record = part as Record<string, unknown>;
    if (typeof record.value === "string") {
      return record.value;
    }
    if (typeof record.mimeType === "string" && record.data instanceof Uint8Array && record.mimeType.startsWith("text/")) {
      return Buffer.from(record.data).toString("utf8");
    }
    if ("value" in record) {
      return JSON.stringify(record.value);
    }
  }
  return "";
};

/**
 * Stable fingerprint of the first user turn of a conversation. Follow-up requests in the same
 * VS Code chat conversation repeat the first user message (plus later turns), so the fingerprint
 * is identical across turns, while a new conversation starts with a different first message.
 */
export const conversationKeyOf = (messages: readonly vscode.LanguageModelChatRequestMessage[]): string => {
  const firstUser = messages.find((message) => message.role === 1);
  const content = firstUser === undefined ? [] : firstUser.content.map(textOf).join("");
  return createHash("sha256").update(`${firstUser?.name ?? ""}\n${content}`).digest("hex");
};

export class SessionTracker {
  private readonly sessions = new Map<string, string>();

  sessionIdFor(messages: readonly vscode.LanguageModelChatRequestMessage[]): string {
    const key = conversationKeyOf(messages);
    const known = this.sessions.get(key);
    if (known !== undefined) {
      return known;
    }
    const sessionId = randomUUID();
    this.sessions.set(key, sessionId);
    return sessionId;
  }
}
