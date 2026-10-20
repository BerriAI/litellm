import { describe, expect, it } from "vitest";
import type * as vscode from "vscode";
import { SESSION_ID_HEADER, SessionTracker, conversationKeyOf } from "../src/sessions";

const userMessage = (text: string, name: string | undefined = undefined): vscode.LanguageModelChatRequestMessage =>
  ({ role: 1, content: [{ value: text }], name }) as vscode.LanguageModelChatRequestMessage;

const assistantMessage = (text: string): vscode.LanguageModelChatRequestMessage =>
  ({ role: 2, content: [{ value: text }], name: undefined }) as vscode.LanguageModelChatRequestMessage;

describe("SESSION_ID_HEADER", () => {
  it("is the header the LiteLLM proxy groups conversations by", () => {
    expect(SESSION_ID_HEADER).toBe("x-litellm-session-id");
  });
});

describe("conversationKeyOf", () => {
  it("matches requests that share the first user turn, even with later turns appended", () => {
    const firstTurn = [userMessage("fix the bug")];
    const secondTurn = [userMessage("fix the bug"), assistantMessage("on it"), userMessage("thanks")];
    expect(conversationKeyOf(secondTurn)).toBe(conversationKeyOf(firstTurn));
  });

  it("differs between conversations that start with a different first message", () => {
    expect(conversationKeyOf([userMessage("fix the bug")])).not.toBe(conversationKeyOf([userMessage("write tests")]));
  });

  it("differs between conversations that start with the same text but different user names", () => {
    expect(conversationKeyOf([userMessage("hi", "alice")])).not.toBe(conversationKeyOf([userMessage("hi", "bob")]));
  });

  it("ignores image bytes when fingerprinting the first user turn", () => {
    const withImage = (data: number): vscode.LanguageModelChatRequestMessage[] => [
      { role: 1, content: [{ value: "look" }, { mimeType: "image/png", data: new Uint8Array([data]) }], name: undefined } as vscode.LanguageModelChatRequestMessage,
    ];
    expect(conversationKeyOf(withImage(1))).toBe(conversationKeyOf(withImage(2)));
  });
});

describe("SessionTracker", () => {
  it("returns the same session id for every turn of one conversation", () => {
    const tracker = new SessionTracker();
    const first = tracker.sessionIdFor([userMessage("fix the bug")]);
    const followUp = tracker.sessionIdFor([userMessage("fix the bug"), assistantMessage("done"), userMessage("and now?")]);
    expect(followUp).toBe(first);
  });

  it("returns a different session id for a new conversation", () => {
    const tracker = new SessionTracker();
    const first = tracker.sessionIdFor([userMessage("fix the bug")]);
    const other = tracker.sessionIdFor([userMessage("write tests")]);
    expect(other).not.toBe(first);
  });

  it("generates unique ids that look like UUIDs", () => {
    const tracker = new SessionTracker();
    const ids = new Set([tracker.sessionIdFor([userMessage("a")]), tracker.sessionIdFor([userMessage("b")])]);
    expect(ids.size).toBe(2);
    for (const id of ids) {
      expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    }
  });
});
