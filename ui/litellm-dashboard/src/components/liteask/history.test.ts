import { describe, expect, it } from "vitest";
import { buildLiteAskHistory } from "./history";
import type { LiteAskMessage } from "./api";

describe("LiteAsk history limits", () => {
  it("keeps a complete recent suffix within the gateway's total character budget", () => {
    const older: LiteAskMessage[] = Array.from({ length: 6 }, (_, index) => ({
      role: "assistant",
      content: String(index).repeat(8000),
    }));
    const latest: LiteAskMessage = { role: "user", content: "New question" };
    const history = buildLiteAskHistory([...older, latest]);
    expect(history).toEqual([...older.slice(1), latest]);
    expect(history.reduce((total, message) => total + message.content.length, 0)).toBeLessThanOrEqual(48_000);
  });

  it("retains the exact 48,000-character boundary", () => {
    const messages: LiteAskMessage[] = Array.from({ length: 6 }, () => ({ role: "user", content: "x".repeat(8000) }));
    expect(buildLiteAskHistory(messages)).toEqual(messages);
  });

  it("bounds individual messages, strips extra data, and always retains the latest request", () => {
    const history = buildLiteAskHistory([
      { role: "assistant", content: "x".repeat(9000), generated_key: "private" } as LiteAskMessage,
      { role: "user", content: "Continue" },
    ]);
    expect(history).toEqual([
      { role: "assistant", content: "x".repeat(8000) },
      { role: "user", content: "Continue" },
    ]);
  });

  it("bounds the number of recent messages without mutating the original transcript", () => {
    const messages: LiteAskMessage[] = Array.from({ length: 45 }, (_, index) => ({
      role: "user",
      content: `Message ${index}`,
    }));
    expect(buildLiteAskHistory(messages)).toEqual(messages.slice(-39));
    expect(messages).toHaveLength(45);
  });
});
