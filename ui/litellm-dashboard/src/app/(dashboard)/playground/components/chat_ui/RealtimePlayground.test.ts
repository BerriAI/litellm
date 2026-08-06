import { describe, expect, it } from "vitest";
import { extractCompletedAssistantText, updateLastAssistantMessage } from "./RealtimePlayground";

describe("extractCompletedAssistantText", () => {
  it("prefers transcript content over plain text so dual streams stay one message", () => {
    const text = extractCompletedAssistantText({
      output: [
        {
          content: [
            { type: "output_text", text: "partial text" },
            { type: "audio_transcript", transcript: "full spoken answer" },
          ],
        },
      ],
    });
    expect(text).toBe("full spoken answer");
  });

  it("falls back to text when no transcript is present", () => {
    const text = extractCompletedAssistantText({
      output: [{ content: [{ type: "output_text", text: "hello" }] }],
    });
    expect(text).toBe("hello");
  });
});

describe("updateLastAssistantMessage", () => {
  it("updates the last assistant bubble even when a status message is after it", () => {
    const now = new Date();
    const next = updateLastAssistantMessage(
      [
        { role: "user", content: "hi", timestamp: now },
        { role: "assistant", content: "Hel", timestamp: now },
        { role: "status", content: "Connected", timestamp: now },
      ],
      "Hello world",
    );

    expect(next.map((m) => [m.role, m.content])).toEqual([
      ["user", "hi"],
      ["assistant", "Hello world"],
      ["status", "Connected"],
    ]);
  });

  it("creates a single assistant bubble when none exists yet", () => {
    const next = updateLastAssistantMessage([{ role: "user", content: "hi", timestamp: new Date() }], "Hello");
    expect(next.filter((m) => m.role === "assistant")).toHaveLength(1);
    expect(next[next.length - 1]).toMatchObject({ role: "assistant", content: "Hello" });
  });
});
