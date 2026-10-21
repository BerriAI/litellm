import { describe, expect, it } from "vitest";
import type { MessageType } from "@/components/chat_ui/types";
import { textHistory } from "./dispatchRequest";

const user = (content: string): MessageType => ({ role: "user", content });
const assistant = (content: string, extra: Partial<MessageType> = {}): MessageType => ({
  role: "assistant",
  content,
  ...extra,
});

describe("textHistory", () => {
  it("resends only answered text turns", () => {
    const history: MessageType[] = [
      user("first question"),
      assistant("first answer"),
      user("question that failed"),
      assistant("rate limited", { isError: true }),
      user("question that broke mid-stream"),
      assistant("half an ans"),
      assistant("connection reset", { isError: true }),
      user("question that was cancelled"),
      user("draw a cat"),
      assistant("data:image/png;base64,AAA", { isImage: true }),
      { role: "user", content: "", mcpToolCall: { tool: "search", arguments: {} } },
      assistant("tool output"),
      user("think hard"),
      assistant(""),
      user("think again"),
      assistant("  \n"),
      user("embed this"),
      assistant("[0.1, 0.2]", { isEmbeddings: true }),
      user("second question"),
      assistant("second answer"),
    ];

    expect(textHistory(history)).toEqual([
      { role: "user", content: "first question" },
      { role: "assistant", content: "first answer" },
      { role: "user", content: "second question" },
      { role: "assistant", content: "second answer" },
    ]);
  });

  it("drops a transcription turn that has no typed text", () => {
    expect(textHistory([user(""), assistant("transcribed words"), user("hi"), assistant("hello")])).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("drops a trailing user turn that never got a reply", () => {
    expect(textHistory([user("hi"), assistant("hello"), user("still waiting")])).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });
});
