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
      user("question that was cancelled"),
      user("draw a cat"),
      assistant("data:image/png;base64,AAA", { isImage: true }),
      user("second question"),
      assistant("second answer"),
    ];

    expect(textHistory(history)).toEqual([
      { role: "user", content: "first question" },
      { role: "assistant", content: "first answer" },
      { role: "user", content: "draw a cat" },
      { role: "user", content: "second question" },
      { role: "assistant", content: "second answer" },
    ]);
  });

  it("drops a trailing user turn that never got a reply", () => {
    expect(textHistory([user("hi"), assistant("hello"), user("still waiting")])).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });
});
