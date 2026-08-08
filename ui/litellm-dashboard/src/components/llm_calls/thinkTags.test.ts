import { describe, expect, it } from "vitest";
import { flushThinkTags, initialThinkTagState, splitThinkTags, ThinkTagState } from "./thinkTags";

function run(deltas: string[]): { text: string; reasoning: string } {
  const final = deltas.reduce<{ state: ThinkTagState; text: string; reasoning: string }>(
    (acc, delta) => {
      const split = splitThinkTags(acc.state, delta);
      return {
        state: split.state,
        text: acc.text + split.text,
        reasoning: acc.reasoning + split.reasoning,
      };
    },
    { state: initialThinkTagState, text: "", reasoning: "" },
  );
  const trailing = flushThinkTags(final.state);
  return { text: final.text + trailing.text, reasoning: final.reasoning + trailing.reasoning };
}

describe("splitThinkTags", () => {
  it("leaves plain text untouched", () => {
    expect(run(["Hello! ", "How can I help?"])).toEqual({ text: "Hello! How can I help?", reasoning: "" });
  });

  it("routes a whole think block to reasoning and keeps the answer as text", () => {
    expect(run(["<think>the user said hi</think>Hello!"])).toEqual({
      text: "Hello!",
      reasoning: "the user said hi",
    });
  });

  it("handles a think block spread across many deltas", () => {
    expect(run(["<think>", "the user ", "said hi", "</think>", "Hello!"])).toEqual({
      text: "Hello!",
      reasoning: "the user said hi",
    });
  });

  it("handles tags split mid-token across delta boundaries", () => {
    expect(run(["<thi", "nk>reasoning</thi", "nk>answer"])).toEqual({
      text: "answer",
      reasoning: "reasoning",
    });
  });

  it("never emits a partial tag as text before it is resolved", () => {
    const first = splitThinkTags(initialThinkTagState, "answer<thi");
    expect(first.text).toBe("answer");
    expect(first.reasoning).toBe("");

    const second = splitThinkTags(first.state, "nk>hidden</think>done");
    expect(second.text).toBe("done");
    expect(second.reasoning).toBe("hidden");
  });

  it("emits held-back text that turns out not to be a tag", () => {
    expect(run(["3 <", " 4"])).toEqual({ text: "3 < 4", reasoning: "" });
  });

  it("supports multiple think blocks in one response", () => {
    expect(run(["<think>a</think>one<think>b</think>two"])).toEqual({ text: "onetwo", reasoning: "ab" });
  });

  it("flushes an unterminated think block as reasoning", () => {
    expect(run(["<think>truncated reasoning"])).toEqual({ text: "", reasoning: "truncated reasoning" });
  });
});
