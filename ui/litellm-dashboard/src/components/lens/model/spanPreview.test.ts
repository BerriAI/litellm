import { describe, expect, it } from "vitest";
import { spanPreviewLines } from "./spanPreview";

describe("span previews", () => {
  it("shows the last chat message of an agent span as role: content, not raw JSON", () => {
    const preview =
      'Input: [{"role": "system", "content": "You are a research assistant."}, {"role": "user", "content": "Find the median latency of the EU region."}]\nOutput: [{"role": "assistant", "content": "I could not find it."}]';
    expect(spanPreviewLines(preview)).toEqual([
      { label: "in", text: "user: Find the median latency of the EU region.", error: false },
      { label: "out", text: "assistant: I could not find it.", error: false },
    ]);
  });

  it("recovers the last message from truncated JSON with the omitted marker", () => {
    const preview =
      'Input: [{"role": "system", "content": "You are a bill\n[... preview omitted; read this span for evidence ...]\nn"}, {"role": "user", "content": "Cancel C2473\'s subscript';
    expect(spanPreviewLines(preview)).toEqual([
      { label: "in", text: "user: Cancel C2473's subscript", error: false },
    ]);
  });

  it("keeps tool payloads compact and drops an unset status", () => {
    const preview = 'Input: {"query":"latency"}\nOutput: {"results": []}\nStatus: STATUS_CODE_UNSET ';
    expect(spanPreviewLines(preview)).toEqual([
      { label: "in", text: '{"query":"latency"}', error: false },
      { label: "out", text: '{"results":[]}', error: false },
    ]);
  });

  it("marks a failed span's output and shows its error", () => {
    const preview =
      'Input: {"query":"q"}\nOutput: {"error": "500 search index unavailable"}\nStatus: STATUS_CODE_ERROR 500 search index unavailable';
    const lines = spanPreviewLines(preview);
    expect(lines.at(-1)).toEqual({ label: "error", text: "500 search index unavailable", error: true });
    expect(lines.find((l) => l.label === "out")?.error).toBe(true);
  });

  it("keeps multi-line payloads inside their section and collapses whitespace", () => {
    const preview = 'Output: {"content": "def total(items):\\n    return 1"}\nmore text\nStatus: STATUS_CODE_OK';
    expect(spanPreviewLines(preview)).toEqual([
      { label: "out", text: '{"content": "def total(items):\\n return 1"} more text', error: false },
    ]);
  });

  it("passes through plain text without sections", () => {
    expect(spanPreviewLines("500 Internal Server Error")).toEqual([
      { label: "", text: "500 Internal Server Error", error: false },
    ]);
    expect(spanPreviewLines("")).toEqual([]);
  });
});
