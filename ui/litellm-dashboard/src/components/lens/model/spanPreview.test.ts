import { describe, expect, it } from "vitest";
import { assistantReply, stepFailure, timeline, toolCall, userAsk } from "./spanPreview";

const OMIT = "\n[... preview omitted; read this span for evidence ...]\n";

const span = (kind: string, name: string, preview: string, cited = false) => ({
  span_id: `${kind}-${name}`,
  kind,
  name,
  preview,
  cited,
});

const agentPreview =
  'Input: [{"role": "system", "content": "You are a research assistant. Cite sources."}, {"role": "user", "content": "Find the median latency of the EU region."}]\nOutput: [{"role": "assistant", "content": "I could not find a source for that."}]';

describe("user ask", () => {
  it("takes the user's message and never the system prompt", () => {
    expect(userAsk(agentPreview)).toBe("Find the median latency of the EU region.");
  });

  it("marks a user message cut off by the preview limit", () => {
    const cut = 'Input: [{"role": "system", "content": "You are a billing agent."}, {"role": "user", "content": "Cancel C2473\'s subscript';
    expect(userAsk(cut)).toBe("Cancel C2473's subscript…");
  });

  it("is null when only a system prompt is visible", () => {
    expect(userAsk(`Input: [{"role": "system", "content": "You are a bill${OMIT}x"}]`)).toBeNull();
  });
});

describe("assistant reply", () => {
  it("reads the assistant output of an agent span", () => {
    expect(assistantReply(agentPreview)).toBe("I could not find a source for that.");
  });

  it("recovers the end of a reply from a truncated llm span without its input", () => {
    const llm = `Input: [{"role": "system", "content": "You are Acme's${OMIT}you want, I can also help you draft the message to your bank."}]\nStatus: STATUS_CODE_UNSET `;
    expect(assistantReply(llm)).toBe("…you want, I can also help you draft the message to your bank.");
  });

  it("ignores a truncated tail that is a tool call or a JSON payload", () => {
    const toolTail = `Input: [{"role": "system", "content": "You are a rese${OMIT}rch_docs", "arguments": "{\\"query\\":\\"latency\\"}"}}]}]\nStatus: STATUS_CODE_UNSET`;
    expect(assistantReply(toolTail)).toBeNull();
  });

  it("is null when the output is empty", () => {
    expect(assistantReply('Input: [{"role": "user", "content": "hi"}]\nOutput: \nStatus: STATUS_CODE_ERROR x')).toBeNull();
  });
});

describe("tool call", () => {
  it("names the tool and keeps args and result compact", () => {
    const tool = span(
      "tool",
      "execute_tool lookup_order",
      'Input: {"order_id":"A4160"}\nOutput: {"order_id": "A4160", "status": "processing"}\nStatus: STATUS_CODE_UNSET ',
    );
    expect(toolCall(tool)).toEqual({
      kind: "tool",
      name: "lookup_order",
      args: '{"order_id":"A4160"}',
      result: '{"order_id":"A4160","status":"processing"}',
      error: false,
    });
  });

  it("flags error status and error payloads but not ordinary results", () => {
    const failed = 'Input: {"query":"q"}\nOutput: {"error": "500 search index unavailable"}\nStatus: STATUS_CODE_ERROR 500';
    expect(toolCall(span("tool", "execute_tool search_docs", failed)).error).toBe(true);
    const denied = 'Input: {}\nOutput: {"status": "denied", "reason": "needs approval"}\nStatus: STATUS_CODE_UNSET';
    expect(toolCall(span("tool", "execute_tool issue_refund", denied)).error).toBe(true);
    const tests = 'Input: {}\nOutput: {"passed": 41, "failed": 3}\nStatus: STATUS_CODE_UNSET';
    expect(toolCall(span("tool", "execute_tool run_tests", tests)).error).toBe(false);
  });

  it("reads a tool result that only survives in the preview tail", () => {
    const cut = `Input: {"url":"https://example.com"}\nOutput: {"url": ${OMIT}, "text": "the figure is 37%"}\nStatus: STATUS_CODE_UNSET`;
    expect(toolCall(span("tool", "execute_tool fetch_url", cut)).result).toBe('…, "text": "the figure is 37%"}');
  });

  it("keeps args clean when the preview cut lands inside the Output header", () => {
    const cut = `Input: {"customer_id":"C3847"}\nO${OMIT}t month", "amount_usd": 416.67}\nStatus: STATUS_CODE_UNSET `;
    const call = toolCall(span("tool", "execute_tool get_invoice", cut));
    expect(call.args).toBe('{"customer_id":"C3847"}');
    expect(call.result).toBe('…t month", "amount_usd": 416.67}');
  });
});

describe("step failure", () => {
  it("reports only non-ok statuses", () => {
    expect(stepFailure("Output: \nStatus: STATUS_CODE_ERROR Request timed out.")).toBe("Request timed out.");
    expect(stepFailure("Output: x\nStatus: STATUS_CODE_UNSET ")).toBeNull();
    expect(stepFailure("plain")).toBeNull();
  });
});

describe("timeline", () => {
  it("reads like a conversation: ask, tool steps, then the final reply, without system text", () => {
    const items = timeline([
      span("agent", "invoke_agent research-agent", agentPreview),
      span("llm", "chat gpt", `Input: [{"role": "system", "content": "You are a rese${OMIT}I could not find a source for that."}]`),
      span(
        "tool",
        "execute_tool search_docs",
        'Input: {"query":"EU latency"}\nOutput: {"error": "500 search index unavailable"}\nStatus: STATUS_CODE_ERROR 500',
        true,
      ),
    ]);
    expect(items.map((i) => i.kind)).toEqual(["ask", "tool", "reply"]);
    expect(items[0]).toMatchObject({ text: "Find the median latency of the EU region.", span: 0 });
    expect(items[1]).toMatchObject({ name: "search_docs", error: true, span: 2 });
    expect(items[2]).toMatchObject({ text: "I could not find a source for that.", span: 0 });
    expect(JSON.stringify(items)).not.toContain("research assistant");
  });

  it("shows a failed model call as a failure step", () => {
    const items = timeline([span("llm", "chat", "Input: x\nOutput: \nStatus: STATUS_CODE_ERROR Request timed out.")]);
    expect(items).toEqual([{ kind: "failure", text: "Model call failed: Request timed out.", span: 0 }]);
  });

  it("falls back to short notes when nothing conversational can be recovered", () => {
    const items = timeline([span("chain", "plan", 'Input: {"step": 1}')]);
    expect(items).toEqual([{ kind: "note", label: "plan", text: '{"step":1}', span: 0 }]);
  });
});
