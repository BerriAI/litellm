import { describe, expect, it } from "vitest";
import { parseResponsesWebSocketTurns } from "./prettyMessagesUtils";

describe("parseResponsesWebSocketTurns", () => {
  it("reads event types a bounded number of times for a large session", () => {
    const reads = { count: 0 };
    const events = Array.from({ length: 2000 }, (_, index) =>
      ["response.created", "response.completed", "response.completed"].map((type) => ({
        get type() {
          reads.count += 1;
          return type;
        },
        response: { id: `resp_${index}`, output: [] },
      })),
    ).flat();
    const turns = parseResponsesWebSocketTurns(events);
    expect(turns).toHaveLength(2000);
    expect(turns?.[1999].id).toBe("resp_1999");
    expect(reads.count).toBeLessThan(events.length * 20);
  });

  it.each([
    null,
    {},
    [],
    { results: null },
    { choices: [] },
    { output: [] },
    { results: [{ type: "response.done" }] },
    { results: [null, "invalid", 42] },
  ])("leaves non-Responses payloads to their existing renderer: %j", (payload) => {
    expect(parseResponsesWebSocketTurns(payload)).toBeNull();
  });

  it("uses terminal output once, skipping deltas, created snapshots, and duplicate terminal IDs", () => {
    const response = {
      id: "resp_1",
      output: [{ type: "message", content: [{ type: "output_text", text: "Final text" }] }],
    };
    expect(
      parseResponsesWebSocketTurns([
        null,
        { type: "response.created", response },
        { type: "response.output_text.delta", delta: "Final text" },
        { type: "response.completed", response },
        { type: "response.completed", response },
      ]),
    ).toEqual([
      {
        id: "resp_1",
        status: "Completed",
        detail: "",
        message: { role: "assistant", content: "Final text", toolCalls: undefined },
      },
    ]);
  });

  it("keeps failed output and errors inspectable alongside an unfinished turn", () => {
    const turns = parseResponsesWebSocketTurns({
      results: [
        {
          type: "response.failed",
          response: {
            id: "failed",
            error: { code: "server_error" },
            output: [{ type: "message", content: [{ type: "output_text", text: "Partial answer" }] }],
          },
        },
        { type: "response.created", response: { id: "pending" } },
        { type: "error", error: { message: "Connection error" } },
      ],
    });
    expect(turns?.map(({ status, detail }) => ({ status, detail }))).toEqual([
      { status: "Failed", detail: "server_error" },
      { status: "No terminal event recorded", detail: "" },
      { status: "Error", detail: "Connection error" },
    ]);
    expect(turns?.[0].message?.content).toBe("Partial answer");
  });

  it("does not invent output for malformed terminal payloads", () => {
    expect(parseResponsesWebSocketTurns([{ type: "response.completed", response: null }])).toEqual([
      { id: "", status: "Completed", message: null, detail: "" },
    ]);
  });
});
