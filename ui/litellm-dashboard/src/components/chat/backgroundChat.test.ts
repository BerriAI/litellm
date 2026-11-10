import { afterEach, describe, expect, it, vi } from "vitest";
import type { Response as ModelResponse } from "openai/resources/responses/responses";
import { backgroundResponseUpdate, cancelBackgroundChat, watchBackgroundChat } from "./backgroundChat";

vi.mock("@/components/networking", () => ({ getGatewayBaseUrl: () => "http://localhost:4000/v1" }));

afterEach(() => vi.unstubAllGlobals());

it("should restore text, reasoning, tool results and usage from a completed response", () => {
  const response = {
    id: "litellm_poll_example",
    status: "completed",
    output: [
      { type: "reasoning", summary: [{ type: "summary_text", text: "Check the inventory" }] },
      {
        type: "mcp_call",
        id: "call_lookup",
        name: "lookup",
        arguments: "{}",
        output: "12 widgets",
        server_label: "inventory",
      },
      { type: "message", content: [{ type: "output_text", text: "There are 12 widgets." }] },
    ],
    usage: { input_tokens: 4, output_tokens: 9, total_tokens: 13, output_tokens_details: { reasoning_tokens: 2 } },
  } as ModelResponse;
  const update = backgroundResponseUpdate(response);
  expect(update.content).toBe("There are 12 widgets.");
  expect(update.reasoningContent).toBe("Check the inventory");
  expect(update.mcpEvents?.[0].item?.output).toBe("12 widgets");
  const usage = { promptTokens: 4, completionTokens: 9, totalTokens: 13, reasoningTokens: 2 };
  expect(update.usage).toMatchObject(usage);
  expect(update.backgroundRun).toEqual({ id: response.id, status: "completed" });
});

describe("background response lifecycle", () => {
  it("should detach polling without cancelling the server run", async () => {
    const controller = new AbortController();
    const fetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      Response.json({ id: "litellm_poll_example", status: "in_progress", output: [] }),
    );
    vi.stubGlobal("fetch", fetch);
    const onUpdate = vi.fn(() => controller.abort());
    const polling = {
      accessToken: "test-token",
      responseId: "litellm_poll_example",
      signal: controller.signal,
      onUpdate,
    };
    await watchBackgroundChat(polling);
    expect(onUpdate).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0][1]?.method).toBe("GET");
  });

  it("should mark an expired run unavailable without clearing its partial answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ error: { message: "Expired" } }, { status: 404 })),
    );
    const onUpdate = vi.fn();
    const polling = {
      accessToken: "test-token",
      responseId: "litellm_poll_expired",
      signal: new AbortController().signal,
      onUpdate,
    };
    await watchBackgroundChat(polling);
    expect(onUpdate).toHaveBeenCalledWith({
      backgroundRun: { id: "litellm_poll_expired", status: "expired" },
      backgroundError: "This saved run is no longer available. Its result may have expired on the server.",
    });
  });

  it("should cancel explicitly through the server and preserve returned partial output", async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      expect(request.method).toBe("POST");
      expect(new URL(request.url).pathname).toBe("/v1/responses/litellm_poll_example/cancel");
      return Response.json({
        id: "litellm_poll_example",
        status: "cancelled",
        output: [{ type: "message", content: [{ type: "output_text", text: "Partial answer" }] }],
      });
    });
    vi.stubGlobal("fetch", fetch);
    expect(await cancelBackgroundChat("test-token", "litellm_poll_example")).toMatchObject({
      content: "Partial answer",
      backgroundRun: { id: "litellm_poll_example", status: "cancelled" },
    });
  });
});

it.each([
  [0.125, 0.125],
  ["0.125", 0.125],
  [0, 0],
  ["invalid", undefined],
])("should preserve valid returned cost %s", (cost, expected) => {
  const response = {
    id: "litellm_poll_cost",
    status: "completed",
    output: [],
    usage: { input_tokens: 4, output_tokens: 9, total_tokens: 13, cost },
  } as ModelResponse;
  expect(backgroundResponseUpdate(response).usage?.cost).toBe(expected);
});

it("should keep polling through unfinished output items until the answer and tools arrive", async () => {
  let polls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      polls++;
      return Response.json({
        id: "litellm_poll_partial",
        status: polls === 1 ? "in_progress" : "completed",
        output:
          polls === 1
            ? [
                { type: "message", id: "message_1" },
                { type: "mcp_list_tools", id: "tools_1", server_label: "inventory" },
              ]
            : [
                { type: "message", content: [{ type: "output_text", text: "12 widgets" }] },
                { type: "mcp_list_tools", server_label: "inventory", tools: [{ name: "lookup", input_schema: {} }] },
              ],
      });
    }),
  );
  const onUpdate = vi.fn();
  const polling = {
    accessToken: "test-token",
    responseId: "litellm_poll_partial",
    signal: new AbortController().signal,
    onUpdate,
  };
  await watchBackgroundChat(polling);
  expect(polls).toBe(2);
  expect(onUpdate.mock.lastCall?.[0]).toMatchObject({
    content: "12 widgets",
    backgroundRun: { status: "completed" },
    mcpEvents: [{ item: { tools: [{ name: "lookup" }] } }],
  });
});

it("should restore server timings from polling without counting time away from the chat", async () => {
  const response = {
    id: "litellm_poll_timing",
    status: "completed",
    created_at: 1000,
    output: [],
    litellm_response_timing: { time_to_first_token_ms: 250, total_latency_ms: 12500 },
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json(response)),
  );
  const onUpdate = vi.fn();
  const polling = {
    accessToken: "test-token",
    responseId: "litellm_poll_timing",
    signal: new AbortController().signal,
    onUpdate,
  };
  await watchBackgroundChat(polling);
  expect(onUpdate).toHaveBeenCalledWith(expect.objectContaining({ timeToFirstToken: 250, totalLatency: 12500 }));
});

it.each(["queued", "completed"] as const)(
  "should not invent timings for a %s response without measurements",
  (status) => {
    const update = backgroundResponseUpdate({ id: "litellm_poll_legacy", status, output: [] } as ModelResponse);
    expect(update).not.toHaveProperty("timeToFirstToken");
    expect(update).not.toHaveProperty("totalLatency");
  },
);

it("should preserve and deduplicate source annotations while excluding unsafe links", () => {
  const annotations = [
    { type: "url_citation", url: "https://example.com/report", title: "Report", start_index: 0, end_index: 6 },
    { type: "url_citation", url: "https://example.com/report", title: "Report", start_index: 8, end_index: 14 },
    { type: "url_citation", url: "javascript:alert(1)", title: "Unsafe", start_index: 8, end_index: 14 },
  ];
  const response = {
    id: "litellm_poll_research",
    status: "completed",
    output: [
      { type: "web_search_call", id: "search", status: "completed", action: { type: "search", query: "inventory" } },
      { type: "message", content: [{ type: "output_text", text: "A report.", annotations }] },
    ],
  } as ModelResponse;
  expect(backgroundResponseUpdate(response)).toMatchObject({
    content: "A report.",
    webSearchCount: 1,
    citations: [{ url: "https://example.com/report", title: "Report" }],
  });
});
