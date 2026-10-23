import { afterEach, describe, expect, it, vi } from "vitest";
import { captureChatEvents } from "./chatPosthog";

afterEach(() => vi.unstubAllGlobals());

describe("Chat PostHog transport", () => {
  it("should send only manual anonymous metrics to the docs project without gateway credentials or referrer", async () => {
    const send = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", send);
    await captureChatEvents("opaque-user", ["LiteLLM Chat UI", "LiteLLM Chat UI Query"], "2026-10-23T12:00:00Z");

    expect(send).toHaveBeenCalledOnce();
    const [url, request] = send.mock.calls[0];
    expect(url).toBe("https://us.i.posthog.com/batch/");
    const expectedRequest = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      referrerPolicy: "no-referrer",
      keepalive: true,
      signal: expect.any(AbortSignal),
      body: expect.any(String),
    };
    expect(request).toEqual(expectedRequest);
    expect(JSON.parse(request.body)).toEqual({
      api_key: expect.any(String),
      batch: ["LiteLLM Chat UI", "LiteLLM Chat UI Query"].map((event) => ({
        event,
        distinct_id: "opaque-user",
        timestamp: "2026-10-23T12:00:00Z",
        properties: { $process_person_profile: false, $geoip_disable: true },
      })),
    });
  });

  it("should reject an unsuccessful ingestion response so the daily marker can be retried", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 503 })));
    await expect(captureChatEvents("opaque-user", ["LiteLLM Chat UI"], "2026-10-23T12:00:00Z")).rejects.toThrow(
      "Chat telemetry was not accepted",
    );
  });
});
