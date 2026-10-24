import { afterEach, describe, expect, it, vi } from "vitest";
import { captureChatRequest } from "./chatTelemetry";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Chat telemetry transport", () => {
  it("should send a minimal PostHog request event without gateway credentials or referrer", async () => {
    const send = vi.fn().mockResolvedValue(new Response("{}"));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    vi.stubGlobal("fetch", send);
    await captureChatRequest("opaque-user", "unique-request");

    expect(send).toHaveBeenCalledOnce();
    const [url, request] = send.mock.calls[0];
    expect(url).toBe("https://us.i.posthog.com/i/v0/e/");
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
    expect(timeout).toHaveBeenCalledExactlyOnceWith(5000);
    const expectedPayload = {
      api_key: expect.stringMatching(/^phc_/),
      event: "litellm_chat_request",
      distinct_id: "opaque-user",
      uuid: "unique-request",
      properties: {
        product: "LiteLLM Chat UI",
        $process_person_profile: false,
        $geoip_disable: true,
      },
    };
    expect(JSON.parse(request.body)).toEqual(expectedPayload);
  });

  it("should reject an unsuccessful ingestion response for the caller to handle", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 503 })));
    await expect(captureChatRequest("opaque-user", "unique-request")).rejects.toThrow(
      "Chat telemetry was not accepted",
    );
    expect(fetch).toHaveBeenCalledOnce();
  });
});
