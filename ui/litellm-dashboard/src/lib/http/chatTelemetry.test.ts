import { afterEach, describe, expect, it, vi } from "vitest";
import { captureChatRequest } from "./chatTelemetry";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Chat telemetry transport", () => {
  it("should send only pseudonymous user and request IDs without gateway credentials or referrer", async () => {
    const send = vi.fn().mockResolvedValue(new Response("{}"));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    vi.stubGlobal("fetch", send);
    await captureChatRequest("opaque-user", "unique-request");

    expect(send).toHaveBeenCalledOnce();
    const [url, request] = send.mock.calls[0];
    expect(url).toBe("https://chat-telemetry.litellm.ai/v1/chat/events");
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
    expect(JSON.parse(request.body)).toEqual({
      user_id: "opaque-user",
      request_id: "unique-request",
    });
  });

  it("should reject an unsuccessful ingestion response for the caller to handle", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 503 })));
    await expect(captureChatRequest("opaque-user", "unique-request")).rejects.toThrow(
      "Chat telemetry was not accepted",
    );
    expect(fetch).toHaveBeenCalledOnce();
  });
});
