import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ capture: vi.fn(), fetch: vi.fn() }));
vi.mock("@/lib/http/chatTelemetry", () => ({ captureChatRequest: mocks.capture }));

let chat: typeof import("./chat");

beforeEach(async () => {
  vi.resetModules();
  vi.resetAllMocks();
  vi.stubEnv("NODE_ENV", "production");
  vi.stubGlobal("window", { location: { origin: "https://gateway.example.com" } });
  vi.stubGlobal("fetch", mocks.fetch);
  mocks.capture.mockResolvedValue(undefined);
  chat = await import("./chat");
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("Chat request metrics", () => {
  it("should do nothing when imported without a submission", () => {
    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.capture).not.toHaveBeenCalled();
  });

  it("should emit by default with a stable user ID, fresh request IDs, and no backend settings request", async () => {
    await chat.recordChatQuery("alice@example.com");
    await chat.recordChatQuery("alice@example.com");

    expect(mocks.capture).toHaveBeenCalledTimes(2);
    const ids = mocks.capture.mock.calls.map(([id]) => id);
    expect(ids[0]).toMatch(/^[\da-f]{8}-[\da-f]{4}-5[\da-f]{3}-[\da-f]{4}-[\da-f]{12}$/);
    expect(ids[0]).toBe(ids[1]);
    const requestIds = mocks.capture.mock.calls.map(([, requestId]) => requestId);
    for (const requestId of requestIds) {
      expect(requestId).toMatch(/^[\da-f]{8}-[\da-f]{4}-4[\da-f]{3}-[\da-f]{4}-[\da-f]{12}$/);
    }
    expect(new Set(requestIds).size).toBe(2);
    expect(mocks.capture.mock.calls.every((args) => args.length === 2)).toBe(true);
    expect(JSON.stringify(mocks.capture.mock.calls)).not.toContain("alice@example.com");
    expect(JSON.stringify(mocks.capture.mock.calls)).not.toContain("gateway.example.com");
    expect(mocks.fetch).not.toHaveBeenCalled();
  });

  it("should use the same user ID across reloads and browsers without stored identifiers", async () => {
    await chat.recordChatQuery("alice");
    vi.resetModules();
    chat = await import("./chat");
    vi.stubGlobal("window", { location: { origin: "https://gateway.example.com" } });
    await chat.recordChatQuery("alice");
    expect(mocks.capture.mock.calls[0][0]).toBe(mocks.capture.mock.calls[1][0]);
  });

  it("should distinguish accounts and deployments", async () => {
    await chat.recordChatQuery("alice");
    await chat.recordChatQuery("bob");
    vi.stubGlobal("window", { location: { origin: "https://another.example.com" } });
    await chat.recordChatQuery("alice");
    const ids = mocks.capture.mock.calls.map(([id]) => id);
    expect(new Set(ids).size).toBe(3);
  });

  it("should preserve user identity across days while leaving event timestamps to the collector", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-23T23:59:00Z"));
    await chat.recordChatQuery("alice");
    vi.setSystemTime(new Date("2026-10-24T00:01:00Z"));
    await chat.recordChatQuery("alice");
    expect(mocks.capture.mock.calls[0][0]).toBe(mocks.capture.mock.calls[1][0]);
    expect(mocks.capture.mock.calls[0][1]).not.toBe(mocks.capture.mock.calls[1][1]);
    expect(JSON.stringify(mocks.capture.mock.calls)).not.toContain("2026-10-");
  });

  it("should not collect from development, unauthenticated, or server rendering", async () => {
    vi.stubEnv("NODE_ENV", "development");
    await chat.recordChatQuery("alice");
    vi.stubEnv("NODE_ENV", "production");
    await chat.recordChatQuery("");
    vi.stubGlobal("window", undefined);
    await chat.recordChatQuery("alice");
    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.capture).not.toHaveBeenCalled();
  });

  it("should count requests on HTTP deployments without secure-context APIs or browser storage", async () => {
    vi.stubGlobal("window", { location: { origin: "http://gateway.internal:4000" } });
    vi.stubGlobal("crypto", { getRandomValues: crypto.getRandomValues.bind(crypto) });
    vi.stubGlobal("localStorage", undefined);
    await chat.recordChatQuery("alice");
    expect(mocks.capture).toHaveBeenCalledOnce();
  });

  it("should separate deployments mounted under different root paths", async () => {
    const { setServerRootPath } = await import("@/lib/serverRootPath");
    setServerRootPath("/team-a/");
    await chat.recordChatQuery("alice");
    setServerRootPath("team-a");
    await chat.recordChatQuery("alice");
    setServerRootPath("/team-b");
    await chat.recordChatQuery("alice");
    const ids = mocks.capture.mock.calls.map(([id]) => id);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[0]).not.toBe(ids[2]);
  });

  it("should swallow transport failures without retrying requests", async () => {
    mocks.capture.mockRejectedValueOnce(new Error("blocked"));
    await expect(chat.recordChatQuery("alice")).resolves.toBeUndefined();
    expect(mocks.capture).toHaveBeenCalledTimes(1);
    await chat.recordChatQuery("alice");
    expect(mocks.capture).toHaveBeenCalledTimes(2);
  });
});
