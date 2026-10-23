import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ get: vi.fn(), capture: vi.fn() }));
vi.mock("@/lib/http/api", () => ({ backgroundFetchClient: { GET: mocks.get } }));
vi.mock("@/lib/http/chatPosthog", () => ({ captureChatEvents: mocks.capture }));
vi.mock("@/contexts/ReactQueryProvider", () => ({
  queryClient: { fetchQuery: (query: { queryFn: () => Promise<boolean> }) => query.queryFn() },
}));

let chat: typeof import("./chat");
let storage: Map<string, string>;

beforeEach(async () => {
  vi.resetModules();
  vi.resetAllMocks();
  vi.stubEnv("NODE_ENV", "production");
  vi.stubGlobal("window", {});
  storage = new Map();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
  });
  mocks.get.mockResolvedValue({ data: { enabled: true, chat_posthog_enabled: true } });
  mocks.capture.mockResolvedValue(undefined);
  chat = await import("./chat");
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("Chat metrics", () => {
  it("should count daily activity once and every accepted query with an opaque ID", async () => {
    await chat.recordChatActive("alice@example.com");
    await chat.recordChatActive("alice@example.com");
    await chat.recordChatQuery("alice@example.com");
    await chat.recordChatQuery("alice@example.com");

    const id = storage.get("litellm-chat-telemetry:alice@example.com");
    expect(id).toMatch(/^[\da-f-]{36}$/);
    expect(mocks.capture.mock.calls.map(([user, events]) => ({ user, events }))).toEqual([
      { user: id, events: ["LiteLLM Chat UI"] },
      { user: id, events: ["LiteLLM Chat UI Query"] },
      { user: id, events: ["LiteLLM Chat UI Query"] },
    ]);
    expect(JSON.stringify(mocks.capture.mock.calls)).not.toContain("alice@example.com");
  });

  it("should count the next UTC day for a tab left open and include activity on query", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-23T23:59:00Z"));
    await chat.recordChatQuery("alice");
    vi.setSystemTime(new Date("2026-10-24T00:01:00Z"));
    await chat.recordChatQuery("alice");
    expect(mocks.capture.mock.calls.map(([, events]) => events)).toEqual([
      ["LiteLLM Chat UI", "LiteLLM Chat UI Query"],
      ["LiteLLM Chat UI", "LiteLLM Chat UI Query"],
    ]);
  });

  it("should reuse identity after reload and separate accounts on the same browser", async () => {
    await chat.recordChatActive("alice");
    vi.resetModules();
    chat = await import("./chat");
    await chat.recordChatActive("alice");
    await chat.recordChatActive("bob");
    const ids = mocks.capture.mock.calls.map(([id]) => id);
    expect(ids[0]).toBe(ids[1]);
    expect(ids[2]).not.toBe(ids[0]);
  });

  it.each([false, undefined])("should send nothing when external telemetry is %s", async (enabled) => {
    mocks.get.mockResolvedValue({ data: { enabled: true, chat_posthog_enabled: enabled } });
    await chat.recordChatQuery("alice");
    expect(mocks.capture).not.toHaveBeenCalled();
    expect(storage.size).toBe(0);
  });

  it("should not collect from development, unauthenticated, or server rendering", async () => {
    vi.stubEnv("NODE_ENV", "development");
    await chat.recordChatQuery("alice");
    vi.stubEnv("NODE_ENV", "production");
    await chat.recordChatQuery("");
    vi.stubGlobal("window", undefined);
    await chat.recordChatQuery("alice");
    expect(mocks.get).not.toHaveBeenCalled();
    expect(mocks.capture).not.toHaveBeenCalled();
  });

  it("should not break Chat when settings or storage are unavailable", async () => {
    mocks.get.mockRejectedValueOnce(new Error("offline"));
    await expect(chat.recordChatQuery("alice")).resolves.toBeUndefined();
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
    });
    await expect(chat.recordChatQuery("alice")).resolves.toBeUndefined();
    expect(mocks.capture).not.toHaveBeenCalled();
  });

  it("should swallow transport failures and retry daily activity on the next visit", async () => {
    mocks.capture.mockRejectedValueOnce(new Error("blocked"));
    await expect(chat.recordChatActive("alice")).resolves.toBeUndefined();
    await chat.recordChatActive("alice");
    expect(mocks.capture).toHaveBeenCalledTimes(2);
    expect(mocks.capture.mock.calls[1][1]).toEqual(["LiteLLM Chat UI"]);
  });

  it("should deduplicate simultaneous mount, focus, and submit activity", async () => {
    await Promise.all([chat.recordChatActive("alice"), chat.recordChatActive("alice"), chat.recordChatQuery("alice")]);
    expect(mocks.capture.mock.calls.flatMap(([, events]) => events)).toEqual([
      "LiteLLM Chat UI",
      "LiteLLM Chat UI Query",
    ]);
  });
});
