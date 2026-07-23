/**
 * Spec tests for `Run` (stream/wait/cancel/conversation).
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("Run", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("wait() resolves once the run is in a terminal state", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "r",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send("go");

    // Complete asynchronously so wait() actually has to poll.
    setTimeout(() => proxy.complete(run.id, "ok"), 50);

    const result = await run.wait();
    expect(result.id).toBe(run.id);
    expect(result.status).toBe("finished");
    expect(result.result).toBe("ok");
  });

  it("stream() yields events in seq order and ends on `done`", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "r",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send("go");

    const driver = (async () => {
      await new Promise((r) => setTimeout(r, 20));
      proxy.emit(run.id, "delta", { text: "1" });
      proxy.emit(run.id, "delta", { text: "2" });
      proxy.complete(run.id, "12");
    })();

    const seqs: number[] = [];
    const types: string[] = [];
    for await (const ev of run.stream()) {
      seqs.push(ev.seq);
      types.push(ev.type);
    }
    await driver;
    expect(seqs).toEqual([...seqs].sort((a, b) => a - b));
    expect(types.at(-1)).toBe("done");
  });

  it("cancel() flips status server-side and is a no-op once terminal", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "r",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send("go");
    await run.cancel();

    const proxyRun = proxy.findRun(run.id);
    expect(proxyRun?.status).toBe("cancelled");
    // Calling again should be a no-op (not throw).
    await expect(run.cancel()).resolves.toBeUndefined();
  });

  it("session.conversation() returns the turn list including the run's user message", async () => {
    // `Run.conversation()` was removed — there is no per-run backend endpoint.
    // Conversation history is session-scoped; callers use SessionHandle instead.
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "r",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    await session.send("hello run");
    const turns = await session.conversation();
    expect(
      turns.some((t) => t.content === "hello run" && t.role === "user"),
    ).toBe(true);
  });

  it("stream() respects an AbortSignal", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "r",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send("go");

    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 30);

    const collected: number[] = [];
    for await (const ev of run.stream({ signal: ctrl.signal })) {
      collected.push(ev.seq);
    }
    // Aborts cleanly without throwing.
    expect(Array.isArray(collected)).toBe(true);
  });
});
