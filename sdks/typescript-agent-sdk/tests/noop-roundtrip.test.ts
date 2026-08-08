/**
 * Validation #2 — Round-trip against noop proxy.
 *
 * Verifies: Agent.create -> agent.createSession -> session.send -> for await
 * stream -> session.getRun.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("noop round-trip", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("creates an agent, opens a session, runs a prompt, and streams events to completion", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "round-tripper",
      model: { id: "test-model" },
    });

    expect(agent.id).toMatch(/^agt_/);
    expect(agent.name).toBe("round-tripper");

    const session = await agent.createSession({
      repos: [{ url: "https://github.com/example/repo", startingRef: "main" }],
    });
    expect(session.id).toMatch(/^ses_/);
    expect(session.agentId).toBe(agent.id);

    const run = await session.send("hello world");
    expect(run.id).toMatch(/^run_/);
    expect(run.sessionId).toBe(session.id);

    // Drive the run from the proxy side.
    const driver = (async () => {
      // Tiny delay so the SDK is connected to /events before we emit.
      await new Promise((r) => setTimeout(r, 25));
      proxy.emit(run.id, "delta", { text: "hi " });
      proxy.emit(run.id, "delta", { text: "there" });
      proxy.complete(run.id, "hi there");
    })();

    const collected: { type: string; data: unknown; seq: number }[] = [];
    for await (const ev of run.stream()) {
      collected.push({ type: ev.type, data: ev.data, seq: ev.seq });
    }
    await driver;

    // Sanity: at least both deltas + done arrived in seq order.
    expect(collected.map((e) => e.type)).toContain("delta");
    expect(collected.at(-1)?.type).toBe("done");
    const seqs = collected.map((e) => e.seq);
    expect(seqs).toEqual([...seqs].sort((a, b) => a - b));

    // Re-fetch the run by ID and confirm the terminal status.
    const fetched = await session.getRun(run.id);
    expect(fetched.id).toBe(run.id);
    expect(fetched.status).toBe("finished");
    expect(fetched.result).toBe("hi there");
  });
});
