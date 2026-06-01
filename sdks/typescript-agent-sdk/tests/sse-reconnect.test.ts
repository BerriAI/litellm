/**
 * Validation #4 — SSE auto-reconnect.
 *
 * Mid-stream, the mock proxy kills the underlying socket. The SDK should
 * reconnect with `?starting_seq=N` transparently and not lose or duplicate
 * events.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("SSE auto-reconnect", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy({ killSseAfterMs: 60 });
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("transparently reconnects mid-stream and preserves event order without loss or duplication", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "resumer",
      model: { id: "test-model" },
    });
    const session = await agent.createSession();
    const run = await session.send("kick off");

    // Emit two events before the mock kills the socket, then continue.
    const driver = (async () => {
      // Wait for the SSE connection to open.
      await new Promise((r) => setTimeout(r, 20));
      proxy.emit(run.id, "delta", { text: "alpha" });
      proxy.emit(run.id, "delta", { text: "beta" });
      // Kill happens at ~60ms after first connect.
      await new Promise((r) => setTimeout(r, 200));
      proxy.emit(run.id, "delta", { text: "gamma" });
      proxy.complete(run.id, "alphabetagamma");
    })();

    const seenSeqs: number[] = [];
    const seenTexts: string[] = [];
    for await (const ev of run.stream()) {
      seenSeqs.push(ev.seq);
      if (ev.type === "delta") {
        seenTexts.push((ev.data as { text: string }).text);
      }
    }
    await driver;

    // Each seq seen exactly once.
    expect(new Set(seenSeqs).size).toBe(seenSeqs.length);
    // Strictly increasing — no out-of-order replays leaked through.
    for (let i = 1; i < seenSeqs.length; i++) {
      expect(seenSeqs[i]).toBeGreaterThan(seenSeqs[i - 1]);
    }
    // All three deltas present (none lost).
    expect(seenTexts).toEqual(["alpha", "beta", "gamma"]);
  });

  it("respects an explicit startingSeq on first connect", async () => {
    // Disable mid-stream kill so we test the resume-from-N path only.
    const proxy2 = new MockProxy();
    const url2 = await proxy2.start();
    try {
      const agent = await Agent.create({
        apiKey: "test-key",
        baseUrl: url2,
        name: "resume-explicit",
        model: { id: "test-model" },
      });
      const session = await agent.createSession();
      const run = await session.send("go");

      // Pre-populate events so they exist when stream() opens.
      proxy2.emit(run.id, "delta", { text: "0" });
      proxy2.emit(run.id, "delta", { text: "1" });
      proxy2.emit(run.id, "delta", { text: "2" });
      proxy2.complete(run.id, "012");

      const seqs: number[] = [];
      for await (const ev of run.stream({ startingSeq: 2 })) {
        seqs.push(ev.seq);
      }
      // Should not see 0 or 1 — only seq >= 2 (delta) and the done event (seq 3).
      expect(seqs.every((s) => s >= 2)).toBe(true);
      expect(seqs).toContain(2);
    } finally {
      await proxy2.stop();
    }
  });
});
