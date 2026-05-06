/**
 * Validation #5 — session.followup() queues correctly.
 *
 * Sends a long prompt via send(); while it's running, calls followup().
 * Expect: no 409, follow-up message appears in conversation after current
 * run terminal.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("session.followup()", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("accepts a follow-up message while a run is in flight without a 409", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "follower",
      model: { id: "test-model" },
    });
    const session = await agent.createSession();
    const run = await session.send("the long initial prompt");

    // Simulate run-in-progress by leaving status non-terminal until we say so.
    // followup() should NOT throw a 409.
    await expect(
      session.followup("please also handle the empty-input case")
    ).resolves.toBeUndefined();

    // Drive the run to completion.
    proxy.emit(run.id, "delta", { text: "ok" });
    proxy.complete(run.id, "done");
    await run.wait();

    const turns = await session.conversation();
    const userTurns = turns.filter((t) => t.role === "user").map((t) => t.content);
    expect(userTurns).toContain("the long initial prompt");
    expect(userTurns).toContain("please also handle the empty-input case");

    // Follow-up should be ordered after the initial prompt.
    const initialIdx = userTurns.indexOf("the long initial prompt");
    const followupIdx = userTurns.indexOf("please also handle the empty-input case");
    expect(followupIdx).toBeGreaterThan(initialIdx);
  });
});
