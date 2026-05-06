/**
 * Validation #3 — Agent reuse across sessions.
 *
 * Creates ONE agent, then 3 sessions under it. Asserts: same agentId,
 * distinct sessionIds, distinct VMs.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("agent reuse across sessions", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("creates 3 sessions under one agent with distinct IDs and distinct VMs", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "shared",
      model: { id: "test-model" },
    });

    const sessions = await Promise.all([
      agent.createSession(),
      agent.createSession(),
      agent.createSession(),
    ]);

    // All sessions share the agent ID.
    for (const s of sessions) {
      expect(s.agentId).toBe(agent.id);
    }

    // Session IDs are unique.
    const ids = sessions.map((s) => s.id);
    expect(new Set(ids).size).toBe(3);

    // Mock proxy assigns each session a distinct VM ID.
    const proxyAgent = proxy.agents.get(agent.id);
    expect(proxyAgent).toBeDefined();
    const vmIds = [...proxyAgent!.sessions.values()].map((s) => s.vmId);
    expect(new Set(vmIds).size).toBe(3);

    // listSessions reflects all 3.
    const listed = await agent.listSessions();
    expect(listed.items.length).toBe(3);
    expect(new Set(listed.items.map((i) => i.id))).toEqual(new Set(ids));
  });
});
