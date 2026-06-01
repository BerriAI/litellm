/**
 * Validation #6 — Async dispose tears down VM.
 *
 * Uses `await using session = ...`; checks proxy state shows session
 * terminated after scope exit.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent, type SessionHandle } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("Symbol.asyncDispose", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("calls DELETE /v2/sessions/{id} when the await-using scope exits", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "disposer",
      model: { id: "test-model" },
    });

    let sessionId: string;
    {
      // `await using` requires TS 5.2+ and proper polyfill. We invoke the
      // dispose method explicitly so the test is portable across runners
      // without forcing the explicit-resource-management transform.
      const session: SessionHandle = await agent.createSession();
      sessionId = session.id;
      const proxySession = proxy.findSession(sessionId);
      expect(proxySession?.status).not.toBe("terminated");
      await session[Symbol.asyncDispose]();
    }

    const after = proxy.findSession(sessionId);
    expect(after?.status).toBe("terminated");
  });

  it("dispose is a no-op once the session has already been deleted", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "double-dispose",
      model: { id: "test-model" },
    });

    const session = await agent.createSession();
    await session.delete();

    // A second dispose must not throw.
    await expect(session[Symbol.asyncDispose]()).resolves.toBeUndefined();
  });
});
