/**
 * Spec tests for `SessionHandle`.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("SessionHandle", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("send() accepts a string and starts a new run", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send("hello");
    expect(run.id).toMatch(/^run_/);
    expect(run.sessionId).toBe(session.id);
    expect(run.status).toBe("queued");
  });

  it("send() accepts a structured input with images", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const run = await session.send({
      text: "describe this",
      images: [{ url: "https://example.com/cat.png" }],
    });
    expect(run.id).toMatch(/^run_/);
  });

  it("send() returns a 409 if a run is already in flight", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    await session.send("first");
    await expect(
      session.send({ text: "second", images: [] })
    ).rejects.toMatchObject({ name: "LiteLLMAgentError", status: 409 });
  });

  it("getRun() round-trips a run by ID", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const created = await session.send("yo");
    const fetched = await session.getRun(created.id);
    expect(fetched.id).toBe(created.id);
    expect(fetched.sessionId).toBe(session.id);
  });

  it("listRuns() returns runs created in this session", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    const r1 = await session.send("one");
    proxy.complete(r1.id, "done one");
    const r2 = await session.send("two");
    proxy.complete(r2.id, "done two");

    const listed = await session.listRuns();
    const ids = new Set(listed.items.map((r) => r.id));
    expect(ids).toEqual(new Set([r1.id, r2.id]));
  });

  it("conversation() returns turns recorded by the proxy", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    await session.send("first message");
    const turns = await session.conversation();
    expect(turns.some((t) => t.role === "user" && t.content === "first message")).toBe(
      true
    );
  });

  it("delete() flips status to terminated server-side", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    await session.delete();
    const proxySession = proxy.findSession(session.id);
    expect(proxySession?.status).toBe("terminated");
    expect(session.status).toBe("terminated");
  });

  it("terminate() is an alias for delete()", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "s",
      model: { id: "m" },
    });
    const session = await agent.createSession();
    await session.terminate();
    expect(proxy.findSession(session.id)?.status).toBe("terminated");
  });
});
