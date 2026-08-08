/**
 * Spec tests for `Agent` static API.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent, LiteLLMAgentError } from "../src/index.js";
import { MockProxy } from "./mock-proxy.js";

describe("Agent", () => {
  let proxy: MockProxy;
  let baseUrl: string;

  beforeEach(async () => {
    proxy = new MockProxy();
    baseUrl = await proxy.start();
  });

  afterEach(async () => {
    await proxy.stop();
  });

  it("Agent.create persists name + model and returns an AgentHandle", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "alpha",
      model: { id: "claude-4.6-sonnet" },
      systemPrompt: "be helpful",
      metadata: { team: "infra" },
    });
    expect(agent.id).toMatch(/^agt_/);
    expect(agent.name).toBe("alpha");

    const proxyAgent = proxy.agents.get(agent.id);
    expect(proxyAgent?.model.id).toBe("claude-4.6-sonnet");
    expect(proxyAgent?.systemPrompt).toBe("be helpful");
    expect(proxyAgent?.metadata.team).toBe("infra");
  });

  it("Agent.get round-trips a previously-created agent", async () => {
    const created = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "fetch-me",
      model: { id: "test-model" },
    });
    const fetched = await Agent.get(created.id, { apiKey: "test-key", baseUrl });
    expect(fetched.id).toBe(created.id);
    expect(fetched.name).toBe("fetch-me");
  });

  it("Agent.delete removes the agent from the server", async () => {
    const agent = await Agent.create({
      apiKey: "test-key",
      baseUrl,
      name: "transient",
      model: { id: "test-model" },
    });
    expect(proxy.agents.has(agent.id)).toBe(true);
    await agent.delete();
    expect(proxy.agents.has(agent.id)).toBe(false);
  });

  it("rejects with a 401 LiteLLMAgentError when apiKey is wrong", async () => {
    await expect(
      Agent.create({
        apiKey: "wrong-key",
        baseUrl,
        name: "x",
        model: { id: "test-model" },
        // no retries on 401
        maxRetries: 0,
      })
    ).rejects.toMatchObject({
      name: "LiteLLMAgentError",
      status: 401,
    });
  });

  it("throws missing_api_key when no apiKey provided and env is unset", async () => {
    const previous = process.env.LITELLM_API_KEY;
    delete process.env.LITELLM_API_KEY;
    try {
      await expect(
        // @ts-expect-error — testing runtime guard with missing apiKey
        Agent.create({ baseUrl, name: "x", model: { id: "m" } })
      ).rejects.toMatchObject({
        name: "LiteLLMAgentError",
        code: "missing_api_key",
      });
    } finally {
      if (previous !== undefined) process.env.LITELLM_API_KEY = previous;
    }
  });

  it("retries on a transient 503 with retry-after and ultimately succeeds", async () => {
    const proxy503 = new MockProxy({ failFirstRequest: true });
    const url503 = await proxy503.start();
    try {
      const agent = await Agent.create({
        apiKey: "test-key",
        baseUrl: url503,
        name: "resilient",
        model: { id: "test-model" },
      });
      expect(agent.id).toMatch(/^agt_/);
    } finally {
      await proxy503.stop();
    }
  });

  it("LiteLLMAgentError carries code, status, and retryable flag", () => {
    const e = new LiteLLMAgentError("nope", {
      code: "rate_limited",
      status: 429,
      retryable: true,
    });
    expect(e.code).toBe("rate_limited");
    expect(e.status).toBe(429);
    expect(e.retryable).toBe(true);
    expect(e.name).toBe("LiteLLMAgentError");
  });
});
