import { describe, expect, it } from "vitest";
import { agentDailyData, isUserAgentTag, topAgents, userAgentProduct, type TagSummaryRow } from "./agentCatalog";
import type { DailyData } from "@/components/UsagePage/types";

const metric = (spend: number, tokens: number) => ({
  metrics: {
    spend,
    total_tokens: tokens,
    api_requests: 1,
    prompt_tokens: 0,
    completion_tokens: 0,
    successful_requests: 1,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  },
  metadata: {},
  api_key_breakdown: {},
});

const row = (
  tag: string,
  spend: number,
  tokens: number,
  { requests = 1, users = 1 }: { requests?: number; users?: number } = {},
): TagSummaryRow => ({
  tag,
  total_spend: spend,
  total_tokens: tokens,
  total_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  unique_users: users,
});

describe("userAgentProduct", () => {
  it("drops the prefix, version and platform detail", () => {
    expect(userAgentProduct("User-Agent: claude-cli/2.1.263 (external, cli)")).toBe("claude-cli");
    expect(userAgentProduct("User-Agent: Codex Desktop/0.160.1 (Mac OS 26.6.0; arm64)")).toBe("Codex Desktop");
    expect(userAgentProduct("User-Agent: python")).toBe("python");
  });
});

describe("isUserAgentTag", () => {
  it("only treats User-Agent tags as agents", () => {
    expect(isUserAgentTag("User-Agent: codex-tui")).toBe(true);
    expect(isUserAgentTag("Credential: gemini")).toBe(false);
    expect(isUserAgentTag("claude-code")).toBe(false);
  });
});

describe("topAgents", () => {
  it("counts a product from its bare rollup row, not again from its versioned children", () => {
    const agents = topAgents([
      row("User-Agent: python", 100, 1000),
      row("User-Agent: python/gevent-http-client-2.3.9", 100, 1000),
    ]);
    expect(agents).toHaveLength(1);
    expect(agents[0]).toMatchObject({ id: "python", spend: 100, tokens: 1000 });
  });

  it("sums versioned rows when a product has no bare rollup", () => {
    const agents = topAgents([
      row("User-Agent: opencode/1.18.29 ai-sdk/provider-utils/4.0.23", 5, 50),
      row("User-Agent: opencode/1.18.30 ai-sdk/provider-utils/4.0.23", 7, 70),
    ]);
    expect(agents[0]).toMatchObject({ id: "opencode", label: "OpenCode", spend: 12, tokens: 120 });
  });

  it("folds every Codex surface into one Codex agent with the OpenAI mark", () => {
    const agents = topAgents([
      row("User-Agent: Codex Desktop", 20, 200),
      row("User-Agent: codex-tui", 5, 50),
      row("User-Agent: codex_exec/0.1", 1, 10),
      row("User-Agent: codex_github_action/1.0", 2, 20),
    ]);
    expect(agents).toHaveLength(1);
    expect(agents[0]).toMatchObject({ id: "codex", label: "Codex", kind: "agent", spend: 28, tokens: 280 });
    expect(agents[0].logo).toBeTruthy();
  });

  it("maps claude-cli to Claude Code and ranks by tokens", () => {
    const agents = topAgents([
      row("User-Agent: python", 600, 300),
      row("User-Agent: claude-cli", 370, 343),
      row("User-Agent: Cursor/1.0", 1, 7),
    ]);
    expect(agents.map((agent) => agent.label)).toEqual(["Claude Code", "Python", "Cursor"]);
    expect(agents[0].kind).toBe("agent");
    expect(agents[1].kind).toBe("sdk");
  });

  it("sums distinct products in a family, each read once from its own rollup (prod shape)", () => {
    const agents = topAgents([
      row("User-Agent: python", 61081, 30508),
      row("User-Agent: python/gevent-http-client-2.3.9", 61081, 30508),
      row("User-Agent: python-httpx", 8616, 1728),
      row("User-Agent: python-httpx/0.28.1", 8616, 1728),
    ]);
    expect(agents).toHaveLength(1);
    expect(agents[0]).toMatchObject({ id: "python", spend: 61081 + 8616, tokens: 30508 + 1728 });
  });

  it("does not let a differently-cased product overwrite another's rollup (prod: python vs Python)", () => {
    const [agent] = topAgents([
      row("User-Agent: python", 61081, 30508),
      row("User-Agent: python/gevent-http-client-2.3.9", 61081, 30508),
      row("User-Agent: Python", 1, 1),
      row("User-Agent: Python/3.13 aiohttp/3.14.3", 1, 1),
    ]);
    expect(agent).toMatchObject({ id: "python", spend: 61082, tokens: 30509 });
  });

  it("recognises the OpenAI Agents SDK user agent", () => {
    expect(topAgents([row("User-Agent: Agents/Python 0.4.2", 14, 10)])[0]).toMatchObject({ id: "openai-agents" });
  });

  it("keeps unrecognised clients under their own name with no logo", () => {
    const [agent] = topAgents([row("User-Agent: SpeechCoach/2.0", 1, 5)]);
    expect(agent).toMatchObject({ id: "ua:speechcoach", label: "SpeechCoach", logo: null });
  });

  it("ignores non user-agent tags and empty agents", () => {
    expect(topAgents([row("Credential: gemini", 200, 0), row("User-Agent: ureq", 0, 0, { requests: 0 })])).toEqual([]);
  });

  it("does not sum unique users across a family", () => {
    const [agent] = topAgents([
      row("User-Agent: Codex Desktop", 1, 1, { users: 8 }),
      row("User-Agent: codex-tui", 1, 1, { users: 5 }),
    ]);
    expect(agent.users).toBe(8);
  });

  it("drills in on exactly the tags it counted, so a request carrying bare and versioned tags is not counted twice", () => {
    const [agent] = topAgents([
      row("User-Agent: claude-cli", 10, 100),
      row("User-Agent: claude-cli/2.1.263 (external, cli)", 10, 100),
      row("User-Agent: claude-code", 2, 20),
    ]);
    expect([...agent.tags].sort()).toEqual(["User-Agent: claude-cli", "User-Agent: claude-code"]);
    expect(agent).toMatchObject({ spend: 12, tokens: 120 });
  });

  it("drills in on every versioned tag when a product has no bare rollup", () => {
    const [agent] = topAgents([
      row("User-Agent: opencode/1.18.29 ai-sdk", 5, 50),
      row("User-Agent: opencode/1.18.30 ai-sdk", 7, 70),
    ]);
    expect([...agent.tags].sort()).toEqual([
      "User-Agent: opencode/1.18.29 ai-sdk",
      "User-Agent: opencode/1.18.30 ai-sdk",
    ]);
  });
});

describe("agentDailyData", () => {
  it("stacks one entry per agent per day, counting a rollup tag once and ignoring non user-agent tags", () => {
    const day = {
      date: "2026-10-01",
      metrics: metric(0, 0).metrics,
      breakdown: {
        models: {},
        model_groups: {},
        mcp_servers: {},
        providers: {},
        api_keys: {},
        entities: {
          "User-Agent: claude-cli": metric(10, 100),
          "User-Agent: claude-cli/2.1.263 (external, cli)": metric(10, 100),
          "User-Agent: Codex Desktop/0.160.1 (Mac OS)": metric(4, 40),
          "Credential: gemini": metric(99, 0),
        },
      },
    } as unknown as DailyData;
    const [result] = agentDailyData([day]);
    // The day total is the agents' own sum, so untagged and non-agent traffic never shows up as "Other".
    expect(result.metrics).toMatchObject({ spend: 14, total_tokens: 140 });
    expect(Object.keys(result.breakdown.models).sort()).toEqual(["Claude Code", "Codex"]);
    expect(result.breakdown.models["Claude Code"].metrics).toMatchObject({ spend: 10, total_tokens: 100 });
    expect(result.breakdown.models["Codex"].metrics).toMatchObject({ spend: 4, total_tokens: 40 });
  });
});
