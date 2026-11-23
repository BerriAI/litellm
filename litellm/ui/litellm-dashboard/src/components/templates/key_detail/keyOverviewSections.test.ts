import { describe, expect, it } from "vitest";

import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

import { hasRateLimits, keyAccess, keyRateLimits, stringList, unsetSections } from "./keyOverviewSections";

describe("keyRateLimits", () => {
  it("merges per-model request and token limits into one row per model", () => {
    const limits = keyRateLimits({
      metadata: {
        model_rpm_limit: { "claude-sonnet-5-5": 120, "gpt-6.1-sol": 60 },
        model_tpm_limit: { "claude-sonnet-5-5": 400000, "gpt-6-astra": 80000 },
      },
    });

    expect(limits.perModel).toEqual([
      { model: "claude-sonnet-5-5", rpm: 120, tpm: 400000 },
      { model: "gpt-6.1-sol", rpm: 60, tpm: null },
      { model: "gpt-6-astra", rpm: null, tpm: 80000 },
    ]);
  });

  it("reads per-tag request limits and drops values that are not numbers", () => {
    const limits = keyRateLimits({ metadata: { tag_rpm_limit: { "batch-jobs": 20, broken: "x", eval: 5 } } });

    expect(limits.perTag).toEqual([
      { tag: "batch-jobs", rpm: 20 },
      { tag: "eval", rpm: 5 },
    ]);
  });

  it("treats null and missing key-level limits as no limit", () => {
    const limits = keyRateLimits({ rpm_limit: null, tpm_limit: 2000000, max_parallel_requests: 0 });

    const expected = { rpm: null, tpm: 2000000, tpd: null, parallel: 0 };
    expect(limits).toMatchObject(expected);
  });
});

describe("hasRateLimits", () => {
  it("is false for a key with no limits at all", () => {
    expect(hasRateLimits(keyRateLimits({ metadata: {} }))).toBe(false);
  });

  it("is true when only a per-tag limit is set", () => {
    expect(hasRateLimits(keyRateLimits({ metadata: { tag_rpm_limit: { eval: 5 } } }))).toBe(true);
  });

  it("counts a zero limit as set", () => {
    expect(hasRateLimits(keyRateLimits({ max_parallel_requests: 0 }))).toBe(true);
  });
});

describe("keyAccess", () => {
  it("reads the no-servers sentinel as blocked, even next to other servers", () => {
    expect(keyAccess({ mcp_servers: ["github", NO_MCP_SERVERS_SENTINEL] }).mcp).toEqual({ kind: "blocked" });
  });

  it("reads the all-proxy sentinel as every server", () => {
    expect(keyAccess({ mcp_servers: [ALL_PROXY_MCP_SERVERS_SENTINEL] }).mcp).toEqual({ kind: "allProxy" });
  });

  it("lists servers, access groups and toolsets granted on the key", () => {
    const access = keyAccess({
      mcp_servers: ["github"],
      mcp_access_groups: ["support-mcp"],
      mcp_toolsets: ["triage-tools"],
    });

    const expected = { kind: "list", servers: ["github"], accessGroups: ["support-mcp"], toolsets: ["triage-tools"] };
    expect(access.mcp).toEqual(expected);
  });

  it("reports no MCP grant when nothing is set", () => {
    expect(keyAccess(null).mcp).toEqual({ kind: "none" });
  });
});

describe("stringList", () => {
  it("keeps only non-empty strings", () => {
    expect(stringList(["a", "", 3, null, "b"])).toEqual(["a", "b"]);
    expect(stringList("a")).toEqual([]);
  });
});

describe("unsetSections", () => {
  const empty = {
    rateLimits: keyRateLimits({}),
    access: keyAccess(null),
    guardrails: [],
    policies: [],
    loggingCount: 0,
    disabledCallbacks: [],
  };

  it("lists every section for a key with nothing configured", () => {
    expect(unsetSections(empty)).toEqual([
      "rateLimits",
      "mcpServers",
      "vectorStores",
      "agents",
      "skills",
      "guardrails",
      "policies",
      "logging",
    ]);
  });

  it("drops guardrails and policies together once either is set", () => {
    const sections = unsetSections({ ...empty, policies: ["hipaa-baseline"] });

    expect(sections).not.toContain("guardrails");
    expect(sections).not.toContain("policies");
  });

  it("treats a blocked MCP grant and an agent access group as set", () => {
    const sections = unsetSections({
      ...empty,
      access: keyAccess({ mcp_servers: [NO_MCP_SERVERS_SENTINEL], agent_access_groups: ["refunds"] }),
    });

    expect(sections).not.toContain("mcpServers");
    expect(sections).not.toContain("agents");
  });

  it("treats logging as set when only callbacks are turned off", () => {
    expect(unsetSections({ ...empty, disabledCallbacks: ["OpenTelemetry"] })).not.toContain("logging");
  });
});
