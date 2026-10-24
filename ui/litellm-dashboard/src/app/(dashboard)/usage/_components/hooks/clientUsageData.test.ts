import { describe, expect, it } from "vitest";
import {
  clientUsageFromLogs,
  identifyUsageClient,
  INCOMPLETE_CLIENT_USAGE,
  UsageClient,
  type ClientUsageScope,
} from "./clientUsageData";

const desktopAgent = "claude-cli/2.1.299 (external, claude-desktop-3p, agent-sdk/0.3.299)";
const log = (overrides: Record<string, unknown> = {}) => ({
  request_id: "request-1",
  user: "user-1",
  startTime: "2026-10-22T10:00:00Z",
  model: "provider-model",
  model_group: "deployment-alias",
  custom_llm_provider: "anthropic",
  spend: 0.013,
  total_tokens: 1200,
  prompt_tokens: 1000,
  completion_tokens: 200,
  request_duration_ms: 250,
  status: "success",
  request_tags: ["User-Agent: claude-cli", `User-Agent: ${desktopAgent}`],
  metadata: {
    user_agent: desktopAgent,
    additional_usage_values: { cache_read_input_tokens: 500, cache_creation_input_tokens: 100 },
  },
  ...overrides,
});
const page = (rows: ReturnType<typeof log>[], overrides: Record<string, unknown> = {}) => ({
  data: rows,
  total: rows.length,
  page: 1,
  total_pages: 1,
  total_is_capped: false,
  ...overrides,
});
const scope: ClientUsageScope = {
  userId: "user-1",
  client: UsageClient.ClaudeDesktop,
  startTime: new Date("2026-10-21T00:00:00Z"),
  endTime: new Date("2026-10-23T00:00:00Z"),
};

describe("client usage from request logs", () => {
  it.each([
    ["claude-desktop/1.0", "claude-desktop"],
    ["Claude_Desktop/1.0", "claude-desktop"],
    ["Claude Desktop/1.0", "claude-desktop"],
    ["claude-code/1.0", "claude-code"],
    ["claude_code/1.0", "claude-code"],
    ["claude-user/1.0", "claude-code"],
    ["claude-cli/1.0 (external, cli)", "claude-code"],
    ["claude-cli/1.0 (external, claude-desktop-3p, cli)", "claude-desktop"],
    ["claude-cli", null],
    ["claude-cli/1.0 (external, agent-sdk/1.0)", null],
  ])("should classify %s as %s consistently with the agent catalog", (agent, expected) => {
    expect(identifyUsageClient(agent, [])).toBe(expected);
  });

  it("should preserve recorded totals, local days, model groups, providers and measured latency", () => {
    const result = clientUsageFromLogs(page([log(), log({ request_id: "request-2", status: "failure" })]), scope);
    expect(result.error).toBeNull();
    const expectedTotals = {
      total_spend: 0.026,
      total_tokens: 2400,
      total_prompt_tokens: 2000,
      total_completion_tokens: 400,
      total_api_requests: 2,
      total_successful_requests: 1,
      total_failed_requests: 1,
      total_cache_read_input_tokens: 1000,
      total_cache_creation_input_tokens: 200,
      total_response_time_ms: 250,
      total_timed_requests: 1,
    };
    expect(result.data.metadata).toMatchObject(expectedTotals);
    expect(result.cacheAvailable).toBe(true);
    const day = result.data.results?.[0];
    expect(day?.breakdown?.models?.["provider-model"].metrics?.api_requests).toBe(2);
    expect(day?.breakdown?.model_groups?.["deployment-alias"].metrics?.spend).toBe(0.026);
    expect(day?.breakdown?.providers?.anthropic.metrics?.total_tokens).toBe(2400);
    expect(day?.breakdown?.api_keys).toEqual({});
    const timestamp = new Date("2026-10-22T10:00:00Z");
    const localDay = `${timestamp.getFullYear()}-${String(timestamp.getMonth() + 1).padStart(2, "0")}-${String(timestamp.getDate()).padStart(2, "0")}`;
    expect(day?.date).toBe(localDay);
  });

  it("should deduplicate request ids without double counting bare and full user-agent tags", () => {
    const result = clientUsageFromLogs(page([log(), log({ spend: 0.02 })]), scope);
    expect(result.data.metadata?.total_api_requests).toBe(1);
    expect(result.data.metadata?.total_spend).toBe(0.02);
    expect(result.data.metadata?.total_tokens).toBe(1200);
  });

  it("should distinguish Desktop from explicit Code clients and exclude ambiguous CLI records", () => {
    const rows = [
      log(),
      log({ request_id: "code", request_tags: [], metadata: { user_agent: "claude-cli/2.1.299 (external, cli)" } }),
      log({ request_id: "unknown", request_tags: ["User-Agent: claude-cli"], metadata: null }),
    ];
    const desktop = clientUsageFromLogs(page(rows), scope);
    const code = clientUsageFromLogs(page(rows), { ...scope, client: UsageClient.ClaudeCode });
    expect(desktop.data.metadata?.total_api_requests).toBe(1);
    expect(code.data.metadata?.total_api_requests).toBe(1);
    expect(identifyUsageClient("claude-code/3.0", [])).toBe("claude-code");
    expect(identifyUsageClient("claude-cli/3.0 (external, agent-sdk/0.3)", [])).toBeNull();
    expect(identifyUsageClient(null, JSON.stringify([`User-Agent: ${desktopAgent}`]))).toBe("claude-desktop");
    expect(identifyUsageClient(null, { "User-Agent": desktopAgent })).toBe("claude-desktop");
    expect(identifyUsageClient(null, [desktopAgent])).toBeNull();
  });

  it.each([{ total: 1001, total_pages: 2 }, { total: 10000, total_is_capped: true }, { total_pages: 2 }])(
    "should refuse totals for a partial page %j",
    (overrides) => {
      const result = clientUsageFromLogs(page([log()], overrides), scope);
      expect(result.error).toBe(INCOMPLETE_CLIENT_USAGE);
      expect(result.data.results).toEqual([]);
      expect(result.data.metadata?.total_spend).toBe(0);
    },
  );

  it("should refuse malformed or unexpectedly truncated data instead of displaying inaccurate totals", () => {
    expect(clientUsageFromLogs(page([log()], { total: 2 }), scope).error).toBeTruthy();
    expect(clientUsageFromLogs(page([log({ spend: "0.013" })]), scope).error).toBeTruthy();
    expect(clientUsageFromLogs(page([log({ startTime: "invalid" })]), scope).error).toBeTruthy();
  });

  it("should reject rows attributed to another request user regardless of key metadata", () => {
    const result = clientUsageFromLogs(page([log({ user: "someone-else" })]), scope);
    expect(result.error).toBe("The proxy returned usage outside the selected user");
    expect(result.data.results).toEqual([]);
  });

  it("should keep unknown cache separate from recorded zero values", () => {
    const unknown = clientUsageFromLogs(page([log({ status: null, metadata: { user_agent: desktopAgent } })]), scope);
    expect(unknown.cacheAvailable).toBe(false);
    expect(unknown.data.metadata?.total_tokens).toBe(1200);
    expect(unknown.data.metadata?.total_spend).toBe(0.013);
    const zero = clientUsageFromLogs(
      page([
        log({
          metadata: {
            user_agent: desktopAgent,
            additional_usage_values: { cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
          },
        }),
      ]),
      scope,
    );
    expect(zero.cacheAvailable).toBe(true);
    expect(zero.data.metadata?.total_cache_read_input_tokens).toBe(0);
  });

  it("should use recorded nested cache counts and metadata status when direct fields are absent", () => {
    const result = clientUsageFromLogs(
      page([
        log({
          status: null,
          metadata: {
            user_agent: desktopAgent,
            status: "failure",
            usage_object: { prompt_tokens_details: { cached_tokens: 500, cache_creation_tokens: 100 } },
          },
        }),
      ]),
      scope,
    );
    expect(result.cacheAvailable).toBe(true);
    expect(result.data.metadata?.total_cache_read_input_tokens).toBe(500);
    expect(result.data.metadata?.total_failed_requests).toBe(1);
    expect(result.data.metadata?.total_timed_requests).toBe(0);
  });

  it("should enforce exact timestamp bounds after the API query rounds to seconds", () => {
    const rows = [
      log({ request_id: "before", startTime: "2026-10-22T10:00:00.100Z" }),
      log({ request_id: "inside", startTime: "2026-10-22T10:00:00.500" }),
      log({ request_id: "after", startTime: "2026-10-22T10:00:01Z" }),
    ];
    const result = clientUsageFromLogs(page(rows), {
      ...scope,
      startTime: new Date("2026-10-22T10:00:00.400Z"),
      endTime: new Date("2026-10-22T10:00:00.800Z"),
    });
    expect(result.data.metadata?.total_api_requests).toBe(1);
  });

  it.each([0, 100])("should retain legacy recorded cache_write_tokens of %s", (cacheWrite) => {
    const result = clientUsageFromLogs(
      page([
        log({
          metadata: {
            user_agent: desktopAgent,
            additional_usage_values: { prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: cacheWrite } },
          },
        }),
      ]),
      scope,
    );
    expect(result.cacheAvailable).toBe(true);
    expect(result.data.metadata?.total_cache_creation_input_tokens).toBe(cacheWrite);
  });

  it("should allow a complete empty page or exactly 1,000 records", () => {
    const empty = clientUsageFromLogs(page([]), scope);
    expect(empty.error).toBeNull();
    expect(empty.data.results).toEqual([]);
    const full = clientUsageFromLogs(
      page(Array.from({ length: 1000 }, (_, index) => log({ request_id: `${index}` }))),
      scope,
    );
    expect(full.error).toBeNull();
    expect(full.data.metadata?.total_api_requests).toBe(1000);
  });
  it("should aggregate all selected-team requests including unassigned team keys", () => {
    const teamScope = { ...scope, userId: null, teamId: "engineering" };
    const rows = [log({ team_id: "engineering" }), log({ request_id: "team-key", user: null, team_id: "engineering" })];
    const result = clientUsageFromLogs(page(rows), teamScope);
    expect(result.error).toBeNull();
    expect(result.data.metadata?.total_api_requests).toBe(2);
    expect(result.data.metadata?.total_spend).toBe(0.026);
  });

  it.each([null, undefined, "different-team"])("should reject team results with mismatched team %s", (teamId) => {
    const result = clientUsageFromLogs(page([log({ team_id: teamId })]), { ...scope, teamId: "engineering" });
    expect(result.error).toBe("The proxy returned usage outside the selected team");
    expect(result.data.results).toEqual([]);
  });

  it("should require both selected member and selected team to match", () => {
    const memberScope = { ...scope, teamId: "engineering" };
    const ownOtherTeam = clientUsageFromLogs(page([log({ team_id: "design" })]), memberScope);
    const otherTeamMember = clientUsageFromLogs(page([log({ team_id: "engineering", user: "user-2" })]), memberScope);
    const member = clientUsageFromLogs(page([log({ team_id: "engineering" })]), memberScope);
    expect(ownOtherTeam.error).toContain("outside the selected team");
    expect(otherTeamMember.error).toContain("outside the selected user");
    expect(member.error).toBeNull();
    expect(member.data.metadata?.total_api_requests).toBe(1);
  });

  it("should refuse unscoped request-log aggregation", () => {
    expect(clientUsageFromLogs(page([log()]), { ...scope, userId: null }).error).toContain("Select a user or team");
  });

  it("should exclude unrelated clients containing a Desktop marker from Desktop totals", () => {
    const rows = [
      log(),
      log({
        request_id: "unrelated-client",
        request_tags: ["User-Agent: python-httpx/1.0 (claude-desktop-3p)"],
        metadata: { user_agent: "python-httpx/1.0 (claude-desktop-3p)" },
      }),
      log({
        request_id: "ambiguous-cli",
        request_tags: ["User-Agent: claude-cli/1.0 (integration=claude-desktop-3p)"],
        metadata: null,
      }),
    ];
    const result = clientUsageFromLogs(page(rows), scope);
    expect(result.error).toBeNull();
    expect(result.data.metadata?.total_api_requests).toBe(1);
    expect(result.data.metadata?.total_spend).toBe(0.013);
  });

  it.each([
    { requestTags: [" User Agent: claude-desktop/1.0 "] },
    { requestTags: { " User Agent ": "claude-desktop/1.0" } },
    { requestTags: { " User-Agent: claude-desktop/1.0 ": true } },
  ])("should recognize the catalog's normalized tag formats %j", ({ requestTags }) => {
    const result = clientUsageFromLogs(page([log({ request_tags: requestTags, metadata: null })]), scope);
    expect(result.data.metadata?.total_api_requests).toBe(1);
  });

  it("should retain recorded zero and positive cache counts from the Responses API usage shape", () => {
    const responseUsage = { input_tokens_details: { cached_tokens: 0, cache_write_tokens: 100 } };
    const result = clientUsageFromLogs(page([log({ metadata: { usage_object: responseUsage } })]), scope);
    expect(result.cacheAvailable).toBe(true);
    expect(result.data.metadata?.total_cache_read_input_tokens).toBe(0);
    expect(result.data.metadata?.total_cache_creation_input_tokens).toBe(100);
  });
});
