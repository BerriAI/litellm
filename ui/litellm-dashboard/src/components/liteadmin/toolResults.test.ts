import { describe, expect, it } from "vitest";
import { generatedKey, projectToolResult } from "./toolResults";

describe("model result privacy", () => {
  it("preserves key policy and hashes while excluding credentials, raw keys and arbitrary metadata", () => {
    const hash = "a".repeat(64);
    const raw = {
      keys: [
        {
          token: hash,
          key: "sk-generated",
          key_alias: "alias for captured-token",
          max_budget: 50,
          models: ["model-1"],
          metadata: { private: "hidden" },
          config: { api_key: "provider-secret" },
        },
      ],
      total_count: 1,
    };
    expect(projectToolResult("key", raw, ["captured-token"])).toEqual({
      keys: [{ token: hash, key_alias: "alias for [redacted]", max_budget: 50, models: ["model-1"] }],
      total_count: 1,
    });
  });

  it.each(["token", "key_hash", "api_key"])("excludes a captured 64-character credential in %s", (field) => {
    const secret = "b".repeat(64);
    const raw = field === "api_key" ? [{ api_key: secret, total_cost: 20 }] : { [field]: secret, max_budget: 20 };
    expect(JSON.stringify(projectToolResult(field === "api_key" ? "spend" : "key", raw, [secret]))).not.toContain(
      secret,
    );
    expect(JSON.stringify(projectToolResult(field === "api_key" ? "spend" : "key", raw, [secret]))).toContain("20");
  });

  it("does not accept raw secrets where key hashes are expected", () => {
    const raw = { token: "sk-secret", key_hash: "other-secret", key_alias: "Bearer provider-secret sk-another-secret" };
    expect(projectToolResult("key", raw, [])).toEqual({
      token: undefined,
      key_hash: undefined,
      key_alias: "[redacted] [redacted]",
    });
  });

  it("keeps team membership and bounded policy details", () => {
    const raw = {
      team_info: {
        team_id: "team-1",
        team_alias: "Engineering",
        max_budget: 100,
        members_with_roles: [{ user_id: "user-1", role: "admin", password: "private" }],
        object_permission: { private: true },
      },
      keys: [{ token: "c".repeat(64), key_alias: "worker" }],
    };
    expect(projectToolResult("team", raw, [])).toEqual({
      team_info: {
        team_id: "team-1",
        team_alias: "Engineering",
        max_budget: 100,
        members_with_roles: [{ user_id: "user-1", role: "admin" }],
      },
      keys: [{ token: "c".repeat(64), key_alias: "worker" }],
    });
  });

  it("keeps budget pagination but drops returned links and arbitrary metadata", () => {
    const raw = {
      data: [{ budget_id: "budget-1", max_budget: 10, metadata: { api_key: "private" } }],
      meta: { page: 2, page_size: 10, total_count: 11, total_pages: 2, token: "private" },
      links: { next: "https://secret.example" },
    };
    expect(projectToolResult("budget", raw, [])).toEqual({
      data: [{ budget_id: "budget-1", max_budget: 10 }],
      meta: { page: 2, page_size: 10, total_count: 11, total_pages: 2 },
    });
  });

  it("projects spend breakdown metadata only into its numeric model fields", () => {
    const raw = [
      {
        group_by_day: "2031-05-01",
        teams: [
          {
            team_id: "team-1",
            team_name: "Engineering",
            total_spend: 25,
            metadata: [
              {
                model: "model-1",
                total_tokens: 30,
                spend: 25,
                headers: { Authorization: "private" },
                messages: ["private"],
              },
            ],
          },
        ],
      },
    ];
    expect(projectToolResult("spend", raw, [])).toEqual([
      {
        group_by_day: "2031-05-01",
        teams: [
          {
            team_id: "team-1",
            team_name: "Engineering",
            total_spend: 25,
            metadata: [{ model: "model-1", total_tokens: 30, spend: 25 }],
          },
        ],
      },
    ]);
  });

  it("retains operational log fields while excluding prompts, responses and errors", () => {
    const row = {
      request_id: "request-1",
      model: "model-1",
      status: "failure",
      spend: 0.01,
      total_tokens: 50,
      startTime: "2031-05-01T00:00:00Z",
      cache_hit: false,
      messages: ["private prompt"],
      response: "private response",
      metadata: { error_information: { error_message: "private error" } },
      proxy_server_request: { headers: { Authorization: "secret" } },
    };
    expect(projectToolResult("log", { data: [row], total: 1 }, [])).toEqual({
      data: [
        {
          request_id: "request-1",
          model: "model-1",
          status: "failure",
          spend: 0.01,
          total_tokens: 50,
          startTime: "2031-05-01T00:00:00Z",
          cache_hit: false,
        },
      ],
      total: 1,
    });
  });

  it.each(["team", "tag"])(
    "should aggregate all token counts across dates and %s usage without exposing private fields",
    (entity) => {
      const days = [
        {
          date: "2031-05-01",
          spend: 10,
          api_requests: 3,
          total_tokens: 50,
          prompt_tokens: 40,
          completion_tokens: 10,
          cache_read_input_tokens: 15,
          cache_creation_input_tokens: 5,
        },
        {
          date: "2031-05-02",
          spend: 20,
          api_requests: 4,
          total_tokens: 75,
          prompt_tokens: 60,
          completion_tokens: 15,
          cache_read_input_tokens: 20,
          cache_creation_input_tokens: 10,
        },
      ];
      const totals = {
        total_spend: 30,
        total_api_requests: 7,
        total_tokens: 125,
        total_prompt_tokens: 100,
        total_completion_tokens: 25,
        total_cache_read_input_tokens: 35,
        total_cache_creation_input_tokens: 15,
      };
      const raw = {
        metadata: { ...totals, secret: "private" },
        results: days.map(({ date, ...metrics }) => ({
          date,
          metrics,
          breakdown: {
            models: { "model captured-token": { metrics, messages: ["private prompt"] } },
            providers: { "provider captured-token": { metrics, messages: ["private prompt"] } },
            entities: {
              [`${entity} sk-secret`]: {
                metrics,
                metadata: { team_alias: "Team captured-token", secret: "private" },
              },
            },
            api_keys: { "sk-private-key": { metrics, metadata: { key_alias: "worker", raw_key: "private" } } },
          },
        })),
      };
      const combined = {
        spend: 30,
        api_requests: 7,
        total_tokens: 125,
        prompt_tokens: 100,
        completion_tokens: 25,
        cache_read_input_tokens: 35,
        cache_creation_input_tokens: 15,
      };
      const expected = {
        totals,
        daily: days,
        breakdown_limit: 20,
        top_models: [{ name: "model [redacted]", ...combined }],
        top_providers: [{ name: "provider [redacted]", ...combined }],
        top_entities: [{ name: `${entity} [redacted]`, alias: "Team [redacted]", ...combined }],
        top_api_keys: [{ name: "[redacted]", alias: "worker", ...combined }],
      };
      expect(projectToolResult("usage", raw, ["captured-token"])).toEqual(expected);
    },
  );

  it.each([undefined, null, 0])("should preserve zero or unavailable token counts reported as %s", (count) => {
    const tokens = {
      prompt_tokens: count,
      completion_tokens: count,
      cache_read_input_tokens: count,
      cache_creation_input_tokens: count,
    };
    const metrics = { spend: 0, api_requests: 0, total_tokens: 0, ...tokens };
    const metadata = {
      total_prompt_tokens: count,
      total_completion_tokens: count,
      total_cache_read_input_tokens: count,
      total_cache_creation_input_tokens: count,
    };
    const daily = ["2031-05-01", "2031-05-02"].map((date) => ({ date, ...metrics }));
    const raw = {
      metadata,
      results: daily.map(({ date }) => ({ date, metrics, breakdown: { models: { model: { metrics } } } })),
    };
    const expected = {
      totals: metadata,
      daily,
      breakdown_limit: 20,
      top_models: [
        {
          name: "model",
          ...metrics,
          prompt_tokens: count ?? undefined,
          completion_tokens: count ?? undefined,
          cache_read_input_tokens: count ?? undefined,
          cache_creation_input_tokens: count ?? undefined,
        },
      ],
      top_providers: [],
      top_entities: [],
      top_api_keys: [],
    };
    expect(projectToolResult("usage", raw, [])).toEqual(expected);
  });

  it.each([0, 1, 2])(
    "should leave partially reported token counts unavailable when day %s omits them",
    (missingDay) => {
      const metrics = { spend: 1, api_requests: 1, total_tokens: 20 };
      const tokens = {
        prompt_tokens: 15,
        completion_tokens: 5,
        cache_read_input_tokens: 6,
        cache_creation_input_tokens: 2,
      };
      const raw = {
        results: Array.from({ length: 3 }, (_, index) => {
          const dayMetrics = index === missingDay ? metrics : { ...metrics, ...tokens };
          return {
            date: `2031-05-0${index + 1}`,
            metrics: dayMetrics,
            breakdown: { models: { model: { metrics: dayMetrics } }, entities: { entity: { metrics: dayMetrics } } },
          };
        }),
      };
      const result = projectToolResult("usage", raw, []);
      expect(result).toHaveProperty("top_models", [{ name: "model", spend: 3, api_requests: 3, total_tokens: 60 }]);
      expect(result).toHaveProperty("top_entities", [{ name: "entity", spend: 3, api_requests: 3, total_tokens: 60 }]);
      expect(result).toHaveProperty("totals", undefined);
    },
  );

  it("should bound usage breakdowns after ranking while keeping complete totals", () => {
    const metrics = {
      spend: 300,
      api_requests: 30,
      total_tokens: 900,
      prompt_tokens: 700,
      completion_tokens: 200,
      cache_read_input_tokens: 250,
      cache_creation_input_tokens: 100,
    };
    const result = projectToolResult(
      "usage",
      {
        metadata: { total_spend: metrics.spend },
        results: [
          {
            date: "2031-05-01",
            metrics,
            breakdown: {
              providers: Object.fromEntries(
                Array.from({ length: 30 }, (_, index) => [
                  `provider-${index}`,
                  { metrics: { ...metrics, spend: index } },
                ]),
              ),
            },
          },
        ],
      },
      [],
    );
    expect(result).toMatchObject({ totals: { total_spend: 300 }, daily: [{ ...metrics }], breakdown_limit: 20 });
    expect(result).toHaveProperty(
      "top_providers",
      Array.from({ length: 20 }, (_, index) => ({
        name: `provider-${29 - index}`,
        ...metrics,
        spend: 29 - index,
      })),
    );
  });

  it("should keep the usage output size limit when token breakdowns enlarge the response", () => {
    const raw = {
      results: Array.from({ length: 200 }, () => ({
        date: "2031-05-01",
        metrics: {
          spend: 1,
          api_requests: 1,
          total_tokens: 20,
          prompt_tokens: 15,
          completion_tokens: 5,
          cache_read_input_tokens: 6,
          cache_creation_input_tokens: 2,
        },
      })),
    };
    expect(projectToolResult("usage", raw, [])).toEqual({
      notice: "The result is too large to summarize safely. Use a smaller page or narrower filters.",
    });
  });

  it("bounds list and string sizes and rejects results that exceed the overall output budget", () => {
    const raw = {
      users: Array.from({ length: 80 }, (_, index) => ({ user_id: String(index), user_alias: "x".repeat(800) })),
    };
    const result = projectToolResult("user", raw, []) as { users: { user_id: string; user_alias: string }[] };
    expect(result.users).toHaveLength(50);
    expect(result.users[0].user_alias).toHaveLength(400);
    const huge = {
      data: Array.from({ length: 50 }, () => ({
        request_id: "x".repeat(400),
        model: "y".repeat(400),
        user: "z".repeat(400),
      })),
    };
    expect(projectToolResult("log", huge, [])).toEqual({
      notice: "The result is too large to summarize safely. Use a smaller page or narrower filters.",
    });
  });

  it("returns an actionable result when the gateway shape cannot be projected", () => {
    expect(projectToolResult("log", [{ request_id: "request-1" }], [])).toEqual({
      notice: "The gateway returned an unsupported result shape. Open the resource to inspect it.",
    });
  });

  it("extracts a generated key only from the explicit response field", () => {
    expect(generatedKey({ key: "private-generated-value", metadata: { key: "other" } })).toBe(
      "private-generated-value",
    );
    expect(generatedKey({ metadata: { key: "other" } })).toBeUndefined();
    expect(generatedKey({ key: "" })).toBeUndefined();
  });
});
