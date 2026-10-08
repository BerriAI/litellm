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
