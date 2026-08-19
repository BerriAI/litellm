import { afterEach, describe, expect, it, vi } from "vitest";
import { buildCreateKeyPayload, type BuildCreateKeyPayloadResult, type CreateKeyUiState } from "./createKeyPayload";

const baseUi: CreateKeyUiState = {
  keyOwner: "you",
  userID: "test-user",
  selectedAgentId: null,
  loggingSettings: [],
  disabledCallbacks: [],
  autoRotationEnabled: false,
  rotationInterval: "30d",
  modelAliases: {},
  routerSettings: null,
  budgetLimits: [],
  tagRateLimits: [],
  budgetFallbacks: {},
};

const build = (values: Record<string, unknown>, ui: Partial<CreateKeyUiState> = {}): BuildCreateKeyPayloadResult =>
  buildCreateKeyPayload(values, { ...baseUi, ...ui });

const payloadOf = (result: BuildCreateKeyPayloadResult): Record<string, unknown> => {
  expect(result.kind).toBe("ok");
  if (result.kind !== "ok") throw new Error("unreachable");
  return result.payload;
};

const aliasOnly = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  key_alias: "my-key",
  user_id: "test-user",
  duration: null,
  metadata: "{}",
  ...overrides,
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("always-present keys", () => {
  it("emits the eight keys the closed form sends, and nothing else", () => {
    const closedFormValues = {
      organization_id: undefined,
      team_id: undefined,
      key_alias: "my-key",
      models: [],
      key_type: "llm_api",
    };
    const closedFormPayload = {
      ...closedFormValues,
      user_id: "test-user",
      duration: null,
      metadata: "{}",
    };
    expect(payloadOf(build(closedFormValues))).toStrictEqual(closedFormPayload);
  });

  it("injects user_id, duration and metadata even when the form reported none of them", () => {
    expect(payloadOf(build({ key_alias: "my-key" }))).toStrictEqual(aliasOnly());
  });

  it("keeps a mounted-but-untouched field as an undefined-valued key rather than dropping it", () => {
    expect(payloadOf(build({ key_alias: "my-key", guardrails: undefined, tags: undefined }))).toStrictEqual(
      aliasOnly({ guardrails: undefined, tags: undefined }),
    );
  });
});

describe("duration", () => {
  it("forwards a typed duration by value", () => {
    expect(payloadOf(build({ key_alias: "my-key", duration: "45d" }))).toStrictEqual(aliasOnly({ duration: "45d" }));
  });

  it.each([
    ["an empty string", ""],
    ["a whitespace-only string", "   "],
    ["undefined", undefined],
  ])("coalesces %s to null", (_label, duration) => {
    expect(payloadOf(build({ key_alias: "my-key", duration }))).toStrictEqual(aliasOnly({ duration: null }));
  });

  it("does not coerce a non-string duration", () => {
    expect(() => build({ key_alias: "my-key", duration: 30 })).toThrow(TypeError);
  });
});

describe("key ownership", () => {
  it("overwrites user_id with the signed-in user when the key is owned by you", () => {
    expect(payloadOf(build({ key_alias: "my-key", user_id: "someone-else" }))).toStrictEqual(
      aliasOnly({ user_id: "test-user" }),
    );
  });

  it("leaves the form's user_id alone for another_user", () => {
    const expected = { key_alias: "my-key", user_id: "someone-else", duration: null, metadata: "{}" };
    expect(
      payloadOf(build({ key_alias: "my-key", user_id: "someone-else" }, { keyOwner: "another_user" })),
    ).toStrictEqual(expected);
  });

  it("adds the selected agent id for an agent-owned key", () => {
    const expected = { key_alias: "my-key", agent_id: "agent-1", duration: null, metadata: "{}" };
    expect(payloadOf(build({ key_alias: "my-key" }, { keyOwner: "agent", selectedAgentId: "agent-1" }))).toStrictEqual(
      expected,
    );
  });

  it("reports agent_required instead of building a payload when no agent is selected", () => {
    expect(build({ key_alias: "my-key" }, { keyOwner: "agent", selectedAgentId: null })).toStrictEqual({
      kind: "agent_required",
    });
  });

  it("stamps the alias into metadata as the service account id and sends no user_id", () => {
    expect(payloadOf(build({ key_alias: "svc-key" }, { keyOwner: "service_account" }))).toStrictEqual({
      key_alias: "svc-key",
      duration: null,
      metadata: '{"service_account_id":"svc-key"}',
    });
  });
});

describe("metadata", () => {
  it("re-serialises the parsed form value", () => {
    expect(payloadOf(build({ key_alias: "my-key", metadata: '{"team":"core"}' }))).toStrictEqual(
      aliasOnly({ metadata: '{"team":"core"}' }),
    );
  });

  it("falls back to an empty object and logs when the JSON is malformed", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(payloadOf(build({ key_alias: "my-key", metadata: "{not json" }))).toStrictEqual(aliasOnly());
    expect(consoleError).toHaveBeenCalledWith("Error parsing metadata:", expect.any(SyntaxError));
  });

  it("merges logging configs, dropping rows with no callback selected", () => {
    expect(
      payloadOf(
        build(
          { key_alias: "my-key", metadata: '{"team":"core"}' },
          { loggingSettings: [{ callback_name: "langfuse" }, { callback_name: "" }] },
        ),
      ),
    ).toStrictEqual(aliasOnly({ metadata: '{"team":"core","logging":[{"callback_name":"langfuse"}]}' }));
  });

  it("maps disabled callbacks from display names to internal names", () => {
    expect(payloadOf(build({ key_alias: "my-key" }, { disabledCallbacks: ["Langfuse"] }))).toStrictEqual(
      aliasOnly({ metadata: '{"litellm_disabled_callbacks":["langfuse"]}' }),
    );
  });

  it("keeps an array metadata as an array when stamping the service account id", () => {
    expect(
      payloadOf(build({ key_alias: "svc-key", metadata: '["a"]' }, { keyOwner: "service_account" })),
    ).toStrictEqual({ key_alias: "svc-key", duration: null, metadata: '["a"]' });
  });

  it("rejects a service account whose metadata parses to a primitive", () => {
    expect(() => build({ key_alias: "svc-key", metadata: "5" }, { keyOwner: "service_account" })).toThrow(TypeError);
  });

  it("keeps every metadata contributor in one object", () => {
    expect(
      payloadOf(
        build(
          { key_alias: "svc-key", metadata: '{"team":"core"}' },
          {
            keyOwner: "service_account",
            loggingSettings: [{ callback_name: "otel" }],
            disabledCallbacks: ["Datadog"],
          },
        ),
      ),
    ).toStrictEqual({
      key_alias: "svc-key",
      duration: null,
      metadata:
        '{"team":"core","service_account_id":"svc-key","logging":[{"callback_name":"otel"}],"litellm_disabled_callbacks":["datadog"]}',
    });
  });
});

describe("object_permission", () => {
  it("is absent when nothing contributes to it", () => {
    expect(payloadOf(build({ key_alias: "my-key", allowed_vector_store_ids: [] }))).toStrictEqual(
      aliasOnly({ allowed_vector_store_ids: [] }),
    );
  });

  it("moves selected vector stores off the top level", () => {
    expect(payloadOf(build({ key_alias: "my-key", allowed_vector_store_ids: ["vs-1"] }))).toStrictEqual(
      aliasOnly({ object_permission: { vector_stores: ["vs-1"] } }),
    );
  });

  it("splits an MCP selection into servers, access groups and toolsets", () => {
    expect(
      payloadOf(
        build({
          key_alias: "my-key",
          allowed_mcp_servers_and_groups: { servers: ["s-1"], accessGroups: ["g-1"], toolsets: ["t-1"] },
        }),
      ),
    ).toStrictEqual(
      aliasOnly({
        object_permission: { mcp_servers: ["s-1"], mcp_access_groups: ["g-1"], mcp_toolsets: ["t-1"] },
      }),
    );
  });

  it("omits the empty halves of an MCP selection", () => {
    expect(
      payloadOf(
        build({
          key_alias: "my-key",
          allowed_mcp_servers_and_groups: { servers: ["s-1"], accessGroups: [], toolsets: [] },
        }),
      ),
    ).toStrictEqual(aliasOnly({ object_permission: { mcp_servers: ["s-1"] } }));
  });

  it("leaves an all-empty MCP selection on the top level", () => {
    expect(
      payloadOf(
        build({ key_alias: "my-key", allowed_mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] } }),
      ),
    ).toStrictEqual(aliasOnly({ allowed_mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] } }));
  });

  it("nests configured MCP tool permissions", () => {
    expect(payloadOf(build({ key_alias: "my-key", mcp_tool_permissions: { "s-1": ["read"] } }))).toStrictEqual(
      aliasOnly({ object_permission: { mcp_tool_permissions: { "s-1": ["read"] } } }),
    );
  });

  it("always strips mcp_tool_permissions from the top level, even when empty", () => {
    expect(payloadOf(build({ key_alias: "my-key", mcp_tool_permissions: {} }))).toStrictEqual(aliasOnly());
  });

  it("lets a standalone access group list win over the one from the MCP selection", () => {
    expect(
      payloadOf(
        build({
          key_alias: "my-key",
          allowed_mcp_servers_and_groups: { servers: ["s-1"], accessGroups: ["from-selector"] },
          allowed_mcp_access_groups: ["standalone"],
        }),
      ),
    ).toStrictEqual(aliasOnly({ object_permission: { mcp_servers: ["s-1"], mcp_access_groups: ["standalone"] } }));
  });

  it("splits an agent selection into agents and agent access groups", () => {
    expect(
      payloadOf(build({ key_alias: "my-key", allowed_agents_and_groups: { agents: ["a-1"], accessGroups: ["ag-1"] } })),
    ).toStrictEqual(aliasOnly({ object_permission: { agents: ["a-1"], agent_access_groups: ["ag-1"] } }));
  });

  it("merges every source into a single object_permission", () => {
    const everySource = {
      key_alias: "my-key",
      allowed_vector_store_ids: ["vs-1"],
      allowed_mcp_servers_and_groups: { servers: ["s-1"], accessGroups: ["g-1"], toolsets: ["t-1"] },
      mcp_tool_permissions: { "s-1": ["read"] },
      allowed_agents_and_groups: { agents: ["a-1"], accessGroups: ["ag-1"] },
    };
    expect(payloadOf(build(everySource))).toStrictEqual(
      aliasOnly({
        object_permission: {
          vector_stores: ["vs-1"],
          mcp_servers: ["s-1"],
          mcp_access_groups: ["g-1"],
          mcp_toolsets: ["t-1"],
          mcp_tool_permissions: { "s-1": ["read"] },
          agents: ["a-1"],
          agent_access_groups: ["ag-1"],
        },
      }),
    );
  });
});

describe("premium and rotation flags", () => {
  it("drops disable_global_guardrails when it is off", () => {
    expect(payloadOf(build({ key_alias: "my-key", disable_global_guardrails: false }))).toStrictEqual(aliasOnly());
  });

  it("keeps disable_global_guardrails when it is on", () => {
    expect(payloadOf(build({ key_alias: "my-key", disable_global_guardrails: true }))).toStrictEqual(
      aliasOnly({ disable_global_guardrails: true }),
    );
  });

  it("adds the rotation fields only when auto rotation is enabled", () => {
    expect(
      payloadOf(build({ key_alias: "my-key" }, { autoRotationEnabled: true, rotationInterval: "7d" })),
    ).toStrictEqual(aliasOnly({ auto_rotate: true, rotation_interval: "7d" }));
  });

  it("sends no rotation fields when auto rotation is off", () => {
    expect(payloadOf(build({ key_alias: "my-key" }, { rotationInterval: "7d" }))).toStrictEqual(aliasOnly());
  });
});

describe("keys sourced from component state", () => {
  it("serialises model aliases", () => {
    expect(payloadOf(build({ key_alias: "my-key" }, { modelAliases: { fast: "gpt-4o-mini" } }))).toStrictEqual(
      aliasOnly({ aliases: '{"fast":"gpt-4o-mini"}' }),
    );
  });

  it("sends router settings that hold at least one value", () => {
    expect(
      payloadOf(build({ key_alias: "my-key" }, { routerSettings: { router_settings: { num_retries: 3 } } })),
    ).toStrictEqual(aliasOnly({ router_settings: { num_retries: 3 } }));
  });

  it("skips router settings whose every field is blank", () => {
    expect(
      payloadOf(
        build(
          { key_alias: "my-key" },
          { routerSettings: { router_settings: { num_retries: null, timeout: undefined, routing_strategy: "" } } },
        ),
      ),
    ).toStrictEqual(aliasOnly());
  });

  it("keeps only budget windows that carry both a duration and a limit", () => {
    expect(
      payloadOf(
        build(
          { key_alias: "my-key" },
          {
            budgetLimits: [
              { budget_duration: "1h", max_budget: 5 },
              { budget_duration: "", max_budget: 3 },
              { budget_duration: "7d", max_budget: null },
            ],
          },
        ),
      ),
    ).toStrictEqual(aliasOnly({ budget_limits: [{ budget_duration: "1h", max_budget: 5 }] }));
  });

  it("keeps a zero budget window rather than treating it as unset", () => {
    expect(
      payloadOf(build({ key_alias: "my-key" }, { budgetLimits: [{ budget_duration: "1h", max_budget: 0 }] })),
    ).toStrictEqual(aliasOnly({ budget_limits: [{ budget_duration: "1h", max_budget: 0 }] }));
  });

  it("omits budget_limits when no window is complete", () => {
    expect(
      payloadOf(build({ key_alias: "my-key" }, { budgetLimits: [{ budget_duration: "7d", max_budget: null }] })),
    ).toStrictEqual(aliasOnly());
  });

  it("reduces tag rows into a tag_rpm_limit map", () => {
    expect(
      payloadOf(
        build(
          { key_alias: "my-key" },
          {
            tagRateLimits: [
              { id: "r-1", tag: "prod", rpm_limit: 10 },
              { id: "r-2", tag: "  ", rpm_limit: 5 },
              { id: "r-3", tag: "dev", rpm_limit: null },
            ],
          },
        ),
      ),
    ).toStrictEqual(aliasOnly({ tag_rpm_limit: { prod: 10 } }));
  });

  it("omits tag_rpm_limit when no row is complete", () => {
    expect(
      payloadOf(build({ key_alias: "my-key" }, { tagRateLimits: [{ id: "r-1", tag: "", rpm_limit: 10 }] })),
    ).toStrictEqual(aliasOnly());
  });

  it("sends configured budget fallbacks", () => {
    expect(payloadOf(build({ key_alias: "my-key" }, { budgetFallbacks: { "gpt-4": ["gpt-4o"] } }))).toStrictEqual(
      aliasOnly({ budget_fallbacks: { "gpt-4": ["gpt-4o"] } }),
    );
  });
});

describe("budget duration", () => {
  it("turns the never-resets sentinel into null", () => {
    expect(payloadOf(build({ key_alias: "my-key", budget_duration: "none" }))).toStrictEqual(
      aliasOnly({ budget_duration: null }),
    );
  });

  it("forwards a real budget duration untouched", () => {
    expect(payloadOf(build({ key_alias: "my-key", budget_duration: "30d" }))).toStrictEqual(
      aliasOnly({ budget_duration: "30d" }),
    );
  });
});

describe("purity", () => {
  it("leaves the submitted form values untouched", () => {
    const values = {
      key_alias: "my-key",
      mcp_tool_permissions: { "s-1": ["read"] },
      allowed_vector_store_ids: ["vs-1"],
      disable_global_guardrails: false,
      duration: "",
    };
    const before = structuredClone(values);
    build(values);
    expect(values).toStrictEqual(before);
  });
});
