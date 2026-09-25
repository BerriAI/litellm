import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatCompletion, ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions";
import { createGatewayClient } from "@/components/llm_calls/gateway_client";
import { registerAuthHeaderNameGetter } from "@/lib/http/runtime";
import { MAX_INPUT_LENGTH, resolveInferenceTarget, runLiteAdmin, type LiteAdminOptions } from "./agent";
import type { ActionResult, LiteAdminAction } from "./operations";

const { managementFetch } = vi.hoisted(() => ({ managementFetch: vi.fn<typeof fetch>() }));
vi.mock("@/components/networking", async () => {
  const { createApiClient } = await import("@/lib/http/client");
  const { getAuthHeaderName } = await import("@/lib/http/runtime");
  const getProxyBaseUrl = () => "https://management.example/root";
  return {
    getProxyBaseUrl,
    apiClient: createApiClient({ getBaseUrl: getProxyBaseUrl, getAuthHeaderName, fetchImpl: managementFetch }),
  };
});

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
const keyHash = "a".repeat(64);
const limits = { max_budget: null, budget_duration: null, rpm_limit: null, tpm_limit: null };
const keyFields = {
  key_alias: "New key",
  team_id: null,
  user_id: null,
  models: null,
  ...limits,
  budget_id: null,
  duration: null,
};
const teamFields = { team_alias: "Engineering", organization_id: null, models: null, ...limits };
const userFields = {
  user_email: "admin@example.com",
  user_alias: null,
  user_role: "internal_user",
  models: null,
  ...limits,
};
const pages = { page: 1, page_size: 20, search: null };
const teamsArgs = { ...pages, organization_id: null };
const dates = { start_date: "2031-05-01", end_date: "2031-05-02" };
const call = (name: string, args: unknown, id = "call") => ({
  id,
  type: "function" as const,
  function: { name, arguments: JSON.stringify(args) },
});
const completion = (
  calls: ReturnType<typeof call>[] = [],
  content: string | null = calls.length ? null : "Done",
): ChatCompletion => ({
  id: "completion",
  object: "chat.completion",
  created: 0,
  model: "selected-model",
  choices: [
    {
      index: 0,
      logprobs: null,
      finish_reason: calls.length ? "tool_calls" : "stop",
      message: { role: "assistant", refusal: null, content, tool_calls: calls },
    },
  ],
});

function transport(responses: ChatCompletion[], statuses: number | readonly number[] = 200) {
  const requests: ChatCompletionCreateParamsNonStreaming[] = [];
  const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    requests.push(JSON.parse(String(init?.body)) as ChatCompletionCreateParamsNonStreaming);
    const response = responses[requests.length - 1];
    if (!response) throw new Error("Unexpected model request");
    return json(response, typeof statuses === "number" ? statuses : statuses[requests.length - 1]);
  });
  return {
    requests,
    fetchImpl,
    client: createGatewayClient({
      accessToken: "test-session",
      baseURL: "https://inference.example",
      fetch: fetchImpl,
    }),
  };
}

const options = () =>
  ({
    model: "selected-model",
    accessToken: "test-session",
    inferenceBaseUrl: "https://inference.example",
    messages: [{ role: "user" as const, content: "Check the team budget" }],
    signal: new AbortController().signal,
    assertCurrent: vi.fn<() => void>(),
    confirm: vi.fn(async (_action: LiteAdminAction) => true),
    onResult: vi.fn<(action: LiteAdminAction, result: ActionResult) => void>(),
    onMessage: vi.fn<(message: string) => void>(),
  }) satisfies LiteAdminOptions;

function deferred<Value>() {
  return Promise.withResolvers<Value>();
}

beforeEach(() => {
  vi.clearAllMocks();
  managementFetch.mockImplementation(async () =>
    json({ team_id: "team-1", team_alias: "Engineering", max_budget: 100 }),
  );
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  registerAuthHeaderNameGetter(() => "Authorization");
});

describe("fixed management operations", () => {
  const cases: readonly [string, string, "GET" | "POST", Record<string, unknown>, boolean][] = [
    ["keys_list", "/key/list", "GET", { ...pages, team_id: null, user_id: null }, false],
    ["key_info", "/key/info", "GET", { key: keyHash }, false],
    ["key_create", "/key/generate", "POST", keyFields, true],
    ["key_update", "/key/update", "POST", { ...keyFields, key: keyHash }, true],
    ["key_delete", "/key/delete", "POST", { keys: [keyHash] }, true],
    ["key_block", "/key/block", "POST", { key: keyHash }, true],
    ["key_unblock", "/key/unblock", "POST", { key: keyHash }, true],
    ["teams_list", "/v2/team/list", "GET", teamsArgs, false],
    ["team_info", "/team/info", "GET", { team_id: "team-1" }, false],
    ["team_create", "/team/new", "POST", teamFields, true],
    ["team_update", "/team/update", "POST", { ...teamFields, team_id: "team-1" }, true],
    ["team_delete", "/team/delete", "POST", { team_ids: ["team-1"] }, true],
    [
      "team_member_add",
      "/team/member_add",
      "POST",
      {
        team_id: "team-1",
        member: { user_id: "user-1", role: "user" },
        max_budget_in_team: null,
        budget_duration: null,
      },
      true,
    ],
    [
      "team_member_update",
      "/team/member_update",
      "POST",
      {
        team_id: "team-1",
        user_id: "user-1",
        role: "admin",
        max_budget_in_team: null,
        budget_duration: null,
        rpm_limit: null,
        tpm_limit: null,
      },
      true,
    ],
    ["team_member_delete", "/team/member_delete", "POST", { team_id: "team-1", user_id: "user-1" }, true],
    ["users_list", "/user/list", "GET", pages, false],
    ["user_info", "/v2/user/info", "GET", { user_id: "user-1" }, false],
    ["user_create", "/user/new", "POST", { ...userFields, user_id: null }, true],
    ["user_update", "/user/update", "POST", { ...userFields, user_id: "user-1" }, true],
    ["user_delete", "/user/delete", "POST", { user_ids: ["user-1"] }, true],
    ["budgets_list", "/management/v1/budgets", "GET", pages, false],
    ["budget_info", "/budget/info", "POST", { budgets: ["budget-1"] }, false],
    ["budget_create", "/budget/new", "POST", { ...limits, budget_id: null }, true],
    ["budget_update", "/budget/update", "POST", { ...limits, budget_id: "budget-1" }, true],
    ["budget_delete", "/budget/delete", "POST", { id: "budget-1" }, true],
    [
      "spend_report",
      "/global/spend/report",
      "GET",
      { ...dates, group_by: "team", api_key: null, team_id: null, internal_user_id: null, customer_id: null },
      false,
    ],
    ["team_spend_report", "/team/spend/report", "GET", { ...dates, team_id: "team-1" }, false],
    ["key_spend_report", "/key/spend/report", "GET", { ...dates, api_key: keyHash }, false],
    [
      "request_logs",
      "/spend/logs/ui",
      "GET",
      {
        ...dates,
        page: 1,
        page_size: 20,
        request_id: null,
        team_id: null,
        user_id: null,
        model: null,
        status_filter: "failure",
      },
      false,
    ],
  ];

  const routeCases = cases.map(([name, path, method, args, write]) => ({ name, path, method, args, write }));
  it.each(routeCases)(
    "$name uses its existing gateway route and session",
    async ({ name, path, method, args, write }) => {
      registerAuthHeaderNameGetter(() => "x-admin-session");
      const input = options();
      const model = transport([completion([call(name, args)]), completion()]);

      await runLiteAdmin(input, model.client);

      expect(managementFetch).toHaveBeenCalledTimes(1);
      const [url, init] = managementFetch.mock.calls[0];
      expect(new URL(String(url)).pathname).toBe(`/root${path}`);
      expect(new URL(String(url)).origin).toBe("https://management.example");
      expect(init?.method).toBe(method);
      expect(init?.signal).toBe(input.signal);
      expect(new Headers(init?.headers).get("x-admin-session")).toBe(`Bearer ${input.accessToken}`);
      expect(input.confirm).toHaveBeenCalledTimes(write ? 1 : 0);
      expect(input.onResult).toHaveBeenCalledTimes(write ? 1 : 0);
      if (write) {
        const action = input.confirm.mock.calls[0][0];
        expect(action.destructive).toBe(name.endsWith("_delete"));
        expect(action.arguments).toEqual(JSON.parse(String(init?.body)));
        expect(input.onResult).toHaveBeenCalledWith(action, { status: "completed" });
      }
      expect(input.onMessage).toHaveBeenCalledExactlyOnceWith("Done");
    },
  );

  it("moves a key to a team without clearing untouched policy and never creates a key with a new user", async () => {
    const update = { ...keyFields, key: keyHash, key_alias: null, team_id: "new-team" };
    const model = transport([
      completion([call("key_update", update, "key"), call("user_create", { ...userFields, user_id: null }, "user")]),
      completion(),
    ]);
    await runLiteAdmin(options(), model.client);
    expect(JSON.parse(String(managementFetch.mock.calls[0][1]?.body))).toEqual({ key: keyHash, team_id: "new-team" });
    expect(JSON.parse(String(managementFetch.mock.calls[1][1]?.body))).toEqual({
      user_email: "admin@example.com",
      user_role: "internal_user",
      auto_create_key: false,
    });
  });

  it("bounds team keys and forwards user searches through their actual endpoint parameters", async () => {
    const model = transport([
      completion([
        call("team_info", { team_id: "team-1" }, "team"),
        call("users_list", { ...pages, search: "admin@example.com" }, "users"),
      ]),
      completion(),
    ]);
    await runLiteAdmin(options(), model.client);
    expect(new URL(String(managementFetch.mock.calls[0][0])).searchParams.get("key_limit")).toBe("20");
    expect(new URL(String(managementFetch.mock.calls[1][0])).searchParams.get("search")).toBe("admin@example.com");
  });

  it("provides strict JSON schemas for nullable and transformed tool arguments", async () => {
    const model = transport([completion()]);
    await runLiteAdmin(options(), model.client);
    const tools = model.requests[0].tools!;
    for (const tool of tools) expect(tool.function.strict).toBe(true);
    const createUser = tools.find((tool) => tool.function.name === "user_create")!;
    const expectedSchema = {
      type: "object",
      additionalProperties: false,
      required: expect.arrayContaining(["user_id", "user_email", "models", "max_budget"]),
      properties: { user_id: { anyOf: [{ type: "string" }, { type: "null" }] } },
    };
    expect(createUser.function.parameters).toMatchObject(expectedSchema);
    expect(tools.find((tool) => tool.function.name === "team_member_add")?.function.parameters).toMatchObject({
      properties: { member: { type: "object", additionalProperties: false, required: ["user_id", "role"] } },
    });
  });

  it.each([
    ["arbitrary_request", { url: "https://other.example", token: "secret" }],
    ["key_delete", { keys: ["sk-raw-secret"] }],
    ["teams_list", { ...teamsArgs, page_size: 10000 }],
    ["team_info", { team_id: "team-1", url: "https://other.example" }],
    ["team_spend_report", { ...dates, start_date: "2031-05-03", team_id: "team-1" }],
    ["team_spend_report", { ...dates, end_date: "2033-05-01", team_id: "team-1" }],
  ])("rejects invalid %s arguments before review or management dispatch", async (name, args) => {
    const input = options();
    const model = transport([completion([call(String(name), args)]), completion()]);
    await runLiteAdmin(input, model.client);
    expect(managementFetch).not.toHaveBeenCalled();
    expect(input.confirm).not.toHaveBeenCalled();
    expect(input.onResult).not.toHaveBeenCalled();
  });
});

describe("SDK conversation and action lifecycle", () => {
  it("emits new assistant text before its action review without replaying input history", async () => {
    const events = vi.fn<(event: string) => void>();
    const input = options();
    input.onMessage.mockImplementation((text) => events(`assistant:${text}`));
    input.confirm.mockImplementation(async () => {
      events("review");
      return true;
    });
    input.onResult.mockImplementation(() => events("completed"));
    const model = transport([completion([call("key_create", keyFields)], "I'll create that key."), completion()]);
    await runLiteAdmin(
      { ...input, messages: [{ role: "assistant", content: "Old answer" }, ...input.messages] },
      model.client,
    );
    expect(events.mock.calls.flat()).toEqual([
      "assistant:I'll create that key.",
      "review",
      "completed",
      "assistant:Done",
    ]);
  });

  it("waits for approval and delivers a generated key only in the action result", async () => {
    const approval = deferred<boolean>();
    const reviewed = deferred<void>();
    const input = options();
    input.confirm.mockImplementation(async () => {
      reviewed.resolve();
      return approval.promise;
    });
    managementFetch.mockImplementation(async () =>
      json({ key: "sk-private-new-key", key_alias: "New key", metadata: { secret: "hidden" } }),
    );
    const model = transport([completion([call("key_create", keyFields)]), completion()]);
    const running = runLiteAdmin(input, model.client);
    await reviewed.promise;
    expect(managementFetch).not.toHaveBeenCalled();
    approval.resolve(true);
    await running;
    expect(input.onResult).toHaveBeenCalledExactlyOnceWith(input.confirm.mock.calls[0][0], {
      status: "completed",
      key: "sk-private-new-key",
    });
    expect(JSON.stringify(model.requests)).not.toContain("sk-private-new-key");
    expect(JSON.stringify(model.requests)).not.toContain("hidden");
  });

  it("cancels a review without dispatch or an action result", async () => {
    const input = options();
    input.confirm.mockResolvedValue(false);
    const model = transport([completion([call("key_create", keyFields)])]);
    await expect(runLiteAdmin(input, model.client)).rejects.toThrow("Action cancelled");
    expect(managementFetch).not.toHaveBeenCalled();
    expect(input.onResult).not.toHaveBeenCalled();
    expect(model.requests).toHaveLength(1);
  });

  it("aborts a pending review without dispatch even if it later resolves as approved", async () => {
    const controller = new AbortController();
    const approval = deferred<boolean>();
    const reviewed = deferred<void>();
    const input = { ...options(), signal: controller.signal };
    input.confirm.mockImplementation(async () => {
      reviewed.resolve();
      return approval.promise;
    });
    const model = transport([completion([call("key_create", keyFields)])]);
    const running = runLiteAdmin(input, model.client);
    const rejected = expect(running).rejects.toBeInstanceOf(Error);
    await reviewed.promise;
    controller.abort();
    approval.resolve(true);
    await rejected;
    expect(managementFetch).not.toHaveBeenCalled();
    expect(input.onResult).not.toHaveBeenCalled();
  });

  it("checks the current session again after approval", async () => {
    const input = options();
    input.confirm.mockImplementation(async () => {
      input.assertCurrent.mockImplementation(() => {
        throw new Error("Session changed");
      });
      return true;
    });
    const model = transport([completion([call("key_create", keyFields)])]);
    await expect(runLiteAdmin(input, model.client)).rejects.toThrow("Session changed");
    expect(managementFetch).not.toHaveBeenCalled();
    expect(input.onResult).not.toHaveBeenCalled();
  });

  it.each([200, 500])("does not publish an old session's write response (HTTP %s)", async (status) => {
    const input = options();
    managementFetch.mockImplementation(async () => {
      input.assertCurrent.mockImplementation(() => {
        throw new Error("Session changed");
      });
      return json({ key: "sk-old-session" }, status);
    });
    const model = transport([completion([call("key_create", keyFields)])]);
    await expect(runLiteAdmin(input, model.client)).rejects.toThrow("Session changed");
    expect(input.onResult).not.toHaveBeenCalled();
    expect(input.onMessage).not.toHaveBeenCalled();
    expect(model.requests).toHaveLength(1);
  });

  it("reports an uncertain write once and stops without retrying or sending its error to the model", async () => {
    managementFetch.mockImplementation(async () => json({ error: { message: "secret from provider" } }, 500));
    const input = options();
    const model = transport([completion([call("key_create", keyFields)])]);
    await expect(runLiteAdmin(input, model.client)).rejects.toThrow("could not be verified");
    expect(managementFetch).toHaveBeenCalledTimes(1);
    expect(input.onResult).toHaveBeenCalledExactlyOnceWith(input.confirm.mock.calls[0][0], {
      status: "unknown",
      message: "The change could not be verified. Check the resource before trying again.",
    });
    expect(model.requests).toHaveLength(1);
    expect(JSON.stringify(input.onResult.mock.calls)).not.toContain("secret from provider");
  });

  it("retains a completed write when the following model request fails", async () => {
    const input = options();
    const model = transport([completion([call("key_create", keyFields)]), completion()], [200, 500]);
    await expect(runLiteAdmin(input, model.client)).rejects.toBeInstanceOf(Error);
    expect(input.onResult).toHaveBeenCalledExactlyOnceWith(input.confirm.mock.calls[0][0], { status: "completed" });
    expect(managementFetch).toHaveBeenCalledTimes(1);
    expect(model.requests).toHaveLength(2);
  });

  it("returns only a bounded failure description for a failed lookup", async () => {
    managementFetch.mockImplementation(async () => json({ detail: "private failure" }, 403));
    const model = transport([completion([call("teams_list", teamsArgs)]), completion()]);
    await runLiteAdmin(options(), model.client);
    const failure = {
      operation: "teams_list",
      success: false,
      status: 403,
      message: "The gateway could not complete this lookup.",
    };
    expect(model.requests[1].messages.at(-1)).toMatchObject({
      role: "tool",
      content: JSON.stringify(failure),
    });
    expect(JSON.stringify(model.requests)).not.toContain("private failure");
  });

  it("limits history, omits action messages and uses the current date", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2031-05-01T00:00:00Z"));
    const model = transport([completion()]);
    const messages: LiteAdminOptions["messages"] = [
      ...Array.from({ length: 30 }, (_, index) => ({ role: "user" as const, content: String(index) })),
      { role: "tool", content: "private action result" },
    ];
    await runLiteAdmin({ ...options(), messages }, model.client);
    expect(model.requests[0].messages).toHaveLength(21);
    expect(model.requests[0].messages[0]).toMatchObject({
      role: "system",
      content: expect.stringContaining(new Date().toISOString().slice(0, 10)),
    });
    expect(model.requests[0].messages[1]).toEqual({ role: "user", content: "10" });
    expect(JSON.stringify(model.requests)).not.toContain("private action result");
  });

  it("rejects oversized messages before requesting inference", async () => {
    const model = transport([completion()]);
    await expect(
      runLiteAdmin(
        { ...options(), messages: [{ role: "user", content: "x".repeat(MAX_INPUT_LENGTH + 1) }] },
        model.client,
      ),
    ).rejects.toThrow("8,000");
    expect(model.fetchImpl).not.toHaveBeenCalled();
  });

  it("truncates a long prior assistant response while still accepting the next user request", async () => {
    const model = transport([completion()]);
    const input = options();
    await runLiteAdmin(
      { ...input, messages: [{ role: "assistant", content: "a".repeat(MAX_INPUT_LENGTH + 500) }, ...input.messages] },
      model.client,
    );
    expect(model.requests[0].messages[1]).toEqual({ role: "assistant", content: "a".repeat(MAX_INPUT_LENGTH) });
    expect(input.onMessage).toHaveBeenCalledExactlyOnceWith("Done");
  });

  it.each(["teams_list", "key_create"])("stops after six model completions containing %s", async (name) => {
    const input = options();
    const args = name === "teams_list" ? teamsArgs : keyFields;
    const model = transport(Array.from({ length: 6 }, () => completion([call(name, args)])));
    await runLiteAdmin(input, model.client);
    expect(model.requests).toHaveLength(6);
    expect(managementFetch).toHaveBeenCalledTimes(6);
    expect(input.onMessage).toHaveBeenCalledExactlyOnceWith(
      "I reached the step limit. Any completed actions remain applied; check the relevant page before continuing.",
    );
  });

  it("allows twelve actions and a final answer", async () => {
    const model = transport([
      completion(Array.from({ length: 12 }, (_, index) => call("teams_list", teamsArgs, String(index)))),
      completion(),
    ]);
    const input = options();
    await runLiteAdmin(input, model.client);
    expect(managementFetch).toHaveBeenCalledTimes(12);
    expect(input.onMessage).toHaveBeenCalledExactlyOnceWith("Done");
  });

  it("blocks a thirteenth tool in one completion", async () => {
    const model = transport([
      completion(Array.from({ length: 13 }, (_, index) => call("teams_list", teamsArgs, String(index)))),
    ]);
    await expect(runLiteAdmin(options(), model.client)).rejects.toThrow("action limit");
    expect(managementFetch).toHaveBeenCalledTimes(12);
    expect(model.requests).toHaveLength(1);
  });

  it.each([200, 503])(
    "counts prior results across rounds before reviewing a thirteenth action (HTTP %s)",
    async (status) => {
      managementFetch.mockImplementation(async () => json({}, status));
      const input = options();
      const model = transport([
        ...Array.from({ length: 4 }, (_, round) =>
          completion(Array.from({ length: 3 }, (_, index) => call("teams_list", teamsArgs, `${round}-${index}`))),
        ),
        completion([call("key_create", keyFields)]),
      ]);
      await expect(runLiteAdmin(input, model.client)).rejects.toThrow("action limit");
      expect(managementFetch).toHaveBeenCalledTimes(12);
      expect(input.confirm).not.toHaveBeenCalled();
      expect(model.requests).toHaveLength(5);
    },
  );

  it("counts SDK-rejected calls toward the action budget", async () => {
    const input = options();
    const model = transport([
      completion([
        ...Array.from({ length: 12 }, (_, index) => call("unknown_tool", {}, String(index))),
        call("key_create", keyFields, "valid"),
      ]),
    ]);
    await expect(runLiteAdmin(input, model.client)).rejects.toThrow("action limit");
    expect(managementFetch).not.toHaveBeenCalled();
    expect(input.confirm).not.toHaveBeenCalled();
  });

  it("does not retry failed model requests", async () => {
    const model = transport([completion()], 500);
    await expect(runLiteAdmin(options(), model.client)).rejects.toBeInstanceOf(Error);
    expect(model.fetchImpl).toHaveBeenCalledTimes(1);
  });

  it.each(["Authorization", "x-admin-session"])(
    "rejects inference redirects and uses %s in the production client",
    async (header) => {
      registerAuthHeaderNameGetter(() => header);
      const model = transport([completion()]);
      vi.stubGlobal("fetch", model.fetchImpl);
      await runLiteAdmin(options());
      const [url, init] = model.fetchImpl.mock.calls[0];
      expect(url).toBe("https://inference.example/chat/completions");
      expect(init?.redirect).toBe("error");
      expect(new Headers(init?.headers).get(header)).toBe("Bearer test-session");
      if (header !== "Authorization") expect(new Headers(init?.headers).has("Authorization")).toBe(false);
    },
  );
});

describe("inference destination", () => {
  const destinations = [
    ["", "/gateway", "https://dashboard.example/ui/", "https://dashboard.example/gateway", false],
    ["v1", "/gateway", "https://dashboard.example/ui/", "https://dashboard.example/v1", false],
    [
      "https://DASHBOARD.example:443/gateway",
      "/gateway",
      "https://dashboard.example/ui/",
      "https://dashboard.example/gateway",
      false,
    ],
    ["https://models.example/v1", "/gateway", "https://dashboard.example/ui/", "https://models.example/v1", true],
    ["//models.example/v1", "/gateway", "https://dashboard.example/ui/", "https://models.example/v1", true],
    ["", "https://management.example/root", "https://dashboard.example/ui/", "https://management.example/root", false],
    ["http://localhost:4001", "http://localhost:4000", "http://localhost:3000/ui", "http://localhost:4001/", true],
  ] as const;
  const destinationCases = destinations.map(([candidate, management, page, baseUrl, requiresConsent]) => ({
    candidate,
    management,
    page,
    baseUrl,
    requiresConsent,
  }));
  it.each(destinationCases)(
    "resolves $candidate and requires consent only for a distinct management origin",
    ({ candidate, management, page, baseUrl, requiresConsent }) => {
      expect(resolveInferenceTarget(candidate, management, page)).toEqual({ baseUrl, requiresConsent, error: null });
    },
  );

  it.each([
    "https://",
    "javascript:alert(1)",
    "https://user:secret@models.example",
    "https://models.example?secret",
    "https://models.example#secret",
    "https://models.example?",
    "https://models.example#",
    "http://models.example",
  ])("rejects unsafe inference configuration without echoing secrets", (candidate) => {
    const result = resolveInferenceTarget(candidate, "https://management.example", "https://dashboard.example/ui/");
    expect(result).toMatchObject({ baseUrl: null, requiresConsent: false, error: expect.any(String) });
    expect(result.error).not.toContain("secret");
  });

  it("rejects an HTTPS management downgrade even on an HTTP development page", () => {
    expect(
      resolveInferenceTarget("http://models.example", "https://management.example", "http://localhost/ui").baseUrl,
    ).toBeNull();
  });
});
