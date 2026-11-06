import { describe, expect, it } from "vitest";
import {
  REDACTED_SECRET,
  agentAuthFormValues,
  agentAuthSupport,
  discoveryAuth,
  isAuthValueMissing,
  withAgentAuth,
} from "./agent_auth";

const BEARER_AND_HEADERS = { bearer: true, headers: true };
const HEADERS_ONLY = { bearer: false, headers: true };
const NO_AUTH_SECTION = { bearer: false, headers: false };

describe("agentAuthSupport", () => {
  it.each([
    ["a2a", undefined, undefined, BEARER_AND_HEADERS],
    ["langgraph", false, { custom_llm_provider: "langgraph" }, BEARER_AND_HEADERS],
    ["pydantic_ai_agents", true, undefined, HEADERS_ONLY],
    ["a2a", undefined, { custom_llm_provider: "watsonx" }, HEADERS_ONLY],
    ["a2a", undefined, { client_secret: REDACTED_SECRET }, HEADERS_ONLY],
    ["a2a", undefined, { azure_ad_token: REDACTED_SECRET }, HEADERS_ONLY],
    ["a2a", undefined, { azure_password: REDACTED_SECRET }, HEADERS_ONLY],
    ["a2a", undefined, { databricks_oauth: { client_id: "c", client_secret: REDACTED_SECRET } }, HEADERS_ONLY],
    ["a2a", undefined, { databricks_oauth: {}, api_key: REDACTED_SECRET }, BEARER_AND_HEADERS],
    ["azure_ai_foundry", null, { custom_llm_provider: "azure_ai" }, NO_AUTH_SECTION],
  ])(
    "%s with A2A form fields %j and saved params %j offers the auth the runtime sends",
    (agentType, usesA2AFormFields, saved, expected) => {
      expect(agentAuthSupport(agentType, usesA2AFormFields, saved)).toEqual(expected);
    },
  );
});

describe("agentAuthFormValues", () => {
  it("reads a saved key as a bearer token, even when the server redacted it", () => {
    const saved = {
      litellm_params: { api_key: REDACTED_SECRET },
      static_headers: { "x-tenant": "t1" },
      extra_headers: ["x-request-id"],
    };

    const bearer = {
      auth_type: "bearer_token",
      api_key: REDACTED_SECRET,
      static_headers: [{ header: "x-tenant", value: "t1" }],
      extra_headers: ["x-request-id"],
    };

    expect(agentAuthFormValues(saved)).toEqual(bearer);
  });

  it.each([[{ litellm_params: { api_key: null } }], [{ litellm_params: {} }], [{ litellm_params: null }], [{}]])(
    "reads an agent without a key as None: %j",
    (agent) => {
      const none = { auth_type: "none", api_key: "", static_headers: [], extra_headers: [] };

      expect(agentAuthFormValues(agent)).toEqual(none);
    },
  );
});

describe("withAgentAuth", () => {
  const SAVED_WITH_KEY = {
    litellm_params: { api_key: REDACTED_SECRET, guardrails: ["pii"], require_trace_id_on_calls_by_agent: true },
    static_headers: { "x-tenant": "t1" },
    extra_headers: ["x-request-id"],
  };
  const UNCHANGED_FORM = agentAuthFormValues(SAVED_WITH_KEY);

  it("adds a token typed on create next to the agent's other params", () => {
    const typed = { auth_type: "bearer_token", api_key: "sk-1", static_headers: [], extra_headers: [] };
    const payload = withAgentAuth(
      { agent_name: "a", litellm_params: { model: "langgraph/asst" } },
      typed,
      BEARER_AND_HEADERS,
    );

    expect(payload).toEqual({ agent_name: "a", litellm_params: { model: "langgraph/asst", api_key: "sk-1" } });
  });

  it("leaves the payload as built when the admin did not touch the auth section", () => {
    const built = { agent_name: "a", tpm_limit: 5 };

    expect(withAgentAuth(built, UNCHANGED_FORM, BEARER_AND_HEADERS, SAVED_WITH_KEY)).toEqual(built);
    expect(withAgentAuth(built, agentAuthFormValues({}), BEARER_AND_HEADERS)).toEqual(built);
  });

  it("clears the saved key on None and keeps every other saved param, even with a token typed earlier", () => {
    const payload = withAgentAuth(
      { agent_name: "a" },
      { ...UNCHANGED_FORM, auth_type: "none", api_key: "sk-typed-then-abandoned" },
      BEARER_AND_HEADERS,
      SAVED_WITH_KEY,
    );

    expect(payload).toEqual({
      agent_name: "a",
      litellm_params: { api_key: null, guardrails: ["pii"], require_trace_id_on_calls_by_agent: true },
    });
  });

  it("replaces the saved key on top of the params the form rebuilt", () => {
    const payload = withAgentAuth(
      { litellm_params: { model: "langgraph/asst" } },
      { ...UNCHANGED_FORM, api_key: "sk-new" },
      BEARER_AND_HEADERS,
      SAVED_WITH_KEY,
    );

    expect(payload).toEqual({ litellm_params: { model: "langgraph/asst", api_key: "sk-new" } });
  });

  it("sends changed headers by trimmed name, drops nameless rows, and sends an emptied list to clear it", () => {
    const payload = withAgentAuth(
      {},
      {
        ...UNCHANGED_FORM,
        static_headers: [
          { header: " x-tenant ", value: "t2" },
          { header: "x-empty-value" },
          { header: "  ", value: "orphan" },
        ],
        extra_headers: [],
      },
      HEADERS_ONLY,
      SAVED_WITH_KEY,
    );

    expect(payload).toEqual({ static_headers: { "x-tenant": "t2", "x-empty-value": "" }, extra_headers: [] });
  });

  it("leaves a provider's own api_key alone when the type has no auth section", () => {
    const payload = withAgentAuth(
      { litellm_params: { api_key: "azure-key" } },
      { auth_type: "none", api_key: "" },
      NO_AUTH_SECTION,
      { litellm_params: { api_key: REDACTED_SECRET } },
    );

    expect(payload).toEqual({ litellm_params: { api_key: "azure-key" } });
  });

  it.each([
    [
      {
        auth_type: "bearer_token",
        api_key: "sk-1",
        static_headers: [{ header: "x-a", value: "1" }],
        extra_headers: [],
      },
    ],
    [{ auth_type: "none", api_key: "", static_headers: [], extra_headers: ["x-request-id"] }],
  ])("round-trips through a newly saved agent: %j", (values) => {
    const saved = withAgentAuth({}, values, BEARER_AND_HEADERS);

    expect(
      agentAuthFormValues({
        litellm_params: saved.litellm_params,
        static_headers: saved.static_headers as Record<string, string> | undefined,
        extra_headers: saved.extra_headers as string[] | undefined,
      }),
    ).toEqual(values);
  });
});

describe("discoveryAuth", () => {
  it("sends a typed token and the static headers when creating an agent", () => {
    expect(
      discoveryAuth(
        { auth_type: "bearer_token", api_key: "sk-1", static_headers: [{ header: "x-tenant", value: "t1" }] },
        BEARER_AND_HEADERS,
      ),
    ).toEqual({ api_key: "sk-1", static_headers: { "x-tenant": "t1" } });
  });

  it("sends no token while the edit form still holds the redacted saved one", () => {
    expect(discoveryAuth({ auth_type: "bearer_token", api_key: REDACTED_SECRET }, BEARER_AND_HEADERS)).toEqual({});
  });

  it("sends no token when Authentication is None", () => {
    expect(discoveryAuth({ auth_type: "none", api_key: "sk-left-over" }, BEARER_AND_HEADERS)).toEqual({});
  });

  it("sends only static headers for a type without bearer auth", () => {
    expect(
      discoveryAuth(
        { auth_type: "bearer_token", api_key: "sk-1", static_headers: [{ header: "x-a", value: "1" }] },
        HEADERS_ONLY,
      ),
    ).toEqual({ static_headers: { "x-a": "1" } });
  });
});

describe("isAuthValueMissing", () => {
  it.each([
    ["none", "", false],
    ["bearer_token", "sk-1", false],
    ["bearer_token", "  ", true],
  ])("auth %s with value %j is missing: %j", (authType, value, expected) => {
    expect(isAuthValueMissing(value, { auth_type: authType, api_key: value })).toBe(expected);
  });
});
