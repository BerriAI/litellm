import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { clearTokenCookies } from "@/utils/cookieUtils";
import * as Networking from "./networking";
import { uiHref } from "@/utils/uiHref";

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
  getCookie: vi.fn(),
  storeLoginToken: vi.fn(),
}));

describe("networking - expired session handling", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const loadFreshHandleError = async () => {
    vi.resetModules();
    const fresh = await import("./networking");
    return fresh.handleError;
  };

  const stubLocation = (pathname: string, search: string, hash: string) => {
    const location = { pathname, search, hash, href: "" };
    vi.stubGlobal("window", { location });
    return location;
  };

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the query string and hash on the redirect after session expiry", async () => {
    const handleError = await loadFreshHandleError();
    const location = stubLocation("/ui/api-keys/", "?filter_team=t1&page=2", "#row-3");

    await handleError("Authentication Error - Expired Key");

    expect(location.href).toBe("/ui/api-keys/?filter_team=t1&page=2#row-3");
    expect(clearTokenCookies).toHaveBeenCalledOnce();
  });

  it("does not navigate or clear cookies for other errors", async () => {
    const handleError = await loadFreshHandleError();
    const location = stubLocation("/ui/api-keys/", "?filter_team=t1&page=2", "");

    await handleError("Some other error");

    expect(location.href).toBe("");
    expect(clearTokenCookies).not.toHaveBeenCalled();
  });

  it("should surface backend detail error when updateSSOSettings fails", async () => {
    expect.hasAssertions();

    const backendError = {
      detail: {
        error: "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature.",
      },
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue(backendError),
    } as any);

    global.fetch = mockFetch as any;

    try {
      await Networking.updateSSOSettings("token", { some: "setting" });
    } catch (error) {
      const thrownError = error as any;
      expect(thrownError).toBeInstanceOf(Error);
      expect(thrownError.message).toBe(backendError.detail.error);
      expect(thrownError.detail).toEqual(backendError.detail);
      expect(thrownError.rawError).toEqual(backendError);
    }

    expect(mockFetch).toHaveBeenCalledOnce();
  });
});

describe("loginCall - storeLoginToken integration", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("calls storeLoginToken when response includes token", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ redirect_url: "/ui/?login=success", token: "my-jwt" }),
    }) as any;
    const { storeLoginToken } = await import("@/utils/cookieUtils");
    await Networking.loginCall("admin", "pass");
    expect(storeLoginToken).toHaveBeenCalledWith("my-jwt");
  });

  it("does not call storeLoginToken when response has no token", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ redirect_url: "/ui/?login=success" }),
    }) as any;
    const { storeLoginToken } = await import("@/utils/cookieUtils");
    await Networking.loginCall("admin", "pass");
    expect(storeLoginToken).not.toHaveBeenCalled();
  });
});

describe("modelInfoCall", () => {
  let currentFetch: typeof global.fetch;

  beforeEach(() => {
    currentFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = currentFetch;
  });

  it("sends the exact model name as the model query param and leaves search alone", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, json: vi.fn().mockResolvedValue({ data: [] }) } as any);
    global.fetch = mockFetch as any;

    await Networking.modelInfoCall(
      "token",
      "user",
      "Admin",
      2,
      25,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      true,
      "gpt-4",
    );

    const parsed = new URL(mockFetch.mock.calls[0][0] as string, "http://example.com");
    expect(parsed.pathname).toBe("/v2/model/info");
    expect(parsed.searchParams.get("model")).toBe("gpt-4");
    expect(parsed.searchParams.has("search")).toBe(false);
    expect(parsed.searchParams.get("page")).toBe("2");
    expect(parsed.searchParams.get("exclude_auto_routers")).toBe("true");
  });
});

describe("UI config and public endpoints", () => {
  const originalFetch = global.fetch;

  const setupMockFetch = (responses: Array<{ url: string; data: any }>) => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      const response = responses.find((r) => url.includes(r.url));
      if (response) {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue(response.data),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        json: vi.fn().mockResolvedValue({}),
      } as any);
    });
    global.fetch = mockFetch as any;
    return mockFetch;
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("should use proxyBaseURL and server_root_path for /public/providers/fields when server_root_path is defined", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/providers/fields", data: [] },
    ]);

    // First call getUiConfig to set up proxyBaseUrl
    await Networking.getUiConfig();

    // Then call the public endpoint
    await Networking.getProviderCreateMetadata();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find((call) =>
      (call[0] as string).includes("/public/providers/fields"),
    );
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/api/v1/public/providers/fields");
  });

  it("should use proxyBaseURL and server_root_path for /public/model_hub/info when server_root_path is defined", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/model_hub/info", data: {} },
    ]);

    await Networking.getUiConfig();
    await Networking.getPublicModelHubInfo();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find((call) =>
      (call[0] as string).includes("/public/model_hub/info"),
    );
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/api/v1/public/model_hub/info");
  });

  it("should use proxyBaseURL and server_root_path for /public/model_hub when server_root_path is defined", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/model_hub", data: [] },
    ]);

    await Networking.getUiConfig();
    await Networking.modelHubPublicModelsCall();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find(
      (call) => (call[0] as string).includes("/public/model_hub") && !(call[0] as string).includes("/info"),
    );
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/api/v1/public/model_hub");
  });

  it("should use proxyBaseURL and server_root_path for /public/agent_hub when server_root_path is defined", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/agent_hub", data: [] },
    ]);

    await Networking.getUiConfig();
    await Networking.agentHubPublicModelsCall();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find((call) => (call[0] as string).includes("/public/agent_hub"));
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/api/v1/public/agent_hub");
  });

  it("should use proxyBaseURL and server_root_path for /public/mcp_hub when server_root_path is defined", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/mcp_hub", data: [] },
    ]);

    await Networking.getUiConfig();
    await Networking.mcpHubPublicServersCall();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find((call) => (call[0] as string).includes("/public/mcp_hub"));
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/api/v1/public/mcp_hub");
  });

  it("should not include server_root_path when it is root path", async () => {
    const uiConfig = {
      server_root_path: "/",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([
      { url: "/litellm/.well-known/litellm-ui-config", data: uiConfig },
      { url: "/public/providers/fields", data: [] },
    ]);

    await Networking.getUiConfig();
    await Networking.getProviderCreateMetadata();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    const publicEndpointCall = mockFetch.mock.calls.find((call) =>
      (call[0] as string).includes("/public/providers/fields"),
    );
    expect(publicEndpointCall).toBeDefined();
    const calledUrl = publicEndpointCall![0] as string;
    expect(calledUrl).toBe("https://example.com/public/providers/fields");
  });

  it("should return UI config from getUiConfig", async () => {
    const uiConfig = {
      server_root_path: "/api/v1",
      proxy_base_url: "https://example.com",
    };

    const mockFetch = setupMockFetch([{ url: "/litellm/.well-known/litellm-ui-config", data: uiConfig }]);

    const result = await Networking.getUiConfig();

    expect(mockFetch).toHaveBeenCalledOnce();
    expect(result).toEqual(uiConfig);
    const configCall = mockFetch.mock.calls.find((call) =>
      (call[0] as string).includes("/litellm/.well-known/litellm-ui-config"),
    );
    expect(configCall).toBeDefined();
  });

  it("updates serverRootPath so path-based nav links carry the root path", async () => {
    const uiConfig = {
      server_root_path: "/litellm",
      proxy_base_url: "https://example.com",
    };

    setupMockFetch([{ url: "/litellm/.well-known/litellm-ui-config", data: uiConfig }]);

    await Networking.getUiConfig();

    expect(Networking.serverRootPath).toBe("/litellm");
    expect(uiHref("api-reference")).toBe("/litellm/ui/api-reference");
  });
});

describe("individualModelHealthCheckCall", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("should call /health with model_id query param so health checks run by deployment id", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        healthy_count: 1,
        unhealthy_count: 0,
        healthy_endpoints: [],
        unhealthy_endpoints: [],
      }),
    } as any);
    global.fetch = mockFetch as any;

    await Networking.individualModelHealthCheckCall("token-123", "deployment-abc-456");

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0];
    const urlStr = typeof url === "string" ? url : (url as Request).url;
    expect(urlStr).toContain("health");
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);
    expect(parsed.searchParams.get("model_id")).toBe("deployment-abc-456");
    expect(parsed.searchParams.has("model")).toBe(false);
  });

  it("should encode model_id in URL", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        healthy_count: 0,
        unhealthy_count: 0,
        healthy_endpoints: [],
        unhealthy_endpoints: [],
      }),
    } as any);
    global.fetch = mockFetch as any;

    await Networking.individualModelHealthCheckCall("token", "id/with/slashes");

    const [url] = mockFetch.mock.calls[0];
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);
    expect(parsed.searchParams.get("model_id")).toBe("id/with/slashes");
  });
});

describe("teamInfoCall", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("should URL-encode team_id query param to handle special characters safely", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      text: vi.fn().mockResolvedValue(JSON.stringify({ team_id: "team with spaces & special?chars" })),
    } as any);
    global.fetch = mockFetch as any;

    const teamID = "team with spaces & special?chars";
    await Networking.teamInfoCall("token", teamID);

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0];
    const urlStr = typeof url === "string" ? url : (url as Request).url;
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);

    expect(urlStr).toContain("/team/info");
    // Special characters are encoded (not present raw) and round-trip back to the original
    expect(urlStr).not.toContain("team with spaces");
    expect(parsed.searchParams.get("team_id")).toBe(teamID);
  });

  it("should not append team_id when teamID is null", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      text: vi.fn().mockResolvedValue("{}"),
    } as any);
    global.fetch = mockFetch as any;

    await Networking.teamInfoCall("token", null);

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0];
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);
    expect(parsed.searchParams.has("team_id")).toBe(false);
  });
});

describe("uiSpendLogsCall exclude_internal_health_checks serialization", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const mockOkFetch = () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ data: [], total: 0, page: 1, page_size: 50, total_pages: 0 }),
    } as any);
    global.fetch = mockFetch as any;
    return mockFetch;
  };

  const callWith = (params: Parameters<typeof Networking.uiSpendLogsCall>[0]["params"]) =>
    Networking.uiSpendLogsCall({
      accessToken: "token",
      start_date: "2026-01-01 00:00:00",
      end_date: "2026-01-02 00:00:00",
      params,
    });

  const lastUrl = (mockFetch: ReturnType<typeof vi.fn>) => {
    const [url] = mockFetch.mock.calls.at(-1) ?? [];
    return new URL(url as string, "http://example.com");
  };

  it("appends exclude_internal_health_checks=true when the toggle is on", async () => {
    const mockFetch = mockOkFetch();

    await callWith({ exclude_internal_health_checks: true });

    expect(lastUrl(mockFetch).searchParams.get("exclude_internal_health_checks")).toBe("true");
  });

  it("omits exclude_internal_health_checks when the toggle is off", async () => {
    const mockFetch = mockOkFetch();

    await callWith({ exclude_internal_health_checks: false });

    expect(lastUrl(mockFetch).searchParams.has("exclude_internal_health_checks")).toBe(false);
  });

  it("omits exclude_internal_health_checks when the param is absent", async () => {
    const mockFetch = mockOkFetch();

    await callWith({});

    expect(lastUrl(mockFetch).searchParams.has("exclude_internal_health_checks")).toBe(false);
  });
});

describe("sessionSpendLogsCall", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("should request the first page with defaults so the caller can page through the session", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ data: [], total: 0, page: 1, page_size: 100, total_pages: 1 }),
    } as any);
    global.fetch = mockFetch as any;

    await Networking.sessionSpendLogsCall("token", "session-123");

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0];
    const urlStr = typeof url === "string" ? url : (url as Request).url;
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);

    expect(urlStr).toContain("/spend/logs/session/ui");
    expect(parsed.searchParams.get("session_id")).toBe("session-123");
    expect(parsed.searchParams.get("page")).toBe("1");
    expect(parsed.searchParams.get("page_size")).toBe("100");
  });

  it("should pass explicit page and page_size query params for later pages", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ data: [], total: 250, page: 3, page_size: 100, total_pages: 3 }),
    } as any);
    global.fetch = mockFetch as any;

    await Networking.sessionSpendLogsCall("token", "session-123", 3, 100);

    const [url] = mockFetch.mock.calls[0];
    const parsed = typeof url === "string" ? new URL(url, "http://example.com") : new URL((url as Request).url);
    expect(parsed.searchParams.get("page")).toBe("3");
    expect(parsed.searchParams.get("page_size")).toBe("100");
  });
});

describe("buildModelGroupTestRequest", () => {
  it("builds a chat completion request with NO max_tokens (reasoning models 400 on a tiny cap)", () => {
    const { path, body } = Networking.buildModelGroupTestRequest("o3", "chat");
    expect(path).toBe("/v1/chat/completions");
    expect(body).toEqual({ model: "o3", messages: [{ role: "user", content: "test from litellm" }] });
    expect(body).not.toHaveProperty("max_tokens");
    expect(body).not.toHaveProperty("max_completion_tokens");
  });

  it("builds an embeddings request for embedding mode", () => {
    const { path, body } = Networking.buildModelGroupTestRequest("text-embedding-3-small", "embedding");
    expect(path).toBe("/v1/embeddings");
    expect(body).toEqual({ model: "text-embedding-3-small", input: "test from litellm" });
  });

  it("adds classifier request parameters to a chat probe", () => {
    const { body } = Networking.buildModelGroupTestRequest("gpt-5-mini", "chat", { reasoning_effort: "low" });
    expect(body).toEqual({
      model: "gpt-5-mini",
      messages: [{ role: "user", content: "test from litellm" }],
      reasoning_effort: "low",
    });
  });
});

describe("testMCPToolsListRequest auth headers", () => {
  const originalFetch = global.fetch;

  const captureFetch = () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: vi.fn().mockResolvedValue({ tools: [] }),
    } as any);
    global.fetch = mockFetch as any;
    return mockFetch;
  };

  const sentHeaders = (mockFetch: ReturnType<typeof vi.fn>): Record<string, string> =>
    (mockFetch.mock.calls[0][1] as RequestInit).headers as Record<string, string>;

  afterEach(() => {
    Networking.setGlobalLitellmHeaderName("Authorization");
    global.fetch = originalFetch;
  });

  it("sends the litellm key under a custom litellm_key_header_name even when an upstream OAuth token uses Authorization", async () => {
    Networking.setGlobalLitellmHeaderName("x-litellm-key");
    const mockFetch = captureFetch();

    await Networking.testMCPToolsListRequest("sk-key", {}, "upstream-oauth-token");

    const headers = sentHeaders(mockFetch);
    expect(headers["x-litellm-key"]).toBe("Bearer sk-key");
    expect(headers["Authorization"]).toBe("Bearer upstream-oauth-token");
  });

  it("Bearer-prefixes x-litellm-api-key when it is the configured key header (raw values fail _get_bearer_token)", async () => {
    Networking.setGlobalLitellmHeaderName("x-litellm-api-key");
    const mockFetch = captureFetch();

    await Networking.testMCPToolsListRequest("sk-key", {}, "upstream-oauth-token");

    const headers = sentHeaders(mockFetch);
    expect(headers["x-litellm-api-key"]).toBe("Bearer sk-key");
    expect(headers["Authorization"]).toBe("Bearer upstream-oauth-token");
  });

  it("never clobbers the upstream OAuth token on default deployments", async () => {
    const mockFetch = captureFetch();

    await Networking.testMCPToolsListRequest("sk-key", {}, "upstream-oauth-token");

    const headers = sentHeaders(mockFetch);
    expect(headers["Authorization"]).toBe("Bearer upstream-oauth-token");
    expect(headers["x-litellm-api-key"]).toBe("sk-key");
  });

  it("sends the litellm key as the bearer on default deployments without an OAuth token", async () => {
    const mockFetch = captureFetch();

    await Networking.testMCPToolsListRequest("sk-key", {});

    const headers = sentHeaders(mockFetch);
    expect(headers["Authorization"]).toBe("Bearer sk-key");
  });
});

describe("fetchMCPServerHealth", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it.each([{ serverIds: undefined }, { serverIds: [] }, { serverIds: ["server one", "server&two"] }])(
    "opts into reachability while preserving requested servers: $serverIds",
    async ({ serverIds }) => {
      const mockFetch = vi.fn<typeof fetch>().mockResolvedValue(new Response("[]", { status: 200 }));
      global.fetch = mockFetch;

      await Networking.fetchMCPServerHealth("test-token", serverIds);

      expect(mockFetch).toHaveBeenCalledOnce();
      const url = new URL(String(mockFetch.mock.calls[0][0]), "http://localhost");
      expect(url.pathname).toMatch(/\/v1\/mcp\/server\/health$/);
      expect(url.searchParams.get("include_reachability")).toBe("true");
      expect(url.searchParams.getAll("server_ids")).toEqual(serverIds ?? []);
    },
  );
});

describe("getAutoRouterClassifierDefaultPromptCall", () => {
  const originalFetch = global.fetch;

  const captureFetch = () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: vi.fn().mockResolvedValue({ system_prompt: "rubric" }),
      text: vi.fn().mockResolvedValue(JSON.stringify({ system_prompt: "rubric" })),
    } as any);
    global.fetch = mockFetch as any;
    return mockFetch;
  };

  const requestedUrl = (mockFetch: ReturnType<typeof vi.fn>): string => String(mockFetch.mock.calls[0][0]);

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("sends renamed tiers as a JSON object so the rubric names them", async () => {
    const mockFetch = captureFetch();

    await Networking.getAutoRouterClassifierDefaultPromptCall("sk-key", 5, { SIMPLE: "Cheap" });

    const url = requestedUrl(mockFetch);
    expect(url).toContain("context_window_size=5");
    expect(decodeURIComponent(url)).toContain('tier_labels={"SIMPLE":"Cheap"}');
  });

  it("omits tier_labels entirely when nothing was renamed", async () => {
    const mockFetch = captureFetch();

    await Networking.getAutoRouterClassifierDefaultPromptCall("sk-key", 5);
    await Networking.getAutoRouterClassifierDefaultPromptCall("sk-key", 5, {});

    expect(requestedUrl(mockFetch)).not.toContain("tier_labels");
    expect(String(mockFetch.mock.calls[1][0])).not.toContain("tier_labels");
  });
});

describe("userListCall search serialization", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const mockOkFetch = () => {
    const emptyPage = { users: [], total: 0, page: 1, page_size: 25, total_pages: 0 };
    const body = JSON.stringify(emptyPage);
    const mockFetch = vi
      .fn()
      .mockResolvedValue(new Response(body, { headers: { "Content-Type": "application/json" } }));
    global.fetch = mockFetch;
    return mockFetch;
  };

  const lastParams = (mockFetch: ReturnType<typeof vi.fn>) => {
    const [url] = mockFetch.mock.calls.at(-1) ?? [];
    return new URL((url as Request).url).searchParams;
  };

  it("sends the combined search term as search, not user_email", async () => {
    const mockFetch = mockOkFetch();

    await Networking.userListCall("token", null, 1, 25, null, null, null, null, null, null, null, "a6f5c02b");

    expect(lastParams(mockFetch).get("search")).toBe("a6f5c02b");
    expect(lastParams(mockFetch).has("user_email")).toBe(false);
  });

  it("omits search when no search term is given and keeps user_email as before", async () => {
    const mockFetch = mockOkFetch();

    await Networking.userListCall("token", null, 1, 25, "ada@example.com");

    expect(lastParams(mockFetch).has("search")).toBe(false);
    expect(lastParams(mockFetch).get("user_email")).toBe("ada@example.com");
  });
});

describe("fetchMemoryList search serialization", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const mockOkFetch = () => {
    const emptyPage = { memories: [], total: 0 };
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, json: vi.fn().mockResolvedValue(emptyPage) } as any);
    global.fetch = mockFetch as any;
    return mockFetch;
  };

  const lastParams = (mockFetch: ReturnType<typeof vi.fn>) => {
    const [url] = mockFetch.mock.calls.at(-1) ?? [];
    return new URL(url as string, "http://example.com").searchParams;
  };

  it("sends the search box value as search and omits key_prefix and key", async () => {
    const mockFetch = mockOkFetch();

    await Networking.fetchMemoryList("token", { search: "mem-abc123", page: 1, pageSize: 50 });

    const params = lastParams(mockFetch);
    expect(params.get("search")).toBe("mem-abc123");
    expect(params.has("key_prefix")).toBe(false);
    expect(params.has("key")).toBe(false);
    expect(params.get("page")).toBe("1");
    expect(params.get("page_size")).toBe("50");
  });

  it("keeps key_prefix and key working when no search is given", async () => {
    const mockFetch = mockOkFetch();

    await Networking.fetchMemoryList("token", { keyPrefix: "user:" });
    expect(lastParams(mockFetch).get("key_prefix")).toBe("user:");
    expect(lastParams(mockFetch).has("search")).toBe(false);

    await Networking.fetchMemoryList("token", { key: "user:profile" });
    expect(lastParams(mockFetch).get("key")).toBe("user:profile");
    expect(lastParams(mockFetch).has("search")).toBe(false);
  });
});

describe("userFilterUICall", () => {
  let currentFetch: typeof global.fetch;

  beforeEach(() => {
    currentFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = currentFetch;
  });

  it("forwards the search param to /user/filter/ui", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, text: async () => "[]" } as any);
    global.fetch = mockFetch as any;

    await Networking.userFilterUICall("sk-test", new URLSearchParams({ search: "svc" }));

    const parsed = new URL(mockFetch.mock.calls[0][0] as string, "http://localhost");
    expect(parsed.pathname).toContain("/user/filter/ui");
    expect(parsed.searchParams.get("search")).toBe("svc");
  });
});

describe("schema-bound dashboard responses", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps null plugin metadata and source maps from the API", async () => {
    const plugin = {
      id: "plugin-1",
      name: "test-skill",
      enabled: true,
      version: null,
      description: null,
      created_at: null,
      updated_at: null,
      keywords: null,
      author: null,
      source: { source: "github", repo: "org/repo" },
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ plugins: [plugin], count: 1 }))));
    const result = await Networking.getClaudeCodePluginsList("explicit-token");
    expect(result.plugins[0]).toEqual(plugin);
  });

  it("validates agent metadata at the HTTP boundary", async () => {
    const agent = {
      agent_id: "agent-1",
      agent_name: "agent",
      enabled: true,
      execution_mode: "autonomous",
      identity_managed: false,
      jwt_auth_configured: false,
      agent_card_params: {},
      litellm_params: { model: 42 },
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify([agent]))));
    await expect(Networking.getAgentsList("explicit-token")).rejects.toThrow();
  });

  it("keeps nullable user fields without asserting they are strings", async () => {
    const user = { user_id: "user-1", user_email: null, user_role: null, created_at: null };
    const page = { users: [user], total: 1, page: 1, page_size: 25, total_pages: 1 };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(page))));
    const result = await Networking.userListCall("explicit-token");
    expect(result.users[0]).toEqual(user);
  });
});

describe("modelCostMap", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests the catalog-only map when catalogOnly is set and the full map otherwise", async () => {
    const mockFetch = vi.fn().mockImplementation(async () => new Response(JSON.stringify({})));
    vi.stubGlobal("fetch", mockFetch);

    await Networking.modelCostMap(true);
    await Networking.modelCostMap();

    expect(mockFetch.mock.calls[0][0]).toMatch(/\/public\/litellm_model_cost_map\?catalog_only=true$/);
    expect(mockFetch.mock.calls[1][0]).toMatch(/\/public\/litellm_model_cost_map$/);
  });
});
