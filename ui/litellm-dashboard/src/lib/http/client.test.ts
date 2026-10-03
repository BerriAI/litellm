import { describe, it, expect, vi } from "vitest";
import {
  createApiClient,
  ApiError,
  deriveErrorMessage,
  extractProxyErrorMessage,
  unwrapProxyErrorMessage,
} from "./client";

const okResponse = (data: unknown): Response =>
  ({ ok: true, status: 200, text: async () => JSON.stringify(data) }) as unknown as Response;

const emptyResponse = (status: number): Response => ({ ok: true, status, text: async () => "" }) as unknown as Response;

const errorResponse = (status: number, body: unknown): Response =>
  ({ ok: false, status, text: async () => JSON.stringify(body) }) as unknown as Response;

describe("createApiClient", () => {
  it("builds the URL from base + path + query and sets the auth + JSON headers", async () => {
    const fetchImpl = vi.fn(async () => okResponse({ ok: true }));
    const client = createApiClient({
      getBaseUrl: () => "https://proxy.example",
      getAuthHeaderName: () => "x-litellm-key",
      fetchImpl,
    });

    const result = await client.get("/models", { accessToken: "sk-123", query: { team: "t1", page: 2 } });

    expect(result).toEqual({ ok: true });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe("https://proxy.example/models?team=t1&page=2");
    expect(init).toMatchObject({ method: "GET" });
    expect(init.headers).toEqual({
      "Content-Type": "application/json",
      "x-litellm-key": "Bearer sk-123",
    });
    expect(init.body).toBeUndefined();
  });

  it("forwards the browser credential policy", async () => {
    const fetchImpl = vi.fn(async () => okResponse({}));
    const client = createApiClient({ getBaseUrl: () => "https://proxy.example", fetchImpl });

    await client.get("/authorize/flow", { credentials: "include" });

    expect(fetchImpl.mock.calls[0][1].credentials).toBe("include");
  });

  it("JSON-serializes the body for writes", async () => {
    const fetchImpl = vi.fn(async () => okResponse({}));
    const client = createApiClient({ getBaseUrl: () => "", fetchImpl });

    await client.post("/model/new", { accessToken: "sk", body: { model_name: "gpt" } });

    const [, init] = fetchImpl.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ model_name: "gpt" }));
  });

  it("throws ApiError with the derived message and invokes onError on a non-2xx response", async () => {
    const fetchImpl = vi.fn(async () => errorResponse(403, { error: { message: "no access" } }));
    const onError = vi.fn();
    const client = createApiClient({ getBaseUrl: () => "", onError, fetchImpl });

    const promise = client.get("/keys", { accessToken: "sk" });

    await expect(promise).rejects.toBeInstanceOf(ApiError);
    await expect(promise).rejects.toMatchObject({ message: "no access", status: 403 });
    expect(onError).toHaveBeenCalledWith("no access");
  });

  it("unwraps an object-shaped detail ({detail:{error}}) rather than dumping the JSON envelope (FastAPI HTTPException shape)", async () => {
    const conflict = "A skill named 'gitlab' already exists. Update the existing skill instead of adding it again.";
    const fetchImpl = vi.fn(async () => errorResponse(409, { detail: { error: conflict } }));
    const onError = vi.fn();
    const client = createApiClient({ getBaseUrl: () => "", onError, fetchImpl });

    const promise = client.get("/claude-code/plugins", { accessToken: "sk" });

    await expect(promise).rejects.toMatchObject({ message: conflict, status: 409 });
    expect(onError).toHaveBeenCalledWith(conflict);
  });

  it.each([
    [
      503,
      "text/html; charset=utf-8",
      "<html><head><title>503 Service Temporarily Unavailable</title></head><body><h1>503 Service Temporarily Unavailable</h1></body></html><!-- a padding to disable MSIE and Chrome friendly error page -->",
    ],
    [503, "TEXT/HTML", "<h1>Unavailable</h1>"],
    [503, "application/xhtml+xml", '<?xml version="1.0"?><html>Unavailable</html>'],
    [502, "text/plain", "<html>Bad Gateway</html>"],
    [504, "", " \n<!DOCTYPE HTML><HTML><BODY>Timeout</BODY></HTML>"],
    [500, "application/json", '<HTML lang="en">Server error</HTML>'],
    [403, "text/html", "<p>Access denied</p>"],
  ])("normalizes HTML errors with status %i and content type %s", async (status, contentType, raw) => {
    const message =
      status === 503
        ? "Service temporarily unavailable (HTTP 503). Try again shortly. If this persists, contact your proxy administrator."
        : `The server returned an HTML error page (HTTP ${status}). Contact your proxy administrator if this persists.`;
    const onError = vi.fn();
    const client = createApiClient({
      getBaseUrl: () => "",
      onError,
      fetchImpl: async () => new Response(raw, { status, headers: { "Content-Type": contentType } }),
    });

    for (const request of [client.get, client.getBlob]) {
      const promise = request("/lens");
      await expect(promise).rejects.toBeInstanceOf(ApiError);
      await expect(promise).rejects.toMatchObject({ message, status, body: raw });
    }
    expect(onError).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenLastCalledWith(message);
  });

  it.each([
    ["upstream connection refused", "upstream connection refused", "text/plain"],
    ["Expected <model> in the request", "Expected <model> in the request", "text/plain"],
    ["", "HTTP 503", "text/plain"],
    ["", "HTTP 503", "text/html"],
    ["", "HTTP 503", "application/xhtml+xml"],
  ])("preserves body %j as %j with content type %s", async (raw, message, contentType) => {
    const client = createApiClient({
      getBaseUrl: () => "",
      fetchImpl: async () => new Response(raw, { status: 503, headers: { "Content-Type": contentType } }),
    });
    await expect(client.get("/lens")).rejects.toMatchObject({ message, status: 503, body: raw });
  });

  it("preserves structured 503 diagnostics even when the response is mislabeled HTML", async () => {
    const body = { detail: "Lens needs a connected Postgres database" };
    const onError = vi.fn();
    const client = createApiClient({
      getBaseUrl: () => "",
      onError,
      fetchImpl: async () =>
        new Response(JSON.stringify(body), { status: 503, headers: { "Content-Type": "text/html" } }),
    });
    await expect(client.get("/lens")).rejects.toMatchObject({ message: body.detail, status: 503, body });
    expect(onError).toHaveBeenCalledWith(body.detail);
  });

  it("returns undefined for an empty success body (e.g. a 204 No Content)", async () => {
    const fetchImpl = vi.fn(async () => emptyResponse(204));
    const client = createApiClient({ getBaseUrl: () => "", fetchImpl });

    await expect(client.delete("/policies/abc", { accessToken: "sk" })).resolves.toBeUndefined();
  });

  it("omits the auth header when no token is provided", async () => {
    const fetchImpl = vi.fn(async () => okResponse({}));
    const client = createApiClient({ getBaseUrl: () => "", getAuthHeaderName: () => "Authorization", fetchImpl });

    await client.get("/public/info");

    const [, init] = fetchImpl.mock.calls[0];
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("getBlob returns the response body as a Blob on success", async () => {
    const blob = new Blob(["csv,data"], { type: "text/csv" });
    const fetchImpl = vi.fn(async () => ({ ok: true, status: 200, blob: async () => blob }) as unknown as Response);
    const client = createApiClient({ getBaseUrl: () => "https://proxy.example", fetchImpl });

    const result = await client.getBlob("/user/daily/activity/export", { accessToken: "sk" });

    expect(result).toBe(blob);
    const [, blobInit] = fetchImpl.mock.calls[0] as unknown as [unknown, RequestInit];
    expect(blobInit.method).toBe("GET");
  });

  it("getBlob throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi.fn(async () => errorResponse(500, { error: "export failed" }));
    const client = createApiClient({ getBaseUrl: () => "", fetchImpl });

    await expect(client.getBlob("/user/daily/activity/export", { accessToken: "sk" })).rejects.toBeInstanceOf(ApiError);
  });

  it("resolves the global fetch per call, so a swap after construction takes effect", async () => {
    const client = createApiClient({ getBaseUrl: () => "" });

    const original = globalThis.fetch;
    const swapped = vi.fn(async () => okResponse({ swapped: true }));
    globalThis.fetch = swapped as unknown as typeof fetch;
    try {
      const result = await client.get("/ping", { accessToken: "sk" });
      expect(result).toEqual({ swapped: true });
      expect(swapped).toHaveBeenCalledTimes(1);
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe("deriveErrorMessage", () => {
  it("extracts error.message from a ProxyException body, the shape the proxy emits for a pre-call hook HTTPException", () => {
    const actionable =
      "MCP semantic tool filtering could not run: embedding model 'text-embedding-3-small' exceeded its context window while embedding the user query. The request was blocked instead of silently passing all tools through. Switch to an embedding model with a larger context window, or disable semantic tool filtering.";
    const wireBody = {
      error: { message: actionable, type: "None", param: "None", code: "400" },
    };
    expect(deriveErrorMessage(wireBody)).toBe(actionable);
  });

  it("returns error directly when it is a plain string", () => {
    expect(deriveErrorMessage({ error: "flat error text" })).toBe("flat error text");
  });

  it("falls back to a string detail field", () => {
    expect(deriveErrorMessage({ detail: "detail text" })).toBe("detail text");
  });

  it("unwraps the HTTPException detail.error shape management endpoints raise", () => {
    expect(deriveErrorMessage({ detail: { error: "Team(s) not found: ghost-team" } })).toBe(
      "Team(s) not found: ghost-team",
    );
  });
});

describe("unwrapProxyErrorMessage", () => {
  it("should unwrap the proxy's stringified python dict message", () => {
    expect(
      unwrapProxyErrorMessage("{'error': 'Cost center CC-9999 is not recognized. Contact the FinOps team.'}"),
    ).toBe("Cost center CC-9999 is not recognized. Contact the FinOps team.");
  });

  it("should unwrap the full JSON error envelope down to the inner message", () => {
    const envelope = JSON.stringify({
      error: {
        message: "{'error': 'cost_center is required in team metadata. Contact the FinOps team.'}",
        type: "internal_server_error",
        param: "None",
        code: "400",
      },
    });
    expect(unwrapProxyErrorMessage(envelope)).toBe(
      "cost_center is required in team metadata. Contact the FinOps team.",
    );
  });

  it("should return plain messages and unparseable input unchanged", () => {
    expect(unwrapProxyErrorMessage("Failed to fetch")).toBe("Failed to fetch");
    expect(unwrapProxyErrorMessage("{}")).toBe("{}");
  });
});

describe("extractProxyErrorMessage", () => {
  it("should unwrap an Error's message without the error name prefix", () => {
    const error = new ApiError("{'error': 'Cost center CC-9999 is not recognized.'}", 400, {});
    expect(extractProxyErrorMessage(error)).toBe("Cost center CC-9999 is not recognized.");
  });

  it("should stringify non-Error inputs", () => {
    expect(extractProxyErrorMessage("plain failure")).toBe("plain failure");
  });
});
