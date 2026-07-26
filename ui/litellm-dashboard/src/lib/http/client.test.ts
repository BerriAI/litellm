import { describe, it, expect, vi } from "vitest";
import { createApiClient, ApiError, deriveErrorMessage } from "./client";

const okResponse = (data: unknown): Response =>
  ({ ok: true, status: 200, text: async () => JSON.stringify(data) }) as unknown as Response;

const emptyResponse = (status: number): Response => ({ ok: true, status, text: async () => "" }) as unknown as Response;

const errorResponse = (status: number, body: unknown): Response =>
  ({ ok: false, status, text: async () => JSON.stringify(body) }) as unknown as Response;

const rawErrorResponse = (status: number, text: string): Response =>
  ({ ok: false, status, text: async () => text }) as unknown as Response;

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

  it("falls back to the raw text body when a non-2xx response is not JSON (e.g. an HTML 502)", async () => {
    const fetchImpl = vi.fn(async () => rawErrorResponse(502, "<html>Bad Gateway</html>"));
    const onError = vi.fn();
    const client = createApiClient({ getBaseUrl: () => "", onError, fetchImpl });

    const promise = client.get("/keys", { accessToken: "sk" });

    await expect(promise).rejects.toMatchObject({ message: "<html>Bad Gateway</html>", status: 502 });
    expect(onError).toHaveBeenCalledWith("<html>Bad Gateway</html>");
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
});
