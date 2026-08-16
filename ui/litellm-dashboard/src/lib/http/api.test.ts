import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchClient } from "./api";
import {
  registerAuthHeaderNameGetter,
  registerAuthTokenGetter,
  registerBaseUrlGetter,
  registerErrorHandler,
} from "./runtime";

const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const capturingFetch = (response: Response) => {
  const requests: Request[] = [];
  const fetch = vi.fn(async (request: Request) => {
    requests.push(request);
    return response;
  });
  return { fetch, requests };
};

describe("typed api client middleware", () => {
  beforeEach(() => {
    registerBaseUrlGetter(() => "");
    registerAuthHeaderNameGetter(() => "Authorization");
    registerErrorHandler(() => {});
    registerAuthTokenGetter(() => null);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("injects the bearer token under the registered auth header name", async () => {
    registerAuthTokenGetter(() => "sk-test");
    registerAuthHeaderNameGetter(() => "x-litellm-key");
    const { fetch, requests } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch });

    expect(requests[0].headers.get("x-litellm-key")).toBe("Bearer sk-test");
    expect(requests[0].headers.get("Authorization")).toBeNull();
  });

  it("omits the auth header when no token is set", async () => {
    const { fetch, requests } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch });

    expect(requests[0].headers.get("Authorization")).toBeNull();
  });

  it("rebases the request onto the registered base url, preserving path and query", async () => {
    registerBaseUrlGetter(() => "https://proxy.example.com/");
    const { fetch, requests } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch, params: { query: { model_group: "gpt-4o" } } });

    const url = new URL(requests[0].url);
    expect(url.origin).toBe("https://proxy.example.com");
    expect(url.pathname).toBe("/model_group/info");
    expect(url.searchParams.get("model_group")).toBe("gpt-4o");
  });

  it("maps a non-2xx response to an ApiError carrying status and the derived message", async () => {
    const { fetch } = capturingFetch(jsonResponse(403, { error: { message: "no access" } }));

    await expect(fetchClient.GET("/model_group/info", { fetch })).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
      message: "no access",
    });
  });

  it("returns the parsed body on a successful response", async () => {
    const body = { data: [{ model_group: "gpt-4o" }] };
    const { fetch } = capturingFetch(jsonResponse(200, body));

    const { data, error } = await fetchClient.GET("/model_group/info", { fetch });

    expect(error).toBeUndefined();
    expect(data).toEqual(body);
  });

  it("reports the derived message to the registered error handler on a non-2xx response", async () => {
    const onError = vi.fn();
    registerErrorHandler(onError);
    const { fetch } = capturingFetch(jsonResponse(401, { error: { message: "Authentication Error - Expired Key" } }));

    await expect(fetchClient.GET("/model_group/info", { fetch })).rejects.toBeInstanceOf(Error);

    expect(onError).toHaveBeenCalledWith("Authentication Error - Expired Key");
  });

  it("does not call the error handler on a successful response", async () => {
    const onError = vi.fn();
    registerErrorHandler(onError);
    const { fetch } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch });

    expect(onError).not.toHaveBeenCalled();
  });
});
