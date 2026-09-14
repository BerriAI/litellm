// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchClient } from "./api";
import { registerAuthTokenGetter, registerBaseUrlGetter, registerErrorHandler } from "./runtime";

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

describe("typed api client on a same-origin deployment", () => {
  beforeEach(() => {
    registerAuthTokenGetter(() => null);
    registerErrorHandler(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends requests to the page origin when no base url is registered", async () => {
    registerBaseUrlGetter(() => "");
    const { fetch, requests } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch });

    expect(requests[0].url).toBe(`${window.location.origin}/model_group/info`);
  });

  it("prefers a registered cross-origin base over the page origin", async () => {
    registerBaseUrlGetter(() => "https://proxy.example.com");
    const { fetch, requests } = capturingFetch(jsonResponse(200, { data: [] }));

    await fetchClient.GET("/model_group/info", { fetch });

    expect(requests[0].url).toBe("https://proxy.example.com/model_group/info");
    expect(new URL(requests[0].url).origin).not.toBe(window.location.origin);
  });
});
