import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchDecisionModels, makeSystemOneRequest } from "./system_one";
import type { SystemOneRequest, SystemOneResponse } from "../components/systemOneUI/lib/schemas";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://proxy.example.com"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

const payload: SystemOneRequest = {
  model: "jev-latest",
  state: "The app will not open",
  questions: {
    category: {
      type: "choice",
      instructions: "Which team should respond?",
      criteria: { technical: "Application issues" },
    },
  },
};

const responseBody: SystemOneResponse = {
  model: "jev-1.13.0",
  answers: { category: { type: "choice", choice: "technical", probabilities: { technical: 1 } } },
};

const malformedResponses: { name: string; body: unknown }[] = [
  { name: "model", body: { model: 1, answers: {} } },
  { name: "answers", body: { model: "jev-1.13.0", answers: [] } },
  { name: "answer type", body: { model: "jev-1.13.0", answers: { category: { type: "unknown" } } } },
  {
    name: "probability",
    body: {
      model: "jev-1.13.0",
      answers: { category: { type: "choice", choice: "technical", probabilities: { technical: 1.5 } } },
    },
  },
  {
    name: "usage",
    body: { model: "jev-1.13.0", answers: {}, usage: { input_tokens: "1", output_tokens: 1 } },
  },
];

describe("makeSystemOneRequest", () => {
  const mockFetch = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();

  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch);
    mockFetch.mockResolvedValue({
      ok: true,
      text: async () => JSON.stringify(responseBody),
    } as Response);
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("sends the typed request to the proxy endpoint with required headers and parses the response", async () => {
    const result = await makeSystemOneRequest(payload, "session-key", "https://custom.example.com/");
    const expectedRequest: Partial<RequestInit> = {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        Authorization: "Bearer session-key",
      },
      body: JSON.stringify(payload),
      signal: undefined,
    };

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      "https://custom.example.com/v1/systemone",
      expect.objectContaining(expectedRequest),
    );
    expect(result.response).toEqual(responseBody);
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
  });

  it("uses the proxy base URL when no custom base is provided", async () => {
    await makeSystemOneRequest(payload, "session-key");

    expect(mockFetch.mock.calls[0]?.[0]).toBe("https://proxy.example.com/v1/systemone");
  });

  it("throws the upstream error text for failed requests", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => "Invalid virtual key",
    } as Response);

    await expect(makeSystemOneRequest(payload, "session-key")).rejects.toThrow("Invalid virtual key");
  });

  it.each(malformedResponses)("returns an error for a malformed System One $name response", async ({ body }) => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => JSON.stringify(body) ?? "",
    } as Response);

    await expect(makeSystemOneRequest(payload, "session-key")).rejects.toThrow(
      "System One response has an invalid shape.",
    );
  });
});

describe("fetchDecisionModels", () => {
  const mockFetch = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();
  const modelGroupInfo = (data: unknown) => ({ ok: true, text: async () => JSON.stringify({ data }) }) as Response;

  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("lists the key's decision models from the proxy that answers the request, sorted by name", async () => {
    mockFetch.mockResolvedValueOnce(
      modelGroupInfo([
        { model_group: "pplx-decider", mode: "evaluation" },
        { model_group: "gpt-5.5", mode: "chat" },
        { model_group: "jev-latest", mode: "evaluation" },
        { model_group: "unknown-mode", mode: null },
      ]),
    );

    await expect(fetchDecisionModels("virtual-key", "https://custom.example.com/")).resolves.toEqual([
      "jev-latest",
      "pplx-decider",
    ]);
    expect(mockFetch).toHaveBeenCalledWith(
      "https://custom.example.com/model_group/info",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ Authorization: "Bearer virtual-key" }),
      }),
    );
  });

  it("rejects a response that is not a model group list", async () => {
    mockFetch.mockResolvedValueOnce(modelGroupInfo([{ mode: "evaluation" }]));

    await expect(fetchDecisionModels("virtual-key")).rejects.toThrow();
  });
});
