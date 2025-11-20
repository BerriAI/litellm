import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIEmbeddingsRequest } from "./embeddings_api";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

describe("embeddings_api", () => {
  const mockUpdateEmbeddingsUI = vi.fn();
  const mockFetch = vi.fn();

  beforeEach(() => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        data: [
          {
            embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
            index: 0,
            object: "embedding",
          },
        ],
        model: "text-embedding-3-small",
        object: "list",
      }),
      text: async () => "",
    } as Response);

    // @ts-ignore - assigning to global for test environment
    global.fetch = mockFetch;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should make a request to the embeddings endpoint", async () => {
    await makeOpenAIEmbeddingsRequest(
      "Hello, world!",
      mockUpdateEmbeddingsUI,
      "text-embedding-3-small",
      "1234567890",
      [],
    );

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith("https://example.com/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer 1234567890",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: "Hello, world!",
      }),
    });
    expect(mockUpdateEmbeddingsUI).toHaveBeenCalledWith(
      JSON.stringify([0.1, 0.2, 0.3, 0.4, 0.5]),
      "text-embedding-3-small",
    );
  });

  it("should not include encoding_format when making the request", async () => {
    await makeOpenAIEmbeddingsRequest("Sample text", mockUpdateEmbeddingsUI, "text-embedding-3-small", "abcdef", []);

    const fetchCall = mockFetch.mock.calls[0];
    const options = fetchCall[1] as RequestInit;
    const body = options.body as string;
    const parsedBody = JSON.parse(body);

    expect(parsedBody).not.toHaveProperty("encoding_format");
    expect(parsedBody).toEqual({
      model: "text-embedding-3-small",
      input: "Sample text",
    });
  });
});
