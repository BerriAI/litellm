import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIEmbeddingsRequest } from "./embeddings_api";
import OpenAI from "openai";

vi.mock("openai");

describe("embeddings_api", () => {
  const mockCreate = vi.fn();
  const mockUpdateEmbeddingsUI = vi.fn();

  beforeEach(() => {
    // Mock the response structure from OpenAI embeddings API
    mockCreate.mockResolvedValue({
      data: [
        {
          embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
          index: 0,
          object: "embedding",
        },
      ],
      model: "text-embedding-3-small",
      object: "list",
      usage: {
        prompt_tokens: 5,
        total_tokens: 5,
      },
    });

    // Mock the OpenAI constructor and its methods
    (OpenAI as any).mockImplementation(() => ({
      embeddings: {
        create: mockCreate,
      },
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should make a request to the embeddings API", async () => {
    await makeOpenAIEmbeddingsRequest(
      "Hello, world!",
      mockUpdateEmbeddingsUI,
      "text-embedding-3-small",
      "1234567890",
      [],
    );

    expect(mockCreate).toHaveBeenCalledWith({
      model: "text-embedding-3-small",
      input: "Hello, world!",
    });
    expect(mockUpdateEmbeddingsUI).toHaveBeenCalledWith(
      JSON.stringify([0.1, 0.2, 0.3, 0.4, 0.5]),
      "text-embedding-3-small",
    );
  });
});
