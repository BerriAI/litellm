import { describe, expect, it } from "vitest";

import {
  buildVectorStoreLitellmParams,
  parseLitellmParams,
  toVectorStoreEditValues,
  vectorStoreEditSchema,
} from "./vector_store_info_utils";

describe("vector store info utilities", () => {
  it("parses object-shaped LiteLLM parameters and rejects other JSON shapes", () => {
    expect(parseLitellmParams('{"api_base":"https://milvus.example"}')).toStrictEqual({
      api_base: "https://milvus.example",
    });
    expect(parseLitellmParams("[]")).toStrictEqual({});
    expect(parseLitellmParams("not-json")).toStrictEqual({});
  });

  it("normalizes the provider and never restores a redacted API key into the form", () => {
    expect(
      toVectorStoreEditValues({
        vector_store_id: "books",
        custom_llm_provider: "Milvus",
        litellm_credential_name: null,
        litellm_params: {
          api_base: "https://milvus.example",
          api_key: "********",
          litellm_embedding_model: "text-embedding-3-small",
        },
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      }),
    ).toMatchObject({
      custom_llm_provider: "milvus",
      litellm_credential_name: null,
      api_base: "https://milvus.example",
      api_key: "",
      embedding_model: "text-embedding-3-small",
    });
  });

  it("omits a blank API key while building Milvus parameters", () => {
    expect(
      buildVectorStoreLitellmParams({
        vector_store_id: "books",
        custom_llm_provider: "milvus",
        api_base: "https://milvus.example",
        api_key: "",
        embedding_model: "text-embedding-3-small",
      }),
    ).toStrictEqual({
      api_base: "https://milvus.example",
      litellm_embedding_model: "text-embedding-3-small",
    });
  });

  it("requires the Milvus endpoint and embedding model", () => {
    const result = vectorStoreEditSchema.safeParse({
      vector_store_id: "books",
      custom_llm_provider: "milvus",
      api_base: " ",
      embedding_model: "",
    });

    expect(result.success).toBe(false);
    if (result.success) return;
    expect(result.error.issues.map((issue) => issue.path)).toStrictEqual([["api_base"], ["embedding_model"]]);
  });
});
