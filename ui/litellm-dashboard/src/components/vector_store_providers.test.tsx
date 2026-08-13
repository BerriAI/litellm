import { describe, expect, it } from "vitest";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import { getVectorStoreProviderLogoAndName, VectorStoreProviders } from "./vector_store_providers";

describe("getVectorStoreProviderLogoAndName", () => {
  it("resolves vector-store-only slugs to their own logo and display name", () => {
    expect(getVectorStoreProviderLogoAndName("pg_vector")).toEqual({
      logo: expect.stringContaining("postgresql"),
      displayName: VectorStoreProviders.PgVector,
    });
    expect(getVectorStoreProviderLogoAndName("milvus")).toEqual({
      logo: expect.stringContaining("milvus"),
      displayName: VectorStoreProviders.Milvus,
    });
    expect(getVectorStoreProviderLogoAndName("s3_vectors")).toEqual({
      logo: expect.stringContaining("s3_vector"),
      displayName: VectorStoreProviders.S3Vectors,
    });
  });

  it("resolves shared slugs to the same bundled logo as the provider map", () => {
    expect(getVectorStoreProviderLogoAndName("bedrock")).toEqual({
      logo: providerLogoMap[Providers.Bedrock],
      displayName: VectorStoreProviders.Bedrock,
    });
    expect(getVectorStoreProviderLogoAndName("vertex_ai/search_api")).toEqual({
      logo: providerLogoMap[Providers.Vertex_AI],
      displayName: VectorStoreProviders.VertexAiSearch,
    });
  });

  it("falls back to the LLM provider resolver for slugs outside the vector-store map", () => {
    expect(getVectorStoreProviderLogoAndName("anthropic")).toEqual({
      logo: providerLogoMap[Providers.Anthropic],
      displayName: Providers.Anthropic,
    });
    expect(getVectorStoreProviderLogoAndName("totally_unknown")).toEqual({
      logo: "",
      displayName: "totally_unknown",
    });
  });
});
