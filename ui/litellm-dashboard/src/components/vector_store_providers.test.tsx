import { describe, expect, it } from "vitest";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import {
  getProviderSpecificFields,
  getVectorStoreProviderLogoAndName,
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
} from "./vector_store_providers";

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
    expect(getVectorStoreProviderLogoAndName("valkey")).toEqual({
      logo: expect.stringContaining("valkey"),
      displayName: VectorStoreProviders.Valkey,
    });
  });

  it("registers mongodb in the provider, logo, and field maps", () => {
    expect(getVectorStoreProviderLogoAndName("mongodb")).toEqual({
      logo: expect.stringContaining("mongodb"),
      displayName: VectorStoreProviders.MongoDB,
    });
    expect(vectorStoreProviderMap.MongoDB).toBe("mongodb");
    expect(getProviderSpecificFields("mongodb").map((field) => field.name)).toEqual([
      "mongodb_connection_string",
      "mongodb_database",
      "mongodb_collection",
      "embedding_model",
      "mongodb_embedding_field",
      "mongodb_text_field",
      "mongodb_num_candidates",
    ]);
  });

  it("hides the mongodb connection string, which carries the database password", () => {
    const connectionString = getProviderSpecificFields("mongodb").find(
      (field) => field.name === "mongodb_connection_string",
    );

    expect(connectionString).toMatchObject({ type: "password", required: true });
  });

  it("picks the mongodb embedding model from the proxy's models rather than a fixed list", () => {
    const embeddingField = getProviderSpecificFields("mongodb").find((field) => field.name === "embedding_model");

    expect(embeddingField).toMatchObject({ type: "select", required: true });
    expect(embeddingField).not.toHaveProperty("options");
  });

  it("defaults the mongodb field names so a standard collection needs no extra input", () => {
    const fields = getProviderSpecificFields("mongodb");
    const byName = (name: string) => fields.find((field) => field.name === name);

    expect(byName("mongodb_embedding_field")).toMatchObject({ required: false, initialValue: "embedding" });
    expect(byName("mongodb_text_field")).toMatchObject({ required: false, initialValue: "text" });
    expect(byName("mongodb_num_candidates")).toMatchObject({ required: false });
  });

  it("registers valkey in the provider, logo, and field maps", () => {
    expect(vectorStoreProviderMap.Valkey).toBe("valkey");
    expect(vectorStoreProviderLogoMap[VectorStoreProviders.Valkey]).toContain("valkey");
    expect(getProviderSpecificFields("valkey").map((field) => field.name)).toEqual([
      "valkey_host",
      "valkey_port",
      "valkey_password",
      "valkey_ssl",
      "embedding_model",
      "valkey_text_field",
      "valkey_embedding_field",
    ]);
  });

  it("picks the valkey embedding model from the proxy's models like milvus does", () => {
    const embeddingField = getProviderSpecificFields("valkey").find((field) => field.name === "embedding_model");

    expect(embeddingField).toMatchObject({ type: "select", required: true });
    expect(embeddingField).not.toHaveProperty("options");
  });

  it("offers valkey_ssl as a false/true select defaulting to false", () => {
    const sslField = getProviderSpecificFields("valkey").find((field) => field.name === "valkey_ssl");

    expect(sslField).toMatchObject({
      type: "select",
      required: false,
      initialValue: "false",
    });
    expect(sslField?.options).toEqual([
      { value: "false", label: "false" },
      { value: "true", label: "true" },
    ]);
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
