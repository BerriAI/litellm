import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CredentialItem } from "@/components/networking";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import { VectorStoreProviders } from "@/components/vector_store_providers";
import VectorStoreForm, { buildVectorStoreLitellmParams } from "./VectorStoreForm";

vi.mock("@/components/networking");

const renderForm = () =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      accessToken="test-token"
      credentials={[] as CredentialItem[]}
    />,
  );

describe("VectorStoreForm", () => {
  it("should render the form when visible", () => {
    renderForm();

    expect(screen.getByText("Add New Vector Store")).toBeInTheDocument();
  });

  it("renders the default provider's bundled logo via the shared Logo component", () => {
    renderForm();

    const logo = screen.getByRole("img", { name: `${VectorStoreProviders.Bedrock} logo` });
    expect(logo.getAttribute("src")).toBe(providerLogoMap[Providers.Bedrock]);
  });
});

describe("buildVectorStoreLitellmParams", () => {
  it("renames embedding_model to litellm_embedding_model for valkey", () => {
    const valkeyFormValues = {
      valkey_host: "my-valkey.example.com",
      valkey_port: "6379",
      valkey_password: "secret",
      valkey_ssl: "true",
      embedding_model: "text-embedding-3-small",
      valkey_text_field: "text",
      valkey_embedding_field: "vector",
    };

    const params = buildVectorStoreLitellmParams("valkey", valkeyFormValues);

    const expectedParams = {
      valkey_host: "my-valkey.example.com",
      valkey_port: "6379",
      valkey_password: "secret",
      valkey_ssl: "true",
      litellm_embedding_model: "text-embedding-3-small",
      valkey_text_field: "text",
      valkey_embedding_field: "vector",
    };
    expect(params).toEqual(expectedParams);
    expect(params).not.toHaveProperty("embedding_model");
  });

  it("renames embedding_model to litellm_embedding_model for milvus", () => {
    const params = buildVectorStoreLitellmParams("milvus", {
      api_key: "user:pass",
      api_base: "https://my-milvus-endpoint.com/",
      embedding_model: "text-embedding-3-small",
    });

    expect(params).toEqual({
      api_key: "user:pass",
      api_base: "https://my-milvus-endpoint.com/",
      litellm_embedding_model: "text-embedding-3-small",
    });
  });

  it("keeps embedding_model as-is for providers outside the rename set", () => {
    const params = buildVectorStoreLitellmParams("s3_vectors", {
      vector_bucket_name: "my-vector-bucket",
      aws_region_name: "us-west-2",
      embedding_model: "text-embedding-3-small",
    });

    expect(params.embedding_model).toBe("text-embedding-3-small");
    expect(params).not.toHaveProperty("litellm_embedding_model");
  });
});
