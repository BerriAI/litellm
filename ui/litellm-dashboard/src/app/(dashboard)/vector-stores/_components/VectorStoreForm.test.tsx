import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CredentialItem, vectorStoreCreateCall } from "@/components/networking";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import { VectorStoreProviders } from "@/components/vector_store_providers";
import VectorStoreForm, { buildVectorStoreLitellmParams } from "./VectorStoreForm";

vi.mock("@/components/networking");

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const renderForm = (onCancel: () => void = vi.fn()) =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={onCancel}
      onSuccess={vi.fn()}
      accessToken="test-token"
      credentials={[] as CredentialItem[]}
    />,
  );

describe("VectorStoreForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the form when visible", () => {
    renderForm();

    expect(screen.getByText("Add New Vector Store")).toBeInTheDocument();
  });

  it("renders the default provider's bundled logo via the shared Logo component", () => {
    renderForm();

    const logo = screen.getByRole("img", { name: `${VectorStoreProviders.Bedrock} logo` });
    expect(logo).toHaveAttribute("src", providerLogoMap[Providers.Bedrock]);
  });

  it("creates the vector store when Create is clicked on a filled form", async () => {
    const user = userEvent.setup();
    renderForm();

    fireEvent.change(screen.getByLabelText(/Vector Store ID/), { target: { value: "vs-created" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(vectorStoreCreateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(vectorStoreCreateCall).mock.calls[0][1]).toMatchObject({ vector_store_id: "vs-created" });
  });

  it("cancels without creating the vector store when Cancel is clicked on a filled form", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderForm(onCancel);

    fireEvent.change(screen.getByLabelText(/Vector Store ID/), { target: { value: "vs-abandoned" } });
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(vectorStoreCreateCall).not.toHaveBeenCalled();
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
