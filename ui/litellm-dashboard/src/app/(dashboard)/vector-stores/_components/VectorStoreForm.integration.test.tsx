import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreCreateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import VectorStoreForm from "./VectorStoreForm";

vi.mock("@/components/networking", () => ({
  vectorStoreCreateCall: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([
    { model_group: "text-embedding-3-small", mode: "embedding" },
    { model_group: "gpt-4o", mode: "chat" },
  ]),
}));

const mockCreate = vi.mocked(vectorStoreCreateCall);
const mockToast = vi.mocked(toast);

const onSuccess = vi.fn();

const renderForm = () =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={vi.fn()}
      onSuccess={onSuccess}
      accessToken="test-token"
      credentials={[{ credential_name: "bedrock-prod", credential_info: {}, credential_values: {} }]}
    />,
  );

const setupUser = () => userEvent.setup({ pointerEventsCheck: 0 });

const chooseFromSelect = async (user: ReturnType<typeof userEvent.setup>, index: number, optionText: string) => {
  const trigger = screen.getAllByRole("combobox")[index];
  await user.click(trigger);
  if (trigger.getAttribute("aria-expanded") !== "true") {
    trigger.focus();
    await user.keyboard("{Enter}");
  }
  const options = await screen.findAllByText(optionText);
  await user.click(options[options.length - 1]);
};

const chooseProvider = (user: ReturnType<typeof userEvent.setup>, providerLabel: string) =>
  chooseFromSelect(user, 0, providerLabel);

const submit = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Create" }));

const createdPayload = () => mockCreate.mock.calls[0][1];

describe("VectorStoreForm submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(undefined);
  });

  it("sends every payload key for the default provider, leaving untouched optional fields undefined", async () => {
    const user = setupUser();
    renderForm();

    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-bedrock");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][0]).toBe("test-token");
    expect(createdPayload()).toStrictEqual({
      vector_store_id: "vs-bedrock",
      custom_llm_provider: "bedrock",
      vector_store_name: undefined,
      vector_store_description: undefined,
      vector_store_metadata: {},
      litellm_credential_name: undefined,
      litellm_params: {},
    });
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("sends the filled optional fields, parsed metadata and the selected credential", async () => {
    const user = setupUser();
    renderForm();

    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-full");
    const textboxes = screen.getAllByRole("textbox");
    await user.type(textboxes[1], "Support docs");
    await user.type(textboxes[2], "Docs for the support team");
    await user.clear(screen.getByPlaceholderText('{"key": "value"}'));
    await user.type(screen.getByPlaceholderText('{"key": "value"}'), '{{"tier": "gold"}');
    await chooseFromSelect(user, 1, "bedrock-prod");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload()).toStrictEqual({
      vector_store_id: "vs-full",
      custom_llm_provider: "bedrock",
      vector_store_name: "Support docs",
      vector_store_description: "Docs for the support team",
      vector_store_metadata: { tier: "gold" },
      litellm_credential_name: "bedrock-prod",
      litellm_params: {},
    });
  });

  it("renames the milvus embedding model to litellm_embedding_model inside litellm_params", async () => {
    const user = setupUser();
    renderForm();

    await chooseProvider(user, "Milvus");
    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-milvus");
    await user.type(screen.getByPlaceholderText("username:password or api key"), "user:pass");
    await user.type(screen.getByPlaceholderText("https://your-milvus-endpoint.com/"), "https://milvus.example.com");
    await chooseFromSelect(user, 1, "text-embedding-3-small");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload().litellm_params).toStrictEqual({
      api_key: "user:pass",
      api_base: "https://milvus.example.com",
      litellm_embedding_model: "text-embedding-3-small",
    });
    expect(createdPayload().custom_llm_provider).toBe("milvus");
  });

  it("sends a provider field's seeded default even when the user never touches it", async () => {
    const user = setupUser();
    renderForm();

    await chooseProvider(user, "Vertex AI Search");
    await user.type(
      screen.getByPlaceholderText('my-datastore_1234567890 (data store ID from Vertex AI / "Agent Search" console)'),
      "vs-vertex",
    );
    await user.type(screen.getByPlaceholderText("my-gcp-project-id"), "gcp-proj");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload().litellm_params).toStrictEqual({
      vertex_project: "gcp-proj",
      vertex_location: "global",
      vertex_collection_id: undefined,
      vertex_engine_id: undefined,
    });
  });

  it("keeps values typed under one provider when a later provider reuses the same field name", async () => {
    const user = setupUser();
    renderForm();

    await chooseProvider(user, "PostgreSQL pgvector (LiteLLM Connector)");
    await user.type(screen.getByPlaceholderText("http://your-deployed-server:8000"), "http://pg:8000");
    await user.type(screen.getByPlaceholderText("your-deployed-api-key"), "pg-key");
    await chooseProvider(user, "Azure OpenAI");
    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-azure");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload().litellm_params).toStrictEqual({
      api_key: "pg-key",
      api_base: "http://pg:8000",
    });
  });

  it("blocks the request and reports invalid metadata JSON instead of submitting", async () => {
    const user = setupUser();
    renderForm();

    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-bad-json");
    await user.clear(screen.getByPlaceholderText('{"key": "value"}'));
    await user.type(screen.getByPlaceholderText('{"key": "value"}'), "not json");
    await submit(user);

    await vi.waitFor(() => expect(mockToast.fromError).toHaveBeenCalledWith("Invalid JSON in metadata field"));
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("keeps the required-field messages that block an empty submit", async () => {
    const user = setupUser();
    renderForm();

    await submit(user);

    expect(await screen.findByText("Please input the vector store ID from your api provider")).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();
  });
});
