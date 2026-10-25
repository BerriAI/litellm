import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreCreateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import VectorStoreForm from "./VectorStoreForm";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/http/client";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";

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
      credentials={[
        {
          credential_name: "bedrock-prod",
          display_name: "Bedrock Prod",
          credential_info: {},
          credential_values: {},
        },
      ]}
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

  it("切换语言保留向量存储草稿，并翻译内置提供商的必填校验", async () => {
    localStorage.clear();
    const user = setupUser();
    render(
      <LanguageProvider defaultLanguage="en" userId="vector-validation" storageScope="test">
        <LanguageSelector />
        <VectorStoreForm isVisible onCancel={vi.fn()} onSuccess={onSuccess} accessToken="test-token" credentials={[]} />
      </LanguageProvider>,
    );
    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "Home");
    await chooseProvider(user, "OpenAI");

    // 向量存储 ID 和提供商是请求数据，语言切换不得改写。
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByPlaceholderText("输入提供商的向量存储 ID")).toHaveValue("Home");
    expect(screen.getAllByText("OpenAI").length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: translate("zh-CN", "Create") }));
    expect(await screen.findByText("请输入API 密钥")).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();

    // 密码显隐按钮的文案随语言切换，已经填写的凭据不发生变化。
    fireEvent.change(screen.getByPlaceholderText("sk-..."), { target: { value: "test-credential" } });
    await user.click(screen.getByRole("button", { name: "显示密码" }));
    expect(screen.getByPlaceholderText("sk-...")).toHaveAttribute("type", "text");
    expect(screen.getByRole("button", { name: "隐藏密码" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(screen.getByPlaceholderText("sk-...")).toHaveValue("test-credential");
    expect(screen.getByRole("button", { name: "Hide Password" })).toBeInTheDocument();
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

  it("shows the credential display name beside the name, searches by it, and still submits the name", async () => {
    const user = setupUser();
    renderForm();

    const picker = screen.getByPlaceholderText("Select or search for existing credentials");
    await user.click(picker);

    const option = await screen.findByRole("option", { name: /bedrock-prod/ });
    expect(option).toHaveTextContent("bedrock-prod");
    expect(option).toHaveTextContent("Bedrock Prod");

    await user.type(picker, "bedrock prod");
    await user.click(await screen.findByRole("option", { name: /bedrock-prod/ }));
    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-labeled");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload().litellm_credential_name).toBe("bedrock-prod");
  });

  it("sends no credential when None is picked after a credential", async () => {
    const user = setupUser();
    renderForm();

    const picker = screen.getByPlaceholderText("Select or search for existing credentials");
    await user.click(picker);
    await user.click(await screen.findByRole("option", { name: /bedrock-prod/ }));
    expect(picker).toHaveValue("Bedrock Prod");
    await user.click(picker);
    await user.click(await screen.findByRole("option", { name: "None" }));
    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-none");
    await submit(user);

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(createdPayload().litellm_credential_name).toBeUndefined();
  });

  it("blocks the request and reports invalid metadata JSON instead of submitting", async () => {
    const user = setupUser();
    renderForm();

    await user.type(screen.getByPlaceholderText("Enter vector store ID from your provider"), "vs-bad-json");
    await user.clear(screen.getByPlaceholderText('{"key": "value"}'));
    await user.type(screen.getByPlaceholderText('{"key": "value"}'), "not json");
    await submit(user);

    await vi.waitFor(() =>
      expect(mockToast.fromError).toHaveBeenCalledWith(
        "Invalid JSON in metadata field",
        expect.objectContaining({ description: expect.anything() }),
      ),
    );
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

describe("向量存储创建的语言边界", () => {
  beforeEach(async () => {
    localStorage.clear();
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(undefined);
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    mockToast.success.mockImplementation(realToast.success);
    mockToast.fromError.mockImplementation(realToast.fromError);
  });

  afterEach(async () => {
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    act(() => realToast.dismiss());
    mockToast.success.mockReset();
    mockToast.fromError.mockReset();
  });

  const renderLocalized = () => {
    const tree = (defaultLanguage: "en" | "zh-CN") => (
      <LanguageProvider defaultLanguage={defaultLanguage} userId="vector-create-review" storageScope="/vector-stores">
        <VectorStoreForm isVisible onCancel={vi.fn()} onSuccess={onSuccess} accessToken="test-token" credentials={[]} />
        <Toaster />
      </LanguageProvider>
    );
    const view = render(tree("en"));
    return { changeLanguage: (language: "en" | "zh-CN") => view.rerender(tree(language)) };
  };

  it("已有 ID 和密码校验提示随语言切换，不提交或覆盖草稿", async () => {
    const user = setupUser();
    const view = renderLocalized();
    await chooseProvider(user, "OpenAI");
    fireEvent.change(screen.getByLabelText("Vector Store Name"), { target: { value: "Home" } });
    fireEvent.change(screen.getByPlaceholderText('{"key": "value"}'), { target: { value: '{"Home":"Home"}' } });
    await submit(user);
    expect(await screen.findByText("Please input the vector store ID from your api provider")).toBeInTheDocument();
    expect(screen.getByText("Please input the api key")).toBeInTheDocument();

    view.changeLanguage("zh-CN");
    expect(
      screen.getByText(translate("zh-CN", "Please input the vector store ID from your api provider")),
    ).toBeInTheDocument();
    expect(screen.getByText("请输入API 密钥")).toBeInTheDocument();
    expect(screen.getByLabelText(translate("zh-CN", "Vector Store Name"))).toHaveValue("Home");
    expect(screen.getByPlaceholderText('{"key": "value"}')).toHaveValue('{"Home":"Home"}');
    view.changeLanguage("en");
    expect(screen.getByText("Please input the api key")).toBeInTheDocument();
    expect(screen.getByLabelText("Vector Store Name")).toHaveValue("Home");
    expect(mockCreate).not.toHaveBeenCalled();
    expect(fetchAvailableModels).toHaveBeenCalledTimes(1);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("已有模型选择校验提示随语言切换，保留提供商凭据", async () => {
    const user = setupUser();
    const view = renderLocalized();
    await chooseProvider(user, "Milvus");
    fireEvent.change(screen.getByLabelText("Vector Store ID"), { target: { value: "Home" } });
    fireEvent.change(screen.getByPlaceholderText("username:password or api key"), {
      target: { value: "synthetic-key" },
    });
    fireEvent.change(screen.getByPlaceholderText("https://your-milvus-endpoint.com/"), {
      target: { value: "https://example.invalid" },
    });
    await submit(user);
    expect(await screen.findByText("Please select the embedding model")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(
      screen.getByText(
        translate("zh-CN", "Please select the {field}", { field: translate("zh-CN", "Embedding Model") }),
      ),
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText("username:password or api key")).toHaveValue("synthetic-key");
    expect(screen.getByLabelText(translate("zh-CN", "Vector Store ID"))).toHaveValue("Home");
    expect(mockCreate).not.toHaveBeenCalled();
    expect(fetchAvailableModels).toHaveBeenCalledTimes(1);
  });

  it("成功通知在请求等待中和显示后随语言切换，提交数据不变", async () => {
    const user = setupUser();
    const pending = Promise.withResolvers<undefined>();
    mockCreate.mockReturnValue(pending.promise);
    const view = renderLocalized();
    fireEvent.change(screen.getByLabelText("Vector Store ID"), { target: { value: "Home" } });
    await submit(user);
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(undefined));
    expect(await screen.findByText(translate("zh-CN", "Vector store created successfully"))).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Vector store created successfully")).toBeInTheDocument();
    const expectedPayload = {
      vector_store_id: "Home",
      custom_llm_provider: "bedrock",
      vector_store_name: undefined,
      vector_store_description: undefined,
      vector_store_metadata: {},
      litellm_credential_name: undefined,
      litellm_params: {},
    };
    expect(createdPayload()).toStrictEqual(expectedPayload);
    expect(mockCreate).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledTimes(1);
    expect(fetchAvailableModels).toHaveBeenCalledTimes(1);
  });

  it("本地 JSON 错误通知随语言切换，不改变无效输入或发送请求", async () => {
    const user = setupUser();
    const view = renderLocalized();
    fireEvent.change(screen.getByLabelText("Vector Store ID"), { target: { value: "Home" } });
    fireEvent.change(screen.getByPlaceholderText('{"key": "value"}'), { target: { value: "not json" } });
    await submit(user);
    expect(await screen.findByText("Invalid JSON in metadata field")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", "Invalid JSON in metadata field"))).toBeInTheDocument();
    expect(screen.getByPlaceholderText('{"key": "value"}')).toHaveValue("not json");
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("服务端限流保持警告分类和原文，只有通知上下文切换语言", async () => {
    const user = setupUser();
    const pending = Promise.withResolvers<undefined>();
    mockCreate.mockReturnValue(pending.promise);
    const view = renderLocalized();
    fireEvent.change(screen.getByLabelText("Vector Store ID"), { target: { value: "Home" } });
    await submit(user);
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    await act(async () =>
      pending.reject(new ApiError("Home", 429, { error: { message: "Home" }, debug: "private-detail" })),
    );
    expect(await screen.findByText(translate("zh-CN", "Rate Limit Exceeded"))).toBeInTheDocument();
    expect(
      screen.getByText(translate("zh-CN", "Error creating vector store: {error}", { error: "Home" })),
    ).toBeInTheDocument();
    expect(screen.queryByText(/private-detail/)).not.toBeInTheDocument();
    expect(screen.getByLabelText(translate("zh-CN", "Vector Store ID"))).toHaveValue("Home");
    view.changeLanguage("en");
    expect(screen.getByText("Rate Limit Exceeded")).toBeInTheDocument();
    expect(screen.getByText("Error creating vector store: Home")).toBeInTheDocument();
    expect(mockCreate).toHaveBeenCalledTimes(1);
    expect(onSuccess).not.toHaveBeenCalled();
  });
});
