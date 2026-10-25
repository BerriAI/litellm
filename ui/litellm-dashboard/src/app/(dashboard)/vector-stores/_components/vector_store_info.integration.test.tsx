import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { credentialListCall, vectorStoreInfoCall, vectorStoreUpdateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import VectorStoreInfoView from "./vector_store_info";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/http/client";

vi.mock("@/components/networking", () => ({
  vectorStoreInfoCall: vi.fn(),
  vectorStoreUpdateCall: vi.fn(),
  credentialListCall: vi.fn(),
}));

vi.mock("./VectorStoreTester", () => ({ __esModule: true, default: () => null }));

const mockInfo = vi.mocked(vectorStoreInfoCall);
const mockUpdate = vi.mocked(vectorStoreUpdateCall);
const mockCredentials = vi.mocked(credentialListCall);
const mockToast = vi.mocked(toast);

const serverRecord = {
  vector_store_id: "vs-1",
  vector_store_name: "support-docs-store",
  vector_store_description: "Docs for support",
  custom_llm_provider: "bedrock",
  vector_store_metadata: { tier: "gold" },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-02-02T00:00:00Z",
  litellm_credential_name: "bedrock-prod",
};

const renderView = (editVectorStore: boolean) =>
  render(
    <VectorStoreInfoView
      vectorStoreId="vs-1"
      onClose={vi.fn()}
      accessToken="sk-test"
      is_admin={true}
      editVectorStore={editVectorStore}
    />,
  );

const savedPayload = () => mockUpdate.mock.calls[0][1];

describe("向量存储编辑的语言边界", () => {
  beforeEach(async () => {
    localStorage.clear();
    vi.clearAllMocks();
    mockInfo.mockResolvedValue({ vector_store: serverRecord });
    mockCredentials.mockResolvedValue({ credentials: [] });
    mockUpdate.mockResolvedValue({});
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
      <LanguageProvider defaultLanguage={defaultLanguage} userId="vector-edit-review" storageScope="/vector-stores">
        <VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin editVectorStore />
        <Toaster />
      </LanguageProvider>
    );
    const view = render(tree("en"));
    return { changeLanguage: (language: "en" | "zh-CN") => view.rerender(tree(language)) };
  };

  it("已有必填错误随语言切换，不重新加载或覆盖未保存名称", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({ vector_store: { ...serverRecord, vector_store_id: "" } });
    const view = renderLocalized();
    fireEvent.change(await screen.findByLabelText("Vector Store Name"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Please input a vector store ID")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText("请输入向量存储 ID")).toBeInTheDocument();
    expect(screen.getByLabelText(translate("zh-CN", "Vector Store Name"))).toHaveValue("Home");
    view.changeLanguage("en");
    expect(screen.getByText("Please input a vector store ID")).toBeInTheDocument();
    expect(mockInfo).toHaveBeenCalledTimes(1);
    expect(mockCredentials).toHaveBeenCalledTimes(1);
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("保存成功通知在请求等待中和显示后切换语言，载荷保持五个原始字段", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<object>();
    mockUpdate.mockReturnValue(pending.promise);
    const view = renderLocalized();
    fireEvent.change(await screen.findByLabelText("Vector Store Name"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    expect(mockInfo).toHaveBeenCalledTimes(1);
    expect(mockCredentials).toHaveBeenCalledTimes(1);
    await act(async () => pending.resolve({}));
    expect(await screen.findByText(translate("zh-CN", "Vector store updated successfully"))).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Vector store updated successfully")).toBeInTheDocument();
    const expectedPayload = {
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "Home",
      vector_store_description: "Docs for support",
      vector_store_metadata: { tier: "gold" },
    };
    expect(savedPayload()).toStrictEqual(expectedPayload);
    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(mockInfo).toHaveBeenCalledTimes(2);
    expect(mockCredentials).toHaveBeenCalledTimes(1);
  });

  it("本地 JSON 通知已显示后切换语言，保留元数据和编辑状态", async () => {
    const user = userEvent.setup();
    const view = renderLocalized();
    fireEvent.change(await screen.findByPlaceholderText('{"key": "value"}'), { target: { value: "not json" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Invalid JSON in metadata field")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", "Invalid JSON in metadata field"))).toBeInTheDocument();
    expect(screen.getByPlaceholderText('{"key": "value"}')).toHaveValue("not json");
    expect(mockInfo).toHaveBeenCalledTimes(1);
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it.each(["load", "save"] as const)("%s 的服务端拒绝保留权限分类与原文，通知上下文随语言切换", async (operation) => {
    const pending = Promise.withResolvers<{ vector_store: typeof serverRecord }>();
    if (operation === "load") mockInfo.mockReturnValue(pending.promise);
    else mockUpdate.mockReturnValue(pending.promise);
    const view = renderLocalized();
    if (operation === "save") {
      fireEvent.change(await screen.findByLabelText("Vector Store Name"), { target: { value: "Home" } });
      await userEvent.setup().click(screen.getByRole("button", { name: "Save Changes" }));
      await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    }
    view.changeLanguage("zh-CN");
    await act(async () =>
      pending.reject(new ApiError("Home", 403, { error: { message: "Home" }, debug: "private-detail" })),
    );
    expect(await screen.findByText(translate("zh-CN", "Access Denied"))).toBeInTheDocument();
    const message =
      operation === "load" ? "Error fetching vector store details: {error}" : "Error updating vector store: {error}";
    expect(screen.getByText(translate("zh-CN", message, { error: "Home" }))).toBeInTheDocument();
    expect(screen.queryByText(/private-detail/)).not.toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Access Denied")).toBeInTheDocument();
    expect(screen.getByText(translate("en", message, { error: "Home" }))).toBeInTheDocument();
    expect(mockInfo).toHaveBeenCalledTimes(1);
    expect(mockCredentials).toHaveBeenCalledTimes(1);
    expect(mockUpdate).toHaveBeenCalledTimes(operation === "save" ? 1 : 0);
  });
});

describe("VectorStoreInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockInfo.mockResolvedValue({ vector_store: serverRecord });
    mockCredentials.mockResolvedValue({ credentials: [{ credential_name: "bedrock-prod" }] });
    mockUpdate.mockResolvedValue({});
  });

  it("切换语言保留向量存储名称修改，并使用中文校验空 ID", async () => {
    localStorage.clear();
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({ vector_store: { ...serverRecord, vector_store_id: "" } });
    render(
      <LanguageProvider defaultLanguage="en" userId="vector-edit-validation" storageScope="test">
        <LanguageSelector />
        <VectorStoreInfoView vectorStoreId="vs-1" onClose={vi.fn()} accessToken="sk-test" is_admin editVectorStore />
      </LanguageProvider>,
    );
    const nameInput = await screen.findByDisplayValue("support-docs-store");
    await user.clear(nameInput);
    await user.type(nameInput, "Home");

    // 切换文案后不得重新读取服务器值覆盖未保存草稿。
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByDisplayValue("Home")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: translate("zh-CN", "Save Changes") }));
    expect(await screen.findByText("请输入向量存储 ID")).toBeInTheDocument();
    expect(mockInfo).toHaveBeenCalledTimes(1);
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("still saves when the server left the nullable name and description null", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({
      vector_store: { ...serverRecord, vector_store_name: null, vector_store_description: null },
    });
    renderView(true);
    await screen.findByRole("button", { name: "Save Changes" });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: null,
      vector_store_description: null,
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends only the five editable keys and drops every server-only field", async () => {
    const user = userEvent.setup();
    renderView(true);

    const nameInput = await screen.findByDisplayValue("support-docs-store");
    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "renamed-store" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][0]).toBe("sk-test");
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "renamed-store",
      vector_store_description: "Docs for support",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends the same five keys when editing is entered from the details view", async () => {
    const user = userEvent.setup();
    renderView(false);

    const editButtons = await screen.findAllByRole("button", { name: "Edit Vector Store" });
    await user.click(editButtons[0]);
    const descriptionInput = await screen.findByDisplayValue("Docs for support");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "new description" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "support-docs-store",
      vector_store_description: "new description",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("keeps the credential field out of the payload even after it is picked", async () => {
    const user = userEvent.setup();
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getAllByRole("combobox")[1]);
    const options = await screen.findAllByText("bedrock-prod");
    await user.click(options[options.length - 1]);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(Object.keys(savedPayload())).toStrictEqual([
      "vector_store_id",
      "custom_llm_provider",
      "vector_store_name",
      "vector_store_description",
      "vector_store_metadata",
    ]);
  });

  it("blocks the request and reports invalid metadata JSON instead of saving", async () => {
    const user = userEvent.setup();
    renderView(true);

    const metadataInput = await screen.findByPlaceholderText('{"key": "value"}');
    await user.clear(metadataInput);
    fireEvent.change(metadataInput, { target: { value: "not json" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() =>
      expect(mockToast.fromError).toHaveBeenCalledWith(
        "Invalid JSON in metadata field",
        expect.objectContaining({ description: expect.anything() }),
      ),
    );
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("keeps the required-field message that blocks saving without a vector store id", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({ vector_store: { ...serverRecord, vector_store_id: "" } });
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please input a vector store ID")).toBeInTheDocument();
    expect(mockUpdate).not.toHaveBeenCalled();
  });
});
