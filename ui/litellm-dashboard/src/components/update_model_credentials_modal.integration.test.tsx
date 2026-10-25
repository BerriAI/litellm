import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import UpdateModelCredentialsModal from "./update_model_credentials_modal";
import * as networking from "./networking";
import { toast } from "@/lib/toast";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/lib/toast");

vi.mock("./networking", async () => {
  const actual = await vi.importActual("./networking");
  return {
    ...actual,
    modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  };
});

const mockModelPatchUpdateCall = vi.mocked(networking.modelPatchUpdateCall);

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

const renderModal = (overrides: Partial<Parameters<typeof UpdateModelCredentialsModal>[0]> = {}) =>
  render(
    <LanguageProvider defaultLanguage="en" userId="credential-rotation" storageScope="test">
      <LanguageSelector />
      <UpdateModelCredentialsModal
        open
        onCancel={vi.fn()}
        accessToken="test-token"
        modelId="model-123"
        onUpdated={vi.fn()}
        {...overrides}
      />
      <Toaster expand />
    </LanguageProvider>,
  );

const changeLanguage = (value: "en" | "zh-CN") =>
  fireEvent.change(screen.getByRole("combobox", { name: /Interface language|界面语言/, hidden: true }), {
    target: { value },
  });

describe("UpdateModelCredentialsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockModelPatchUpdateCall.mockResolvedValue({});
  });

  afterEach(() => act(() => toast.dismiss()));

  // 已有校验错误随语言更新，不能靠重新提交或重新校验刷新文案。
  it.each(["en", "zh-CN"] as const)("%s 的已显示校验随语言切换且保留输入", async (initial) => {
    renderModal();
    if (initial === "zh-CN") changeLanguage("zh-CN");
    fireEvent.click(screen.getByRole("button", { name: initial === "en" ? "Update API Key" : "更新 API 密钥" }));
    const original = initial === "en" ? "Enter a new API key" : "请输入新的 API 密钥。";
    const next = initial === "en" ? "请输入新的 API 密钥。" : "Enter a new API key";
    expect(await screen.findByText(original)).toBeInTheDocument();
    changeLanguage(initial === "en" ? "zh-CN" : "en");
    expect(screen.getByText(next)).toBeInTheDocument();
    expect(screen.queryByText(original)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^(New API Key|新 API 密钥)$/)).toHaveValue("");
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });

  // 密钥和模型 ID 是原始请求数据，即使与词典键同名也不能翻译。
  it.each(["success", "failure"] as const)("待处理的 %s 提示采用当前语言并可再次切换", async (result) => {
    const request = Promise.withResolvers<object>();
    mockModelPatchUpdateCall.mockReturnValue(request.promise);
    const onUpdated = vi.fn();
    const onCancel = vi.fn();
    renderModal({ onUpdated, onCancel });
    fireEvent.change(screen.getByLabelText("New API Key"), { target: { value: "  Home  " } });
    fireEvent.click(screen.getByRole("button", { name: "Update API Key" }));
    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    changeLanguage("zh-CN");
    expect(screen.getByLabelText("新 API 密钥")).toHaveValue("  Home  ");
    await act(async () => {
      if (result === "success") request.resolve({});
      else request.reject(new Error("test-only-secret"));
    });
    expect(await screen.findByText(result === "success" ? "API 密钥已更新" : "API 密钥更新失败。")).toBeInTheDocument();
    changeLanguage("en");
    expect(screen.getByText(result === "success" ? "API key updated" : "Failed to update API key")).toBeInTheDocument();
    expect(screen.queryByText("test-only-secret")).not.toBeInTheDocument();
    expect(onUpdated).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
    expect(onCancel).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
    if (result === "failure") expect(screen.getByLabelText("New API Key")).toHaveValue("  Home  ");
    expect(mockModelPatchUpdateCall).toHaveBeenCalledExactlyOnceWith(
      "test-token",
      { litellm_params: { api_key: "Home" }, model_info: { id: "model-123" } },
      "model-123",
    );
  });

  it("sends a minimal PATCH with only the new api_key", async () => {
    const user = userEvent.setup();
    const onUpdated = vi.fn();
    const onCancel = vi.fn();
    renderModal({ onUpdated, onCancel });

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "sk-rotated-9988" } });
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    const [token, payload, modelId] = mockModelPatchUpdateCall.mock.calls[0];
    expect(token).toBe("test-token");
    expect(modelId).toBe("model-123");
    // Exactly the new key plus the id — nothing else from the deployment.
    expect(payload).toEqual({ litellm_params: { api_key: "sk-rotated-9988" }, model_info: { id: "model-123" } });
    expect(onUpdated).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("does not call the update API when the field is left blank", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: /update api key/i }));

    // Required-field validation blocks submit; give it a tick then assert no call.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("renders the required message when the field is left blank", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: /update api key/i }));

    expect(await screen.findByText("Enter a new API key")).toBeInTheDocument();
  });

  it("rejects a whitespace-only key without sending a PATCH", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "   " } });
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    expect(await screen.findByText("Enter a new API key")).toBeInTheDocument();
    changeLanguage("zh-CN");
    expect(screen.getByText("请输入新的 API 密钥。")).toBeInTheDocument();
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("trims surrounding whitespace off the key it sends", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/new api key/i), { target: { value: "  sk-pad-77  " } });
    await user.click(screen.getByRole("button", { name: /update api key/i }));

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(mockModelPatchUpdateCall.mock.calls[0][1]).toEqual({
      litellm_params: { api_key: "sk-pad-77" },
      model_info: { id: "model-123" },
    });
  });

  it("submits on Enter from inside the key field", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByLabelText(/new api key/i), "sk-enter-1{Enter}");

    await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(mockModelPatchUpdateCall.mock.calls[0][1]).toEqual({
      litellm_params: { api_key: "sk-enter-1" },
      model_info: { id: "model-123" },
    });
  });

  it("reveals and re-hides the key without touching the value", async () => {
    const user = userEvent.setup();
    renderModal();

    const field = screen.getByLabelText(/new api key/i);
    fireEvent.change(field, { target: { value: "sk-peek-42" } });
    expect(field).toHaveAttribute("type", "password");

    await user.click(screen.getByRole("button", { name: /show password/i }));
    expect(field).toHaveAttribute("type", "text");
    expect(field).toHaveValue("sk-peek-42");

    await user.click(screen.getByRole("button", { name: /hide password/i }));
    expect(field).toHaveAttribute("type", "password");
    expect(field).toHaveValue("sk-peek-42");
  });
});
