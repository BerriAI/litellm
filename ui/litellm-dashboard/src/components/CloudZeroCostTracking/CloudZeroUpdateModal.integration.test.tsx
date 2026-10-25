import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

const mutate = vi.fn();
const updateHook = vi.hoisted(() => vi.fn());
const authorization = vi.hoisted(() => ({ accessToken: "test-token" }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => authorization,
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings", () => ({
  useCloudZeroUpdateSettings: (accessToken: string) => updateHook(accessToken),
}));

const STORED_SETTINGS: CloudZeroSettings = {
  connection_id: "stored-connection-id",
  api_key_masked: "sk-cz-****last4",
  timezone: "Europe/Berlin",
  status: "Active",
};

const renderModal = (settings: CloudZeroSettings = STORED_SETTINGS) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onOk = vi.fn();
  const onCancel = vi.fn();
  const tree = (defaultLanguage: "en" | "zh-CN") => (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider defaultLanguage={defaultLanguage} userId="cloudzero-update" storageScope="cloudzero-update">
        <CloudZeroUpdateModal open={true} onOk={onOk} onCancel={onCancel} settings={settings} />
        <Toaster />
      </LanguageProvider>
    </QueryClientProvider>
  );
  const view = render(tree("en"));
  return { ...view, onOk, changeLanguage: (value: "en" | "zh-CN") => view.rerender(tree(value)) };
};

const submittedPayload = (): Record<string, unknown> => {
  expect(mutate).toHaveBeenCalledTimes(1);
  return mutate.mock.calls[0][0] as Record<string, unknown>;
};

describe("CloudZeroUpdateModal submit payload", () => {
  beforeEach(() => {
    localStorage.clear();
    authorization.accessToken = "test-token";
    mutate.mockClear();
    updateHook.mockReturnValue({ mutate, isPending: false });
  });

  it("seeds the stored connection id and timezone but never the stored key", () => {
    renderModal();

    expect(screen.getByLabelText("Connection ID")).toHaveValue("stored-connection-id");
    expect(screen.getByLabelText("Timezone")).toHaveValue("Europe/Berlin");
    expect(screen.getByLabelText("CloudZero API Key")).toHaveValue("");
  });

  it("omits api_key entirely when the key field is left untouched, preserving the stored secret", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "Europe/Berlin",
      }),
    );
    expect("api_key" in submittedPayload()).toBe(false);
  });

  it("sends api_key only once the user types a replacement key", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-rotated-key" } });
    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "Europe/Berlin",
        api_key: "cz-rotated-key",
      }),
    );
  });

  it("falls back to UTC when the stored timezone is cleared", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Timezone"));
    await user.click(screen.getByRole("button", { name: "Update" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "stored-connection-id",
        timezone: "UTC",
      }),
    );
  });

  it("falls back to UTC when the stored settings carry no timezone", () => {
    renderModal({ ...STORED_SETTINGS, timezone: null });

    expect(screen.getByLabelText("Timezone")).toHaveValue("UTC");
  });

  it("reports a null connection id from the server as the required field, not as a type error", async () => {
    const user = userEvent.setup();
    renderModal({ ...STORED_SETTINGS, connection_id: null, timezone: null });

    await user.click(screen.getByRole("button", { name: "Update" }));

    expect(await screen.findByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(screen.queryByText(/expected string, received null/i)).not.toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("blocks submission when the connection id is cleared, and leaves the key optional", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Connection ID"));
    await user.click(screen.getByRole("button", { name: "Update" }));

    expect(await screen.findByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(screen.queryByText("Please enter your CloudZero API key")).not.toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not submit when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByLabelText("CloudZero API Key"), "cz-rotated-key{Enter}");

    expect(mutate).not.toHaveBeenCalled();
  });
});

describe("CloudZero 编辑表单的语言边界", () => {
  const fetchRequest = vi.fn<typeof fetch>();
  let restoreFetch: () => void;

  beforeEach(async () => {
    localStorage.clear();
    authorization.accessToken = "test-token";
    fetchRequest.mockReset().mockResolvedValue(new Response("{}"));
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(fetchRequest);
    restoreFetch = () => fetchSpy.mockRestore();
    const actual = await vi.importActual<typeof import("@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings")>(
      "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings",
    );
    updateHook.mockImplementation(actual.useCloudZeroUpdateSettings);
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    vi.mocked(toast.success).mockImplementation(realToast.success);
    vi.mocked(toast.error).mockImplementation(realToast.error);
  });

  afterEach(async () => {
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    act(() => realToast.dismiss());
    vi.mocked(toast.success).mockReset();
    vi.mocked(toast.error).mockReset();
    updateHook.mockReset();
    restoreFetch();
  });

  it("已有连接 ID 错误随语言更新，替换密钥草稿和可选语义保持不变", async () => {
    const user = userEvent.setup();
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Update", exact: true }));
    expect(await screen.findByText("Please enter your CloudZero connection ID")).toBeInTheDocument();

    view.changeLanguage("zh-CN");
    expect(screen.getByText("请输入 CloudZero 连接 ID")).toBeInTheDocument();
    expect(screen.queryByText("请输入 CloudZero API 密钥")).not.toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
    expect(screen.getByLabelText("时区")).toHaveValue("Europe/Berlin");
    expect(fetchRequest).not.toHaveBeenCalled();
    view.changeLanguage("en");
    expect(screen.getByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it("更新成功提示随语言切换，空 API 密钥仍不发送", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    fetchRequest.mockReturnValue(pending.promise);
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Update", exact: true }));
    await waitFor(() => expect(fetchRequest).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(new Response("{}")));

    expect(await screen.findByText("CloudZero 集成已更新")).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("CloudZero integration updated successfully")).toBeInTheDocument();
    expect(JSON.parse(String(fetchRequest.mock.calls[0][1]?.body))).toEqual({
      connection_id: "Home",
      timezone: "Europe/Berlin",
    });
    expect(fetchRequest).toHaveBeenCalledTimes(1);
    expect(view.onOk).toHaveBeenCalledOnce();
  });

  // 非 JSON 或空错误正文只能生成本地兜底，服务端明确详情仍按原文显示。
  it.each(["{}", "not-json"])("错误正文 %s 没有可用详情时，本地提示随语言更新", async (body) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    fetchRequest.mockReturnValue(pending.promise);
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Update", exact: true }));
    await waitFor(() => expect(fetchRequest).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(new Response(body, { status: 500 })));

    expect(await screen.findByText("无法更新 CloudZero 设置")).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Failed to update CloudZero settings")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API Key")).toHaveValue("Home");
    expect(fetchRequest).toHaveBeenCalledTimes(1);
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it("缺少令牌在本地拒绝，提示可切换语言且不清空替换密钥", async () => {
    authorization.accessToken = "";
    const user = userEvent.setup();
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Update", exact: true }));
    expect(await screen.findByText("Access token is required")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText("需要访问令牌")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
    expect(fetchRequest).not.toHaveBeenCalled();
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it.each([
    "Home",
    "Failed to update CloudZero integration",
    "Failed to update CloudZero settings",
    "Access token is required",
  ])("服务端详情 %s 不翻译，失败不清空替换密钥", async (message) => {
    const user = userEvent.setup();
    fetchRequest.mockResolvedValue(new Response(JSON.stringify({ detail: message }), { status: 400 }));
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Update", exact: true }));
    expect(await screen.findByText(message)).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
    expect(fetchRequest).toHaveBeenCalledTimes(1);
    expect(view.onOk).not.toHaveBeenCalled();
  });
});
