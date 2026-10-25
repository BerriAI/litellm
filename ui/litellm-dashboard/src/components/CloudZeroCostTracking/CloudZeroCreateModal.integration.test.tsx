import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CloudZeroCreateModal from "./CloudZeroCreateModal";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

const mutate = vi.fn();
const createHook = vi.hoisted(() => vi.fn());
const authorization = vi.hoisted(() => ({ accessToken: "test-token" }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => authorization,
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate", () => ({
  useCloudZeroCreate: (accessToken: string) => createHook(accessToken),
}));

const renderModal = (language: "en" | "zh-CN" = "en") => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onOk = vi.fn();
  const onCancel = vi.fn();
  const tree = (defaultLanguage: "en" | "zh-CN") => (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider defaultLanguage={defaultLanguage} userId="cloudzero-language" storageScope="cloudzero-test">
        <CloudZeroCreateModal open={true} onOk={onOk} onCancel={onCancel} />
        <Toaster />
      </LanguageProvider>
    </QueryClientProvider>
  );
  const view = render(tree(language));
  return { ...view, onOk, changeLanguage: (value: "en" | "zh-CN") => view.rerender(tree(value)) };
};

const submittedPayload = (): Record<string, unknown> => {
  expect(mutate).toHaveBeenCalledTimes(1);
  return mutate.mock.calls[0][0] as Record<string, unknown>;
};

describe("CloudZeroCreateModal submit payload", () => {
  beforeEach(() => {
    localStorage.clear();
    authorization.accessToken = "test-token";
    mutate.mockClear();
    createHook.mockReturnValue({ mutate, isPending: false });
  });

  it("中文校验与密钥显隐可用，提交字段及用户输入保持原样", async () => {
    const user = userEvent.setup();
    renderModal("zh-CN");
    await user.click(screen.getByRole("button", { name: "创建", exact: true }));
    expect(await screen.findByText("请输入 CloudZero API 密钥")).toBeInTheDocument();
    expect(screen.getByText("请输入 CloudZero 连接 ID")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("CloudZero API 密钥"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("连接 ID"), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: "显示 API 密钥" }));
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveAttribute("type", "text");
    await user.click(screen.getByRole("button", { name: "创建", exact: true }));
    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({ api_key: "Home", connection_id: "Settings", timezone: "UTC" }),
    );
  });

  it("sends every filled field verbatim", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-42" } });
    fireEvent.change(screen.getByLabelText("Timezone"), { target: { value: "America/New_York" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "conn-42",
        timezone: "America/New_York",
        api_key: "cz-secret-key",
      }),
    );
  });

  it("defaults an untouched timezone to UTC", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-42" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "conn-42",
        timezone: "UTC",
        api_key: "cz-secret-key",
      }),
    );
  });

  it("blocks submission and shows both required messages when the form is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(screen.getByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not submit when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    await user.type(screen.getByLabelText("Connection ID"), "conn-42{Enter}");

    expect(mutate).not.toHaveBeenCalled();
  });
});

describe("CloudZero 创建表单的语言边界", () => {
  const fetchRequest = vi.fn<typeof fetch>();
  let restoreFetch: () => void;

  beforeEach(async () => {
    localStorage.clear();
    authorization.accessToken = "test-token";
    fetchRequest.mockReset().mockResolvedValue(new Response("{}"));
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(fetchRequest);
    restoreFetch = () => fetchSpy.mockRestore();
    // 本组保留真实 mutation 和通知，只替换 HTTP 边界。
    const actual = await vi.importActual<typeof import("@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate")>(
      "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate",
    );
    createHook.mockImplementation(actual.useCloudZeroCreate);
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    vi.mocked(toast.success).mockImplementation(realToast.success);
    vi.mocked(toast.error).mockImplementation(realToast.error);
  });

  afterEach(async () => {
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    act(() => realToast.dismiss());
    vi.mocked(toast.success).mockReset();
    vi.mocked(toast.error).mockReset();
    createHook.mockReset();
    restoreFetch();
  });

  it("已有必填错误随语言更新，输入草稿不重置且切换不发送请求", async () => {
    const user = userEvent.setup();
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("Timezone"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));
    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();

    view.changeLanguage("zh-CN");
    expect(screen.getByText("请输入 CloudZero API 密钥")).toBeInTheDocument();
    expect(screen.getByText("请输入 CloudZero 连接 ID")).toBeInTheDocument();
    expect(screen.getByLabelText("时区")).toHaveValue("Home");
    expect(fetchRequest).not.toHaveBeenCalled();
    view.changeLanguage("en");
    expect(screen.getByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it("成功提示在请求前后切换语言，敏感业务值只进入原始请求载荷", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    fetchRequest.mockReturnValue(pending.promise);
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));
    await waitFor(() => expect(fetchRequest).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
    await act(async () => pending.resolve(new Response("{}")));

    expect(await screen.findByText("CloudZero 集成已创建")).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("CloudZero integration created successfully")).toBeInTheDocument();
    expect(JSON.parse(String(fetchRequest.mock.calls[0][1]?.body))).toEqual({
      connection_id: "Home",
      timezone: "UTC",
      api_key: "Home",
    });
    expect(fetchRequest).toHaveBeenCalledTimes(1);
    expect(view.onOk).toHaveBeenCalledOnce();
  });

  // 本地兜底与服务端同名文案必须由来源区分，不能只比较错误字符串。
  it.each(["{}", "not-json"])("错误正文 %s 没有可用详情时，本地提示随语言更新", async (body) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Response>();
    fetchRequest.mockReturnValue(pending.promise);
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));
    await waitFor(() => expect(fetchRequest).toHaveBeenCalledTimes(1));
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(new Response(body, { status: 500 })));

    expect(await screen.findByText("无法创建 CloudZero 集成")).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Failed to create CloudZero integration")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API Key")).toHaveValue("Home");
    expect(fetchRequest).toHaveBeenCalledTimes(1);
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it("缺少令牌在本地拒绝，提示可切换语言且不清空草稿", async () => {
    authorization.accessToken = "";
    const user = userEvent.setup();
    const view = renderModal();
    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));
    expect(await screen.findByText("Access token is required")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText("需要访问令牌")).toBeInTheDocument();
    expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
    expect(fetchRequest).not.toHaveBeenCalled();
    expect(view.onOk).not.toHaveBeenCalled();
  });

  it.each(["Home", "Failed to create CloudZero integration", "Access token is required"])(
    "服务端详情 %s 不作为翻译键，失败保留草稿",
    async (message) => {
      const user = userEvent.setup();
      fetchRequest.mockResolvedValue(new Response(JSON.stringify({ detail: message }), { status: 400 }));
      const view = renderModal();
      fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "Home" } });
      fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
      await user.click(screen.getByRole("button", { name: "Create", exact: true }));
      expect(await screen.findByText(message)).toBeInTheDocument();
      view.changeLanguage("zh-CN");
      expect(screen.getByText(message)).toBeInTheDocument();
      expect(screen.getByLabelText("CloudZero API 密钥")).toHaveValue("Home");
      expect(fetchRequest).toHaveBeenCalledTimes(1);
      expect(view.onOk).not.toHaveBeenCalled();
    },
  );
});
