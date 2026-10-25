import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import CloudZeroExportModal from "./cloudzero_export_modal";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Toaster } from "@/components/ui/sonner";
import { translate } from "@/i18n/language";
import { toast } from "@/lib/toast";

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), info: vi.fn() },
}));

vi.mock("@/components/networking", () => ({
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const jsonResponse = (status: number, body: unknown) =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body }) as Response;

const bodyOf = (call: unknown[]) => JSON.parse((call[1] as RequestInit).body as string);

const callsTo = (fetchMock: ReturnType<typeof vi.fn>, path: string) =>
  fetchMock.mock.calls.filter((call) => call[0] === path);

describe("CloudZeroExportModal", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn(async (url: string) => {
      if (url === "/cloudzero/settings") return jsonResponse(404, { error: "not configured" });
      if (url === "/cloudzero/init") return jsonResponse(200, { message: "saved" });
      if (url === "/cloudzero/export") return jsonResponse(200, { message: "exported" });
      return jsonResponse(500, { error: "unexpected" });
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  const open = () => render(<CloudZeroExportModal isOpen onClose={vi.fn()} accessToken="sk-test" />);

  it("saves the settings with a hardcoded UTC timezone before exporting", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(1));

    const [initCall] = callsTo(fetchMock, "/cloudzero/init");
    expect((initCall[1] as RequestInit).method).toBe("POST");
    expect(bodyOf(initCall)).toEqual({
      api_key: "cz-key-123",
      connection_id: "conn-abc",
      timezone: "UTC",
    });
  });

  it("exports with a fixed limit and operation once the settings are saved", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));

    const [exportCall] = callsTo(fetchMock, "/cloudzero/export");
    expect((exportCall[1] as RequestInit).method).toBe("POST");
    expect(bodyOf(exportCall)).toEqual({ limit: 100000, operation: "replace_hourly" });
  });

  it("sends nothing while a required field is still empty", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await screen.findByText("Please enter the CloudZero connection ID");
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
  });

  it("reports both required messages when the form is untouched", async () => {
    const user = userEvent.setup();
    open();

    await screen.findByLabelText("CloudZero API Key");
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(screen.getByText("Please enter the CloudZero connection ID")).toBeInTheDocument();
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
  });

  it("skips the save entirely when CloudZero is already configured", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/cloudzero/settings")
        return jsonResponse(200, {
          api_key_masked: "cz-1****4567",
          connection_id: "conn-existing",
          status: "configured",
        });
      if (url === "/cloudzero/export") return jsonResponse(200, { message: "exported" });
      return jsonResponse(500, { error: "unexpected" });
    });
    const user = userEvent.setup();
    open();

    await screen.findByText(/conn-existing/);
    expect(screen.queryByLabelText("CloudZero API Key")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
  });

  it("masks the API key it echoes back after a successful save", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123456789" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));
    expect(bodyOf(callsTo(fetchMock, "/cloudzero/init")[0]).api_key).toBe("cz-key-123456789");
  });

  it("switches to the CSV destination without touching the CloudZero endpoints", async () => {
    const user = userEvent.setup();
    open();

    await screen.findByLabelText("CloudZero API Key");
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Export to CSV" }));

    expect(screen.queryByLabelText("CloudZero API Key")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Export CSV" }));

    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
  });

  it("keeps the API key entry obscured", async () => {
    open();
    expect(await screen.findByLabelText("CloudZero API Key")).toHaveAttribute("type", "password");
  });
});

describe("CloudZero 导出的语言边界", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    localStorage.clear();
    vi.clearAllMocks();
    fetchMock = vi.fn(async (url: string) => {
      if (url === "/cloudzero/settings") return jsonResponse(404, {});
      return jsonResponse(200, {});
    });
    vi.stubGlobal("fetch", fetchMock);
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    vi.mocked(toast.success).mockImplementation(realToast.success);
    vi.mocked(toast.fromError).mockImplementation(realToast.fromError);
    vi.mocked(toast.info).mockImplementation(realToast.info);
  });

  afterEach(async () => {
    const realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    act(() => realToast.dismiss());
    vi.mocked(toast.success).mockReset();
    vi.mocked(toast.fromError).mockReset();
    vi.mocked(toast.info).mockReset();
    vi.unstubAllGlobals();
  });

  const renderLocalized = (accessToken: string | null = "sk-test") => {
    const onClose = vi.fn();
    const tree = (defaultLanguage: "en" | "zh-CN") => (
      <LanguageProvider defaultLanguage={defaultLanguage} userId="cloudzero-export-review" storageScope="/usage">
        <CloudZeroExportModal isOpen onClose={onClose} accessToken={accessToken} />
        <Toaster />
      </LanguageProvider>
    );
    const view = render(tree("en"));
    return { onClose, changeLanguage: (language: "en" | "zh-CN") => view.rerender(tree(language)) };
  };

  const fillSettings = async () => {
    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "synthetic-key" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "Home" } });
  };

  it("已有必填错误切换语言，不重新加载配置或清除已填密钥", async () => {
    const view = renderLocalized();
    await screen.findByLabelText("CloudZero API Key");
    await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(screen.getByText("Please enter the CloudZero connection ID")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", "Please enter your CloudZero API key"))).toBeInTheDocument();
    expect(screen.getByText(translate("zh-CN", "Please enter the CloudZero connection ID"))).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(translate("zh-CN", "CloudZero API Key")), {
      target: { value: "synthetic-key" },
    });
    view.changeLanguage("en");
    expect(screen.getByLabelText("CloudZero API Key")).toHaveValue("synthetic-key");
    expect(screen.getByLabelText("CloudZero API Key")).toHaveAttribute("type", "password");
    expect(callsTo(fetchMock, "/cloudzero/settings")).toHaveLength(1);
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
  });

  it("保存和导出的本地成功提示在等待中及显示后切换语言，载荷不变", async () => {
    const save = Promise.withResolvers<Response>();
    const exported = Promise.withResolvers<Response>();
    fetchMock.mockImplementation((url: string) => {
      if (url === "/cloudzero/settings") return Promise.resolve(jsonResponse(404, {}));
      return url === "/cloudzero/init" ? save.promise : exported.promise;
    });
    const view = renderLocalized();
    await fillSettings();
    await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(1));
    view.changeLanguage("zh-CN");
    await act(async () => save.resolve(jsonResponse(200, {})));
    expect(await screen.findByText(translate("zh-CN", "CloudZero settings saved successfully"))).toBeInTheDocument();
    expect(bodyOf(callsTo(fetchMock, "/cloudzero/init")[0])).toEqual({
      api_key: "synthetic-key",
      connection_id: "Home",
      timezone: "UTC",
    });
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1);
    await act(async () => exported.resolve(jsonResponse(200, {})));
    expect(
      await screen.findByText(translate("zh-CN", "Export to CloudZero completed successfully")),
    ).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("CloudZero settings saved successfully")).toBeInTheDocument();
    expect(screen.getByText("Export to CloudZero completed successfully")).toBeInTheDocument();
    expect(bodyOf(callsTo(fetchMock, "/cloudzero/export")[0])).toEqual({ limit: 100000, operation: "replace_hourly" });
    expect(callsTo(fetchMock, "/cloudzero/settings")).toHaveLength(1);
    expect(view.onClose).toHaveBeenCalledTimes(1);
  });

  it("加载配置失败时保留服务端正文与权限分类，不展示错误包装的调试内容", async () => {
    const pending = Promise.withResolvers<Response>();
    fetchMock.mockReturnValue(pending.promise);
    const view = renderLocalized();
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(jsonResponse(403, { error: { message: "Home" }, debug: "private-detail" })));
    expect(await screen.findByText(translate("zh-CN", "Access Denied"))).toBeInTheDocument();
    expect(
      screen.getByText(translate("zh-CN", "Failed to load existing settings: {error}", { error: "Home" })),
    ).toBeInTheDocument();
    expect(screen.queryByText(/private-detail|\[object Object\]/)).not.toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText("Failed to load existing settings: Home")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each(["save", "export"] as const)("%s 的空错误正文使用可切换的本地提示，失败不关闭弹窗", async (operation) => {
    const pending = Promise.withResolvers<Response>();
    fetchMock.mockImplementation((url: string) => {
      if (url === "/cloudzero/settings") return Promise.resolve(jsonResponse(404, {}));
      if (url === "/cloudzero/init" && operation === "export") return Promise.resolve(jsonResponse(200, {}));
      return pending.promise;
    });
    const view = renderLocalized();
    await fillSettings();
    await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
    const endpoint = operation === "save" ? "/cloudzero/init" : "/cloudzero/export";
    await waitFor(() => expect(callsTo(fetchMock, endpoint)).toHaveLength(1));
    view.changeLanguage("zh-CN");
    await act(async () => pending.resolve(jsonResponse(operation === "save" ? 403 : 429, {})));
    const message = operation === "save" ? "Failed to save CloudZero settings" : "Failed to export to CloudZero";
    expect(await screen.findByText(translate("zh-CN", message))).toBeInTheDocument();
    expect(
      screen.getByText(translate("zh-CN", operation === "save" ? "Access Denied" : "Rate Limit Exceeded")),
    ).toBeInTheDocument();
    view.changeLanguage("en");
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(callsTo(fetchMock, endpoint)).toHaveLength(1);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(operation === "save" ? 0 : 1);
    expect(view.onClose).not.toHaveBeenCalled();
  });

  it.each(["save", "export"] as const)("%s 的服务端同名错误保留原文，不误用本地词典", async (operation) => {
    const message = operation === "save" ? "Failed to save CloudZero settings" : "Failed to export to CloudZero";
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/cloudzero/settings") return jsonResponse(404, {});
      if (url === "/cloudzero/init" && operation === "export") return jsonResponse(200, {});
      return jsonResponse(400, { error: message });
    });
    const view = renderLocalized();
    await fillSettings();
    view.changeLanguage("zh-CN");
    await userEvent.setup().click(screen.getByRole("button", { name: translate("zh-CN", "Export to CloudZero") }));
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.queryByText(translate("zh-CN", message))).not.toBeInTheDocument();
    expect(view.onClose).not.toHaveBeenCalled();
  });

  it("服务端同名成功消息保留原文，成功仍只导出一次", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/cloudzero/settings") return jsonResponse(404, {});
      return jsonResponse(200, {
        message:
          url === "/cloudzero/init"
            ? "CloudZero settings saved successfully"
            : "Export to CloudZero completed successfully",
      });
    });
    const view = renderLocalized();
    await fillSettings();
    view.changeLanguage("zh-CN");
    await userEvent.setup().click(screen.getByRole("button", { name: translate("zh-CN", "Export to CloudZero") }));
    expect(await screen.findByText("CloudZero settings saved successfully")).toBeInTheDocument();
    expect(await screen.findByText("Export to CloudZero completed successfully")).toBeInTheDocument();
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1);
    expect(view.onClose).toHaveBeenCalledTimes(1);
  });

  it("缺少登录凭证时本地通知随语言切换，不发送配置和导出请求", async () => {
    const view = renderLocalized(null);
    await fillSettings();
    await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
    expect(await screen.findByText("No access token available")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", "No access token available"))).toBeInTheDocument();
    expect(screen.getByLabelText(translate("zh-CN", "Connection ID"))).toHaveValue("Home");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(view.onClose).not.toHaveBeenCalled();
  });

  it.each(["load", "save", "export"] as const)(
    "%s 的非 JSON 错误响应保留 HTTP 分类和本地可切换提示",
    async (operation) => {
      const invalid = {
        ok: false,
        status: 403,
        json: async () => {
          throw new SyntaxError("private-parser-detail");
        },
      } as unknown as Response;
      fetchMock.mockImplementation(async (url: string) => {
        if (url === "/cloudzero/settings") return operation === "load" ? invalid : jsonResponse(404, {});
        if (url === "/cloudzero/init" && operation === "export") return jsonResponse(200, {});
        return invalid;
      });
      const view = renderLocalized();
      if (operation !== "load") {
        await fillSettings();
        await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
      }
      const message = {
        load: "Failed to load existing settings",
        save: "Failed to save CloudZero settings",
        export: "Failed to export to CloudZero",
      }[operation];
      expect(await screen.findByText(message)).toBeInTheDocument();
      expect(screen.getByText("Access Denied")).toBeInTheDocument();
      view.changeLanguage("zh-CN");
      expect(screen.getByText(translate("zh-CN", message))).toBeInTheDocument();
      expect(screen.getByText(translate("zh-CN", "Access Denied"))).toBeInTheDocument();
      expect(screen.queryByText(/private-parser-detail/)).not.toBeInTheDocument();
      expect(view.onClose).not.toHaveBeenCalled();
    },
  );

  it("CSV 占位通知随语言切换，不触发导出或保存请求", async () => {
    const user = userEvent.setup();
    const view = renderLocalized();
    await screen.findByLabelText("CloudZero API Key");
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Export to CSV" }));
    await user.click(screen.getByRole("button", { name: "Export CSV" }));
    expect(await screen.findByText("CSV export functionality coming soon!")).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", "CSV export functionality coming soon!"))).toBeInTheDocument();
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
    expect(view.onClose).toHaveBeenCalledTimes(1);
  });

  it.each(["load", "save", "export"] as const)("%s 的网络异常使用本地可切换提示，不显示内部异常", async (operation) => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/cloudzero/settings" && operation !== "load") return jsonResponse(404, {});
      if (url === "/cloudzero/init" && operation === "export") return jsonResponse(200, {});
      throw new TypeError("private-network-detail");
    });
    const view = renderLocalized();
    if (operation !== "load") {
      await fillSettings();
      await userEvent.setup().click(screen.getByRole("button", { name: "Export to CloudZero" }));
    }
    const message = {
      load: "Failed to load existing settings",
      save: "Failed to save CloudZero settings",
      export: "Failed to export to CloudZero",
    }[operation];
    expect(await screen.findByText(message)).toBeInTheDocument();
    view.changeLanguage("zh-CN");
    expect(screen.getByText(translate("zh-CN", message))).toBeInTheDocument();
    expect(screen.queryByText(/private-network-detail/)).not.toBeInTheDocument();
    expect(view.onClose).not.toHaveBeenCalled();
  });
});
