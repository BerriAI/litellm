import { render, screen, fireEvent, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import UISettings from "./UISettings";
import { toast } from "@/lib/toast";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/lib/toast");

const mockUseAuthorized = vi.hoisted(() => vi.fn());
const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockUseUpdateUISettings = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings", () => ({
  useUpdateUISettings: mockUseUpdateUISettings,
}));

const buildSettingsResponse = (overrides?: Partial<Record<string, unknown>>) => ({
  data: {
    field_schema: {
      description: "UI settings description",
      properties: {
        default_ui_language: { enum: ["en", "zh-CN"], default: "en" },
        disable_model_add_for_internal_users: {
          description: "Disable model add for internal users",
        },
        disable_team_admin_delete_team_user: {
          description: "Disable team admin delete team user",
        },
        require_auth_for_public_ai_hub: {
          description: "Require authentication for public AI Hub",
        },
      },
    },
    values: {
      disable_model_add_for_internal_users: false,
      disable_team_admin_delete_team_user: false,
      require_auth_for_public_ai_hub: false,
    },
  },
  isLoading: false,
  isError: false,
  error: null,
  ...overrides,
});

describe("UISettings", () => {
  it("汉化默认用量范围及选项，提交原始设置值", async () => {
    const mutate = vi.fn();
    mockUseUpdateUISettings.mockReturnValue({ mutate, isPending: false, error: null });
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="usage-i18n" storageScope="usage-i18n">
        <UISettings />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox", { name: "默认用量时间范围" })).toHaveTextContent("最近 7 天");
    await chooseSelectOption(userEvent.setup(), screen.getByRole("combobox", { name: "默认用量时间范围" }), "本月至今");
    expect(mutate).toHaveBeenCalledExactlyOnceWith({ usage_default_time_range: "month_to_date" }, expect.any(Object));
  });

  it("汉化配置管理的用量范围且保持只读", () => {
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          values: { usage_default_time_range: "year_to_date" },
          source: { usage_default_time_range: "config" },
        },
      }),
    );
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="usage-config" storageScope="usage-i18n">
        <UISettings />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox", { name: "默认用量时间范围" })).toHaveTextContent("本年至今");
    expect(screen.getByRole("combobox", { name: "默认用量时间范围" })).toBeDisabled();
    expect(screen.getByText("由部署配置管理。")).toBeVisible();
  });

  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "proxy_admin" });
    mockUseUISettings.mockReturnValue(buildSettingsResponse());
    mockUseUpdateUISettings.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["language", "usage", "switch", "error"] as const)(
    "异步 %s 通知使用完成时的语言，切换语言不重复保存",
    async (operation) => {
      const mutate = vi.fn();
      mockUseUpdateUISettings.mockReturnValue({ mutate, isPending: false, error: null });
      render(
        <LanguageProvider defaultLanguage="en" userId="settings-notices" storageScope="settings-notices">
          <LanguageSelector />
          <UISettings />
          <Toaster expand />
        </LanguageProvider>,
      );
      if (operation === "usage") {
        await chooseSelectOption(
          userEvent.setup(),
          screen.getByRole("combobox", { name: "Default Usage time range" }),
          "Month to date",
        );
      } else if (operation === "switch") {
        fireEvent.click(screen.getByRole("switch", { name: "Disable model add for internal users" }));
      } else {
        fireEvent.change(screen.getByLabelText("Default UI language"), { target: { value: "zh-CN" } });
      }
      expect(mutate).toHaveBeenCalledOnce();
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      act(() => {
        const options = mutate.mock.calls[0][1];
        if (operation === "error") options.onError(new Error("private-stack-marker"));
        else options.onSuccess();
      });
      const chinese = operation === "error" ? "无法更新界面设置" : "界面设置已更新";
      expect(await screen.findByText(chinese, { exact: true })).toBeVisible();
      expect(screen.queryByText(/private-stack-marker/)).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(
        screen.getByText(operation === "error" ? "Could not update UI settings" : "UI settings updated successfully", {
          exact: true,
        }),
      ).toBeVisible();
      expect(mutate).toHaveBeenCalledOnce();
    },
  );

  it("saves the selected deployment default through existing UI settings", async () => {
    const mutate = vi.fn();
    mockUseUpdateUISettings.mockReturnValue({ mutate, isPending: false, error: null });
    render(<UISettings />);
    expect(screen.getByRole("combobox", { name: "Default Usage time range" })).toHaveTextContent("Last 7 days");
    await chooseSelectOption(
      userEvent.setup(),
      screen.getByRole("combobox", { name: "Default Usage time range" }),
      "Month to date",
    );
    expect(mutate).toHaveBeenCalledWith({ usage_default_time_range: "month_to_date" }, expect.any(Object));
  });

  it("shows configuration-owned defaults without offering an ineffective override", () => {
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          values: { usage_default_time_range: "year_to_date" },
          source: { usage_default_time_range: "config" },
        },
      }),
    );
    render(<UISettings />);
    expect(screen.getByRole("combobox", { name: "Default Usage time range" })).toHaveTextContent("Year to date");
    expect(screen.getByRole("combobox", { name: "Default Usage time range" })).toBeDisabled();
    expect(screen.getByText("Managed by the deployment configuration.")).toBeInTheDocument();
  });

  it("should render", () => {
    render(<UISettings />);

    expect(screen.getByText("UI Settings")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Disable model add for internal users" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Disable team admin delete team user" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Require authentication for public AI Hub" })).toBeInTheDocument();
  });

  it("汉化固定标签但保留后端 schema 描述原文", () => {
    localStorage.clear();
    const response = buildSettingsResponse();
    response.data.field_schema.description = "Home";
    response.data.field_schema.properties.disable_model_add_for_internal_users.description = "Settings";
    mockUseUISettings.mockReturnValue(response);
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="schema-test" storageScope="management">
        <UISettings />
      </LanguageProvider>,
    );
    expect(screen.getByText("Home")).toBeVisible();
    expect(screen.getByText("Settings")).toBeVisible();
    expect(screen.getByRole("switch", { name: "Settings" })).toBeVisible();
    expect(screen.getByRole("combobox", { name: "全站默认语言" })).toBeVisible();
  });

  it("加载失败时允许重试且不显示内部错误", () => {
    const refetch = vi.fn();
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        isError: true,
        error: new Error("internal database stack trace"),
        refetch,
      }),
    );
    render(<UISettings />);
    expect(screen.getByText("Could not load UI settings")).toBeVisible();
    expect(screen.queryByText("internal database stack trace")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(refetch).toHaveBeenCalledOnce();
  });

  it("保存全站默认语言时不覆盖其他设置", () => {
    const mutate = vi.fn();
    mockUseUpdateUISettings.mockReturnValue({ mutate, isPending: false, error: null });
    render(<UISettings />);
    fireEvent.change(screen.getByRole("combobox", { name: "Default UI language" }), { target: { value: "zh-CN" } });
    expect(mutate).toHaveBeenCalledWith(
      { default_ui_language: "zh-CN" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("配置文件管理的默认语言不能在界面覆盖", () => {
    const response = buildSettingsResponse();
    mockUseUISettings.mockReturnValue({
      ...response,
      data: { ...response.data, source: { default_ui_language: "config" } },
    });
    render(<UISettings />);
    expect(screen.getByRole("combobox", { name: "Default UI language" })).toBeDisabled();
    expect(screen.getByText("Managed by configuration file")).toBeInTheDocument();
  });

  it("默认语言保存失败时只显示安全提示和错误码", async () => {
    const error = new Error("internal database stack trace");
    mockUseUpdateUISettings.mockReturnValue({
      mutate: vi.fn((_settings, options) => options.onError(error)),
      isPending: false,
      error,
    });
    render(
      <>
        <UISettings />
        <Toaster expand />
      </>,
    );
    fireEvent.change(screen.getByRole("combobox", { name: "Default UI language" }), { target: { value: "zh-CN" } });
    expect(await screen.findByRole("listitem", { hidden: true })).toHaveTextContent("Could not update UI settings");
    expect(screen.getByText(/UI_SETTINGS_WRITE_FAILED/)).toBeVisible();
    expect(screen.queryByText(error.message)).not.toBeInTheDocument();
  });

  it("只读管理员不能修改默认语言", () => {
    // 真实鉴权钩子将只读管理员映射为 Admin，并单独返回只读标记
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin", isViewOnly: true });
    render(<UISettings />);
    expect(screen.getByRole("combobox", { name: "Default UI language" })).toBeDisabled();
  });

  it("should toggle setting and call update", async () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(
      <>
        <UISettings />
        <Toaster expand />
      </>,
    );

    const toggle = screen.getByRole("switch", { name: "Disable model add for internal users" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { disable_model_add_for_internal_users: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(await screen.findByText("UI settings updated successfully")).toBeVisible();
  });

  it("should toggle disable team admin delete team user setting and call update", async () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(
      <>
        <UISettings />
        <Toaster expand />
      </>,
    );

    const toggle = screen.getByRole("switch", { name: "Disable team admin delete team user" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { disable_team_admin_delete_team_user: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(await screen.findByText("UI settings updated successfully")).toBeVisible();
  });

  it("disconnects Moyai by patching moyai_url to null", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          ...buildSettingsResponse().data,
          values: { ...buildSettingsResponse().data.values, moyai_url: "https://moyai.example.com" },
        },
      }),
    );

    render(<UISettings />);

    expect(screen.getByText("Connected to https://moyai.example.com")).toBeInTheDocument();
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { moyai_url: null },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("shows a link to the Moyai page when not connected", () => {
    render(<UISettings />);

    expect(screen.getByText("Not connected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Connect from the Moyai page" })).toHaveAttribute("href", "/ui/moyai");
  });

  it("should toggle require auth for public AI Hub setting and call update", async () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(
      <>
        <UISettings />
        <Toaster expand />
      </>,
    );

    const toggle = screen.getByRole("switch", { name: "Require authentication for public AI Hub" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { require_auth_for_public_ai_hub: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(await screen.findByText("UI settings updated successfully")).toBeVisible();
  });
});
