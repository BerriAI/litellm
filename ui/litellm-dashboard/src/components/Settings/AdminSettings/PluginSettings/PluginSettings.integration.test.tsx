import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PluginSettings from "./PluginSettings";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";

const { getConfigFieldSettingMock, updateConfigFieldSettingMock } = vi.hoisted(() => ({
  getConfigFieldSettingMock: vi.fn(),
  updateConfigFieldSettingMock: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  getConfigFieldSetting: getConfigFieldSettingMock,
  updateConfigFieldSetting: updateConfigFieldSettingMock,
}));

const REDACTED_PLUGIN = {
  name: "alpha",
  display_name: "Alpha",
  url: "https://alpha.example.com",
  plugin_key: "***",
};

const savedPayload = () => updateConfigFieldSettingMock.mock.calls[0];

describe("PluginSettings config payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    updateConfigFieldSettingMock.mockResolvedValue({});
  });

  it("插件字段错误切换语言时即时更新并保留草稿和请求次数", async () => {
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [] });
    render(
      <LanguageProvider defaultLanguage="en" userId="plugin-errors" storageScope="management">
        <LanguageSelector />
        <PluginSettings />
      </LanguageProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Add Plugin" }));
    fireEvent.change(screen.getByLabelText("Name (identifier)"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Display Name"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("URL"), { target: { value: "invalid" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByText("Must be a valid URL")).toBeVisible();

    // 弹窗有焦点隔离，直接切换选择器来验证上下文传播，不关闭表单。
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(await screen.findByText("请输入有效的 URL")).toBeVisible();
    expect(screen.getByLabelText("名称（标识符）")).toHaveValue("Home");
    expect(screen.getByLabelText("显示名称")).toHaveValue("Settings");
    expect(screen.getByLabelText("URL")).toHaveValue("invalid");
    expect(screen.queryByText("必填")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(await screen.findByText("Must be a valid URL")).toBeVisible();
    expect(getConfigFieldSettingMock).toHaveBeenCalledTimes(1);
    expect(updateConfigFieldSettingMock).not.toHaveBeenCalled();
  });

  it("使用中文校验提示，同时保留插件名称、URL 和提交字段", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [] });
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="plugin-language" storageScope="management">
        <PluginSettings />
      </LanguageProvider>,
    );
    await user.click(await screen.findByRole("button", { name: "添加插件" }));
    await user.click(screen.getByRole("button", { name: "保存" }));
    expect(await screen.findAllByText("必填")).toHaveLength(3);
    expect(updateConfigFieldSettingMock).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("名称（标识符）"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("显示名称"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("URL"), { target: { value: "https://plugins.example.com" } });
    await user.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [{ name: "Home", display_name: "Settings", url: "https://plugins.example.com", plugin_key: undefined }],
    ]);
  });

  it("sends a new plugin with no plugin_key when the key field is left blank", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [] });
    render(<PluginSettings />);
    expect(await screen.findByText("No data", { ignore: "title" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add plugin/i }));
    fireEvent.change(await screen.findByLabelText(/Name \(identifier\)/), { target: { value: "beta" } });
    fireEvent.change(screen.getByLabelText(/Display Name/), { target: { value: "Beta" } });
    fireEvent.change(screen.getByLabelText(/^URL/), { target: { value: "https://beta.example.com" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "beta",
          display_name: "Beta",
          url: "https://beta.example.com",
          plugin_key: undefined,
        },
      ],
    ]);
  });

  it("seeds the key field blank on edit and sends a blank key when it is left untouched", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    expect(await screen.findByLabelText(/Plugin Key/)).toHaveValue("");

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "alpha",
          display_name: "Alpha",
          url: "https://alpha.example.com",
          plugin_key: "",
        },
      ],
    ]);
  });

  it("sends the typed key on edit when the key field is filled in", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    fireEvent.change(await screen.findByLabelText(/Plugin Key/), { target: { value: "sk-brand-new" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "alpha",
          display_name: "Alpha",
          url: "https://alpha.example.com",
          plugin_key: "sk-brand-new",
        },
      ],
    ]);
  });
});

describe("PluginSettings plugin key reveal (post-migration shadcn affordance)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
  });

  it("flips the key field between hidden and revealed and relabels the toggle", async () => {
    const user = userEvent.setup();
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    const keyInput = await screen.findByLabelText(/Plugin Key/);
    expect(keyInput).toHaveAttribute("type", "password");

    await user.click(screen.getByRole("button", { name: "Show plugin key" }));
    expect(keyInput).toHaveAttribute("type", "text");
    expect(screen.queryByRole("button", { name: "Show plugin key" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Hide plugin key" }));
    expect(keyInput).toHaveAttribute("type", "password");
    expect(screen.queryByRole("button", { name: "Hide plugin key" })).not.toBeInTheDocument();
  });
});
