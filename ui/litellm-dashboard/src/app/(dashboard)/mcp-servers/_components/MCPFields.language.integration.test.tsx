import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormProvider, useForm } from "react-hook-form";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import {
  MountedFormProvider,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import AwsSigV4Fields from "./AwsSigV4Fields";
import OAuthFormFields from "./OAuthFormFields";
import IdJagFormFields from "./IdJagFormFields";
import TokenExchangeFormFields from "./TokenExchangeFormFields";
import StdioConfiguration from "./StdioConfiguration";

function FormHarness({
  children,
  onSubmit = vi.fn(),
  defaultValues = {},
}: {
  children: React.ReactNode;
  onSubmit?: (values: MountedFormValues) => void;
  defaultValues?: MountedFormValues;
}) {
  const form = useForm<MountedFormValues>({ defaultValues });
  const registry = useMountRegistry();
  const { setPreference } = useLanguage();
  return (
    <FormProvider {...form}>
      <MountedFormProvider value={{ control: form.control, registry }}>
        <form onSubmit={form.handleSubmit(onSubmit)}>
          {children}
          <button type="button" onClick={() => setPreference("en")}>
            English
          </button>
          <button type="submit">提交</button>
        </form>
      </MountedFormProvider>
    </FormProvider>
  );
}

describe("MCP 认证字段汉化", () => {
  it("AWS 成对凭据校验使用中文且不会接受缺失的 Secret Access Key", async () => {
    const onSubmit = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="aws-language" storageScope="mcp-fields-test">
        <FormHarness
          onSubmit={onSubmit}
          defaultValues={{ credentials: { aws_region_name: "us-east-1", aws_access_key_id: "preview-only" } }}
        >
          <AwsSigV4Fields />
        </FormHarness>
      </LanguageProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "提交" }));
    expect(await screen.findByText("填写 Access Key ID 时，必须填写 Secret Access Key")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it.each([
    { name: "OAuth", fields: <OAuthFormFields isM2M />, message: "M2M OAuth 必须填写 Client ID" },
    { name: "ID-JAG", fields: <IdJagFormFields />, message: "ID-JAG 必须填写组织令牌端点" },
    { name: "令牌交换", fields: <TokenExchangeFormFields />, message: "令牌交换必须填写 Client ID" },
  ])("$name 必填错误显示中文", async ({ name, fields, message }) => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId={name} storageScope="mcp-fields-test">
        <FormHarness>{fields}</FormHarness>
      </LanguageProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "提交" }));
    expect(await screen.findByText(message)).toBeInTheDocument();
  });

  it("stdio 无效 JSON 显示中文，输入原文保留", async () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="stdio-language" storageScope="mcp-fields-test">
        <FormHarness>
          <StdioConfiguration isVisible />
        </FormHarness>
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "{Home" } });
    fireEvent.click(screen.getByRole("button", { name: "提交" }));
    expect(await screen.findByText("请输入有效的 JSON")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toHaveValue("{Home");
  });

  it("OAuth 编辑占位符、已选值、悬浮说明切换语言时保留草稿和原始枚举", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="oauth-edit-language" storageScope="mcp-fields-test">
        <FormHarness onSubmit={onSubmit} defaultValues={{ oauth_flow_type: "m2m" }}>
          <OAuthFormFields isM2M isEditing />
        </FormHarness>
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox", { name: "OAuth 流程类型" })).toHaveTextContent("机器到机器（M2M）");
    const clientId = screen.getByPlaceholderText("输入 OAuth 客户端 ID（留空保留现有值）");
    fireEvent.change(clientId, { target: { value: "Home" } });
    const label = screen.getByText("资源标识（可选）");
    const trigger = label.querySelector('[data-slot="tooltip-trigger"]');
    expect(trigger).not.toBeNull();
    await user.hover(trigger!);
    expect(await screen.findByText(/^发送给授权服务器的 RFC 8707 资源标识/)).toBeVisible();
    await user.unhover(trigger!);
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("combobox", { name: "OAuth Flow Type" })).toHaveTextContent("Machine-to-Machine (M2M)");
    expect(screen.getByPlaceholderText("Enter OAuth client ID (leave blank to keep existing)")).toHaveValue("Home");
    fireEvent.click(screen.getByRole("button", { name: "提交" }));
    expect(await screen.findByDisplayValue("Home")).toBeInTheDocument();
    await vi.waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          oauth_flow_type: "m2m",
          credentials: expect.objectContaining({ client_id: "Home" }),
        }),
        expect.anything(),
      ),
    );
  });
});
