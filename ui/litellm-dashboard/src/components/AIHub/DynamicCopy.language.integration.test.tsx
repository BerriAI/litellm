import { act, fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { GuardrailTestPanel } from "@/app/(dashboard)/guardrails/_components/GuardrailTestPanel";
import GuardrailSelectionModal from "@/app/(dashboard)/policies/_components/guardrail_selection_modal";
import { SearchResultsDisplay } from "@/app/(dashboard)/playground/components/chat_ui/SearchResultsDisplay";
import MakeSkillPublicForm from "@/components/claude_code_plugins/MakeSkillPublicForm";
import type { Plugin } from "@/components/claude_code_plugins/types";
import MakeModelPublicForm from "./forms/MakeModelPublicForm";
import MakeAgentPublicForm from "./forms/MakeAgentPublicForm";
import MakeMCPPublicForm from "./forms/MakeMCPPublicForm";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

vi.unmock("@/lib/toast");

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("动态文案语言边界", () => {
  afterEach(() => {
    act(() => toast.dismiss());
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const publicationProps = { visible: true, accessToken: "token", onClose: vi.fn(), onSuccess: vi.fn() };
  it.each([
    {
      english: "Select Models",
      chinese: "选择模型",
      form: <MakeModelPublicForm {...publicationProps} modelHubData={[]} />,
    },
    {
      english: "Select Agents",
      chinese: "选择智能体",
      form: <MakeAgentPublicForm {...publicationProps} agentHubData={[]} />,
    },
    {
      english: "Select Servers",
      chinese: "选择服务器",
      form: <MakeMCPPublicForm {...publicationProps} mcpHubData={[]} />,
    },
    {
      english: "Select Skills",
      chinese: "选择技能",
      form: <MakeSkillPublicForm {...publicationProps} skillsList={[]} />,
    },
  ])("公开向导步骤 $english 支持中英文切换", ({ english, chinese, form }) => {
    const tree = (language: "zh-CN" | "en") => (
      <LanguageProvider defaultLanguage={language} userId={null} storageScope={`publication-steps-${english}`}>
        {form}
      </LanguageProvider>
    );
    const view = renderWithProviders(tree("zh-CN"));
    const steps = within(screen.getByRole("list"));
    expect(steps.getByText(chinese, { exact: true })).toBeInTheDocument();
    expect(steps.getByText("确认", { exact: true })).toBeInTheDocument();
    view.rerender(tree("en"));
    expect(steps.getByText(english, { exact: true })).toBeInTheDocument();
    expect(steps.getByText("Confirm", { exact: true })).toBeInTheDocument();
  });

  it("技能发布步骤和复数整句翻译，技能名及描述保留原文", () => {
    const skill: Plugin = {
      id: "skill-home",
      name: "Home",
      description: "Settings",
      source: {},
      enabled: true,
      created_at: null,
      updated_at: null,
      version: null,
    };
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="dynamic" storageScope="publish-skill-count">
        <MakeSkillPublicForm
          visible
          accessToken="token"
          skillsList={[skill, { ...skill, id: "skill-other", name: "Language" }]}
          onClose={vi.fn()}
          onSuccess={vi.fn()}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("选择技能")).toBeInTheDocument();
    expect(screen.getByText("将发布 2 个技能")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getAllByText("Settings")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "下一步" }));
    expect(screen.getByText("合计将发布 2 个技能")).toBeInTheDocument();
  });

  it("防护规则数量和本地校验按整句翻译，规则名和输入保持原值", () => {
    const onSubmit = vi.fn();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="dynamic" storageScope="guardrail-test-count">
        <GuardrailTestPanel
          guardrailNames={["Home", "Settings"]}
          onSubmit={onSubmit}
          isLoading={false}
          results={null}
          errors={null}
          onClose={vi.fn()}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("Home")).toBeInTheDocument();
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Settings" } });
    fireEvent.change(screen.getByPlaceholderText('{"forbidden_topics": ["tax", "finance"]}'), {
      target: { value: "[]" },
    });
    fireEvent.click(screen.getByRole("button", { name: "测试 2 条防护规则" }));
    expect(screen.getByText("元数据必须是 JSON 对象")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // 使用真实通知树，覆盖通知发出之后和异步复制期间的语言切换。
  it.each([
    { input: "", metadata: "", chinese: "请输入要测试的文本", english: "Please enter text to test" },
    {
      input: "Home",
      metadata: "[]",
      chinese: "元数据：元数据必须是 JSON 对象",
      english: "Metadata: Metadata must be a JSON object",
    },
    { input: "Home", metadata: "{invalid", chinese: "元数据：JSON 格式无效", english: "Metadata: Invalid JSON" },
  ])("已显示的防护规则校验通知随语言更新 $english", async ({ input, metadata, chinese, english }) => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId={null} storageScope={`guardrail-toast-${english}`}>
        <SwitchLanguage />
        <Toaster expand />
        <GuardrailTestPanel
          guardrailNames={["Home"]}
          onSubmit={vi.fn()}
          isLoading={false}
          results={null}
          errors={null}
          onClose={vi.fn()}
        />
      </LanguageProvider>,
    );
    const textboxes = screen.getAllByRole("textbox");
    fireEvent.change(textboxes[0], { target: { value: input } });
    fireEvent.change(textboxes[1], { target: { value: metadata } });
    fireEvent.keyDown(textboxes[0], { key: "Enter" });
    expect(await screen.findByText(chinese)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(await screen.findByText(english)).toBeVisible();
    expect(screen.queryByText(chinese)).not.toBeInTheDocument();
  });

  it.each([
    { success: true, english: "Input copied to clipboard" },
    { success: false, english: "Failed to copy input" },
  ])("复制期间切换语言使用新语言通知 $english", async ({ success, english }) => {
    const pendingCopy = Promise.withResolvers<void>();
    const copy = vi.fn(async () => {
      await pendingCopy.promise;
      if (!success) throw new Error("clipboard denied");
    });
    vi.stubGlobal("navigator", { clipboard: { writeText: copy } });
    vi.stubGlobal("isSecureContext", true);
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId={null} storageScope={`guardrail-copy-${success}`}>
        <SwitchLanguage />
        <Toaster expand />
        <GuardrailTestPanel
          guardrailNames={["Home"]}
          onSubmit={vi.fn()}
          isLoading={false}
          results={null}
          errors={null}
          onClose={vi.fn()}
        />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Home" } });
    if (!success) {
      vi.spyOn(console, "error").mockImplementation(() => undefined);
    }
    fireEvent.click(screen.getByRole("button", { name: "复制输入" }));
    expect(copy).toHaveBeenCalledWith("Home");
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    await act(async () => pendingCopy.resolve());
    expect(await screen.findByText(english)).toBeVisible();
  });

  it("策略确认整句翻译，但模板和规则定义原样提交", () => {
    const onConfirm = vi.fn();
    const definition = { guardrail_name: "Home", litellm_params: { guardrail: "presidio", mode: "pre_call" } };
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="dynamic" storageScope="guardrail-selection-count">
        <GuardrailSelectionModal
          visible
          template={{ title: "Settings", guardrailDefinitions: [definition] }}
          existingGuardrails={new Set()}
          onConfirm={onConfirm}
          onCancel={vi.fn()}
          progressInfo={{ current: 2, total: 5 }}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("模板 2 / 5")).toBeInTheDocument();
    expect(screen.getByText("将创建 1 条防护规则")).toBeInTheDocument();
    expect(screen.getByText("暂无描述")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "创建 1 条防护规则并使用模板" }));
    expect(onConfirm).toHaveBeenCalledWith([definition]);
  });

  it("搜索结果整句翻译并切换回英文，文件名与结果内容不翻译", () => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="dynamic" storageScope="search-result-count">
        <SwitchLanguage />
        <SearchResultsDisplay
          searchResults={[
            {
              object: "list",
              search_query: "Settings",
              data: [
                { score: 0.5, filename: "Home", content: [{ text: "Language", type: "text" }] },
                { score: 0.4, content: [] },
              ],
            },
          ]}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("2 条结果")).toBeInTheDocument();
    expect(screen.getByText("结果 2")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "隐藏来源" }));
    expect(screen.getByRole("button", { name: "显示来源（2）" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("button", { name: "Show sources (2)" })).toBeInTheDocument();
  });
});
