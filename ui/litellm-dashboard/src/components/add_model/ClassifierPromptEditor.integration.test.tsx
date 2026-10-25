import { act, fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi, type Mock } from "vitest";
import ClassifierPromptEditor from "./ClassifierPromptEditor";
import { ClassificationRubric } from "./ComplexityRouterConfig";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

const getDefaultPrompt = vi.hoisted(() => vi.fn());
vi.mock("@/components/networking", () => ({
  getAutoRouterClassifierDefaultPromptCall: getDefaultPrompt,
}));

const DEFAULT_PROMPT = "Classify the complexity of a user request into exactly one tier. Tiers: SIMPLE ...";

beforeEach(() => {
  getDefaultPrompt.mockReset();
  getDefaultPrompt.mockResolvedValue(DEFAULT_PROMPT);
});

interface OpenEditorOptions {
  systemPrompt?: string;
  onChange?: Mock;
  contextWindowSize?: number;
  tierLabels?: Record<string, string>;
  classificationRubric?: ClassificationRubric;
}

const openEditor = async ({
  systemPrompt,
  onChange = vi.fn(),
  contextWindowSize = 3,
  tierLabels,
  classificationRubric = "agentic",
}: OpenEditorOptions = {}) => {
  renderWithProviders(
    <ClassifierPromptEditor
      systemPrompt={systemPrompt}
      onChange={onChange}
      contextWindowSize={contextWindowSize}
      tierLabels={tierLabels}
      classificationRubric={classificationRubric}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: /prompt/i }));
  expect(await screen.findByLabelText("Classifier system prompt")).toBeInTheDocument();
  return onChange;
};

describe("ClassifierPromptEditor", () => {
  it("prefills the live rubric fetched for the configured context window", async () => {
    await openEditor({ contextWindowSize: 7 });
    // Prefilling from the backend rather than a frontend copy is the whole point: a copy would
    // drift the moment the rubric is edited.
    expect(getDefaultPrompt).toHaveBeenCalledWith("sk-test", 7, undefined, "agentic");
    expect(screen.getByLabelText("Classifier system prompt")).toHaveValue(DEFAULT_PROMPT);
  });

  it("prefills the preset the router is on, not always the default one", async () => {
    // The editor is how an operator inspects the rubric before replacing it. Prefilling the agentic
    // text for a router on chat would show them examples their classifier never receives.
    await openEditor({ contextWindowSize: 7, classificationRubric: "chat" });
    expect(getDefaultPrompt).toHaveBeenCalledWith("sk-test", 7, undefined, "chat");
  });

  it("prefills the rubric named by the operator's renamed tiers", async () => {
    // A renamed router sends a rubric using its own labels, and its classifier must return them,
    // so prefilling the canonical names would hand back a prompt that router rejects.
    const tierLabels = { SIMPLE: "Cheap", REASONING: "Deep" };
    await openEditor({ contextWindowSize: 7, tierLabels });
    expect(getDefaultPrompt).toHaveBeenCalledWith("sk-test", 7, tierLabels, "agentic");
  });

  it("warns that the prompt replaces the injection-defense text", async () => {
    await openEditor();
    expect(screen.getByText("Proceed with caution")).toBeInTheDocument();
    expect(screen.getByText(/entire system role/)).toBeInTheDocument();
  });

  it("warns that this mode freezes the tier definitions into the operator's text", async () => {
    // The whole point of the derived prompt is that a tier rename reaches the classifier. An
    // operator staying on this editor has to be told their text will not follow one.
    await openEditor({ systemPrompt: "Grade data sensitivity" });
    expect(screen.getByText(/legacy whole-prompt mode/)).toBeInTheDocument();
    expect(screen.getByText(/renaming a tier or changing the rubric will not update it/)).toBeInTheDocument();
  });

  it("saves an edited prompt as an override", async () => {
    const onChange = await openEditor();
    const textarea = screen.getByLabelText("Classifier system prompt");
    await userEvent.clear(textarea);
    fireEvent.change(textarea, { target: { value: "Grade data sensitivity" } });
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith("Grade data sensitivity");
  });

  it("saves an untouched prompt as no override at all", async () => {
    const onChange = await openEditor();
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("offers a reset that clears a stored override", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ClassifierPromptEditor
        systemPrompt="Grade data sensitivity"
        onChange={onChange}
        contextWindowSize={3}
        classificationRubric="agentic"
      />,
    );
    expect(screen.getByRole("button", { name: "Edit custom prompt" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reset to default" }));
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("seeds the editor from the stored override, not the default", async () => {
    await openEditor({ systemPrompt: "Grade data sensitivity" });
    expect(screen.getByLabelText("Classifier system prompt")).toHaveValue("Grade data sensitivity");
  });
});

describe("分类器提示词加载的语言边界", () => {
  beforeEach(async () => {
    localStorage.clear();
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    vi.mocked(toast.fromError).mockImplementation(actual.toast.fromError);
  });

  afterEach(async () => {
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    act(() => actual.toast.dismiss());
    vi.mocked(toast.fromError).mockReset();
  });

  const tree = (defaultLanguage: string, onChange: Mock) => (
    <LanguageProvider defaultLanguage={defaultLanguage} userId="classifier-review" storageScope="/classifier">
      <LanguageSelector />
      <ClassifierPromptEditor
        systemPrompt="Home"
        onChange={onChange}
        contextWindowSize={7}
        tierLabels={{ SIMPLE: "Home" }}
        classificationRubric="agentic"
      />
      <Toaster />
    </LanguageProvider>
  );

  it("加载期间全站默认语言变化后，失败通知使用当前语言，且不重新加载或提交", async () => {
    const pending = Promise.withResolvers<string>();
    getDefaultPrompt.mockReturnValue(pending.promise);
    const onChange = vi.fn();
    const view = renderWithProviders(tree("en", onChange));
    await userEvent.click(screen.getByRole("button", { name: "Edit custom prompt" }));

    // 真实模态框会阻止页外点击，此处模拟全站默认设置更新，不绕过模态焦点约束。
    view.rerender(tree("zh-CN", onChange));
    await act(async () => pending.reject(new Error("Home")));
    expect(await screen.findByText("默认分类器提示词加载失败，请重试。")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByText("Could not load the default classifier prompt")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(getDefaultPrompt).toHaveBeenCalledExactlyOnceWith("sk-test", 7, { SIMPLE: "Home" }, "agentic");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("已显示的本地失败提示随个人语言切换", async () => {
    getDefaultPrompt.mockRejectedValue(new Error("Home"));
    const onChange = vi.fn();
    renderWithProviders(tree("en", onChange));
    await userEvent.click(screen.getByRole("button", { name: "Edit custom prompt" }));
    expect(await screen.findByText("Could not load the default classifier prompt")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText("默认分类器提示词加载失败，请重试。")).toBeInTheDocument();
    expect(getDefaultPrompt).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("加载和草稿编辑时切换语言不翻译提示词、层级名称，也不再次请求或保存", async () => {
    const pending = Promise.withResolvers<string>();
    getDefaultPrompt.mockReturnValue(pending.promise);
    const onChange = vi.fn();
    const view = renderWithProviders(tree("en", onChange));
    await userEvent.click(screen.getByRole("button", { name: "Edit custom prompt" }));
    view.rerender(tree("zh-CN", onChange));
    await act(async () => pending.resolve("Home"));

    const prompt = screen.getByRole("textbox");
    expect(prompt).toHaveValue("Home");
    fireEvent.change(prompt, { target: { value: "Home draft" } });
    view.rerender(tree("en", onChange));
    expect(screen.getByRole("textbox", { name: "Classifier system prompt" })).toHaveValue("Home draft");
    expect(onChange).not.toHaveBeenCalled();
    expect(getDefaultPrompt).toHaveBeenCalledExactlyOnceWith("sk-test", 7, { SIMPLE: "Home" }, "agentic");
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith("Home draft");
  });
});
