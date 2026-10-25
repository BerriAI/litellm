import { fireEvent, render, screen } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { CredentialItem } from "@/components/networking";

import ModelInfoEditForm, { perMillionToPerToken } from "./ModelInfoEditForm";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { translate, type Language } from "@/i18n/language";

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  };
});

const credentialsList: CredentialItem[] = [
  {
    credential_name: "openai-main",
    display_name: "Main OpenAI",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
];

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

const renderForm = ({
  onSubmit = vi.fn().mockResolvedValue(undefined),
  isEditing = true,
  litellmParams = { model: "gpt-4o" } as Record<string, unknown>,
  language = "en" as Language,
  modelName = "gpt-4o",
  modelInfo = {} as Record<string, unknown>,
  withLanguageSwitch = false,
} = {}) =>
  render(
    <LanguageProvider defaultLanguage={language} userId="model-info" storageScope="model-info-language">
      {withLanguageSwitch && <SwitchLanguage />}
      <ModelInfoEditForm
        localModelData={{
          model_name: modelName,
          litellm_params: litellmParams,
          model_info: modelInfo,
        }}
        modelData={{ model_info: {} }}
        teamAlias={null}
        accessToken="test-token"
        isEditing={isEditing}
        isSaving={false}
        isWildcardModel={false}
        ptuCostAttributionEnabled={false}
        showCacheControl={false}
        setShowCacheControl={vi.fn()}
        onCancel={vi.fn()}
        onSubmit={onSubmit}
        modelAccessGroups={[]}
        guardrailsList={[]}
        tagsList={{}}
        credentialsList={credentialsList}
        healthCheckModelOptions={[]}
        teams={[]}
      />
    </LanguageProvider>,
  );

describe("perMillionToPerToken", () => {
  it.each([
    ["2.5", 0.0000025],
    [3, 0.000003],
    ["0", 0],
    ["", null],
    ["  ", null],
    [null, null],
    [undefined, null],
  ])("parses %j into %j per token", (perMillion, perToken) => {
    expect(perMillionToPerToken.parse(perMillion)).toBe(perToken);
  });

  it.each(["abc", "-1", -0.5, "Infinity", Number.NaN])("rejects %j", (perMillion) => {
    expect(perMillionToPerToken.safeParse(perMillion).success).toBe(false);
  });
});

describe("ModelInfoEditForm pricing fields", () => {
  it("只读模型详情翻译空值，不翻译模型名称", () => {
    const formOptions = {
      isEditing: false,
      language: "zh-CN" as const,
      modelName: "Home",
      modelInfo: { access_groups: [] },
    };
    renderForm(formOptions);
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getAllByText("未设置").length).toBeGreaterThan(0);
    expect(screen.queryAllByText("Not Set")).toHaveLength(0);
    expect(screen.getByText("模型名称")).toBeInTheDocument();
    expect(screen.getByText("输入费用（每百万 Token）")).toBeInTheDocument();
    expect(screen.getByText("未分配分组")).toBeInTheDocument();
  });

  it("编辑标签和提示随语言切换，保留模型草稿且不提交", () => {
    const onSubmit = vi.fn();
    const formOptions = { language: "zh-CN" as const, modelName: "Home", withLanguageSwitch: true, onSubmit };
    renderForm(formOptions);
    const modelInput = screen.getByPlaceholderText(translate("zh-CN", "Enter model name"));
    expect(modelInput).not.toHaveAttribute("placeholder", "Enter model name");
    fireEvent.change(modelInput, { target: { value: "Home draft" } });
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByPlaceholderText("Enter model name")).toHaveValue("Home draft");
    expect(screen.getByText("Model Name")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  const storedPrices = {
    model: "gpt-4o",
    input_cost_per_token: 0.000003,
    output_cost_per_token: 0.000015,
    cache_read_input_token_cost: 0.0000003,
  };

  it("hands the submit handler per-token prices, with null for a cleared field", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit, litellmParams: storedPrices });

    expect(screen.getByPlaceholderText("Enter input cost")).toHaveValue(3);
    fireEvent.change(screen.getByPlaceholderText("Enter input cost"), { target: { value: "2.5" } });
    fireEvent.change(screen.getByPlaceholderText("Enter output cost"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const [values, isFieldTouched] = onSubmit.mock.calls[0];
    const expectedPrices = {
      input_cost: 0.0000025,
      output_cost: null,
      cache_read_cost: 0.0000003,
      cache_write_cost: null,
    };
    expect(values).toMatchObject(expectedPrices);
    expect(isFieldTouched("input_cost")).toBe(true);
    expect(isFieldTouched("cache_read_cost")).toBe(false);
  });

  it("shows a field error and does not submit a negative price", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit, litellmParams: storedPrices });

    fireEvent.change(screen.getByPlaceholderText("Enter output cost"), { target: { value: "-1" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(await screen.findByText("Enter a price of 0 or more")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter output cost")).toHaveAttribute("aria-invalid", "true");
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe("ModelInfoEditForm existing-credentials picker", () => {
  it("shows the display name beside the credential name, searches by it, and submits the name", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit });

    const picker = await screen.findByPlaceholderText("Select or search for existing credentials");
    await user.click(picker);

    const option = await screen.findByRole("option", { name: /openai-main/ });
    expect(option).toHaveTextContent("openai-main");
    expect(option).toHaveTextContent("Main OpenAI");

    await user.type(picker, "main openai");
    await user.click(await screen.findByRole("option", { name: /openai-main/ }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0].litellm_credential_name).toBe("openai-main");
  });

  it("keeps the attached credential when the search text is emptied and only clears through None", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit, litellmParams: { model: "gpt-4o", litellm_credential_name: "openai-main" } });

    const picker = await screen.findByPlaceholderText("Select or search for existing credentials");
    expect(picker).toHaveValue("Main OpenAI");
    await user.clear(picker);
    await user.tab();
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].litellm_credential_name).toBe("openai-main");

    await user.click(picker);
    await user.click(await screen.findByRole("option", { name: "None" }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(2));
    expect(onSubmit.mock.calls[1][0].litellm_credential_name).toBeNull();
  });

  it("shows the attached credential's display name with its credential name beneath when not editing", () => {
    renderForm({ isEditing: false, litellmParams: { model: "gpt-4o", litellm_credential_name: "openai-main" } });

    expect(screen.getByText("Main OpenAI")).toBeInTheDocument();
    expect(screen.getByText("openai-main")).toBeInTheDocument();
  });

  it("shows the selected credential's name under the picker only when its label differs", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ litellmParams: { model: "gpt-4o", litellm_credential_name: "openai-main" } });

    const picker = await screen.findByPlaceholderText("Select or search for existing credentials");
    expect(picker).toHaveValue("Main OpenAI");
    expect(screen.getByText("openai-main")).toBeInTheDocument();

    await user.click(picker);
    await user.click(await screen.findByRole("option", { name: "None" }));
    expect(screen.queryByText("openai-main")).not.toBeInTheDocument();
  });

  it("falls back to the credential name, then Manual, when not editing", () => {
    const { unmount } = renderForm({
      isEditing: false,
      litellmParams: { model: "gpt-4o", litellm_credential_name: "not-listed" },
    });
    expect(screen.getAllByText("not-listed")).toHaveLength(1);
    unmount();

    renderForm({ isEditing: false });
    expect(screen.getByText("Manual")).toBeInTheDocument();
  });
});
