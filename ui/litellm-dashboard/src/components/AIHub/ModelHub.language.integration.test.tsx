import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import ModelFilters from "@/components/model_filters";
import { ModelCustomMetadata } from "./ModelCustomMetadata";
import { inheritedGrantTooltip } from "@/components/permissions/inheritedGrants";
import { translate } from "@/i18n/language";
import { DataTable } from "@/components/shared/DataTable";
import { getPublicModelHubColumns, type ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { getModelHubTableColumns, type ModelHubData } from "./ModelHubTableColumns";

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("模型中心语言边界", () => {
  it.each([false, true])("AI Hub 列表保留扩展模式与能力，已知枚举可切换语言，聊天样式 %s", (showProviderLogos) => {
    const defaults: ModelHubData = {
      model_group: "model-home",
      providers: ["openai"],
      supports_parallel_function_calling: false,
      supports_vision: true,
      supports_function_calling: false,
      supports_settings: true,
      is_public_model_group: true,
    };
    const rows = [
      { ...defaults, mode: "Home" },
      { ...defaults, model_group: "model-vision", mode: "Vision" },
      { ...defaults, model_group: "model-chat", mode: "chat" },
      { ...defaults, model_group: "model-decision", mode: "evaluation" },
    ];
    const table = (language: "zh-CN" | "en") => (
      <DataTable
        data={rows}
        columns={getModelHubTableColumns({
          onModelClick: vi.fn(),
          showProviderLogos,
          t: (message, values) => translate(language, message, values),
        })}
        getRowId={(row) => row.model_group}
      />
    );
    const view = render(table("zh-CN"));
    const row = (name: string) => screen.getByRole("row", { name: new RegExp(name) });
    // 扩展模式不能命中其他领域的词条，扩展能力也不能被当作导航标签。
    expect(within(row("model-home")).getByRole("cell", { name: "Home", exact: true })).toBeInTheDocument();
    expect(within(row("model-vision")).getByRole("cell", { name: "Vision", exact: true })).toBeInTheDocument();
    for (const name of rows.map((model) => model.model_group)) {
      expect(within(row(name)).getByText("Settings")).toBeInTheDocument();
      expect(within(row(name)).getByText("视觉")).toBeInTheDocument();
    }
    expect(within(row("model-chat")).getByRole("cell", { name: "对话", exact: true })).toBeInTheDocument();
    expect(within(row("model-decision")).getByRole("cell", { name: "决策", exact: true })).toBeInTheDocument();
    view.rerender(table("en"));
    expect(
      within(row("model-chat")).getByRole("cell", { name: showProviderLogos ? "Chat" : "chat", exact: true }),
    ).toBeInTheDocument();
    expect(
      within(row("model-decision")).getByRole("cell", {
        name: showProviderLogos ? "Decisions" : "decisions",
        exact: true,
      }),
    ).toBeInTheDocument();
    expect(within(row("model-home")).getByText("Settings")).toBeInTheDocument();
  });

  it("切换语言后保留原始筛选值和自定义模型、标签、元数据", async () => {
    const user = userEvent.setup();
    const model = {
      model_group: "Home",
      mode: "chat",
      providers: ["openai"],
      supports_parallel_function_calling: false,
      supports_vision: true,
      supports_function_calling: false,
      is_public_model_group: true,
    };
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="hub-language" storageScope="model-hub-language">
        <SwitchLanguage />
        <ModelFilters modelHubData={[model]} onFilteredDataChange={vi.fn()} />
        <ModelCustomMetadata customTags={["Home"]} customMetadata={{ Language: "Settings" }} />
      </LanguageProvider>,
    );
    const chatOption = screen.getByRole("option", { name: "对话" });
    const modeSelect = chatOption.parentElement as HTMLSelectElement;
    await user.selectOptions(modeSelect, "chat");
    expect(modeSelect).toHaveValue("chat");
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Language")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("自定义元数据：")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(modeSelect).toHaveValue("chat");
    expect(screen.getByRole("option", { name: "chat" })).toBeInTheDocument();
    expect(screen.getByText("Custom Metadata:")).toBeInTheDocument();
  });

  it("权限提示翻译解释但不翻译组名及资源 ID", () => {
    const grant = { id: "mcp-Home", accessGroupNames: ["Home", "Settings"] };
    expect(inheritedGrantTooltip(grant, (message, values) => translate("zh-CN", message, values))).toBe(
      "通过访问组 Home, Settings 授权。完整 ID：mcp-Home",
    );
    expect(inheritedGrantTooltip(grant)).toBe("Granted via access groups Home, Settings. Full ID: mcp-Home");
  });

  it("健康状态翻译固定枚举，保留自定义状态和模型名称", () => {
    const defaults = {
      providers: ["openai"],
      supports_parallel_function_calling: false,
      supports_vision: false,
      supports_function_calling: false,
      is_public_model_group: true,
    };
    const rows: ModelGroupInfo[] = [
      { ...defaults, model_group: "Home", health_status: "healthy" },
      { ...defaults, model_group: "Settings", health_status: "unhealthy" },
      { ...defaults, model_group: "Language", health_status: "Home" },
    ];
    render(
      <DataTable
        data={rows}
        columns={getPublicModelHubColumns({
          onModelClick: vi.fn(),
          t: (message, values) => translate("zh-CN", message, values),
        })}
        getRowId={(row) => row.model_group}
      />,
    );
    expect(screen.getByText(translate("zh-CN", "healthy"))).toBeInTheDocument();
    expect(screen.getByText("异常")).toBeInTheDocument();
    expect(screen.getAllByText("Home")).toHaveLength(2);
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("Language")).toBeInTheDocument();
  });
});
