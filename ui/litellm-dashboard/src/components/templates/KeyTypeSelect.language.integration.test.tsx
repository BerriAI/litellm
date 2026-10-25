import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { ModelTeamSelect } from "@/components/view_model/ModelTeamSelect";
import type { Team } from "@/components/key_team_helpers/key_list";
import { KeyTypeSelect } from "./KeyEditViewControls";

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("模型和密钥表单语言", () => {
  it("切换语言更新密钥类型文案，提交仍使用原始接口值", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="key-type-language" storageScope="models-language">
        <SwitchLanguage />
        <KeyTypeSelect id="key-type" value="default" onChange={onChange} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent("完整访问权限");
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: /AI 接口 仅可调用 AI 接口路由/ }));
    expect(onChange).toHaveBeenCalledWith("llm_api");
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("combobox")).toHaveTextContent("Full Access");
  });

  it("中文占位符不改变与翻译键同名的团队名称或 ID", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const teams = [{ team_id: "team-original-id", team_alias: "Home" }] as Team[];
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="model-team-language" storageScope="models-language">
        <ModelTeamSelect id="model-team" value={undefined} onChange={onChange} onBlur={vi.fn()} teams={teams} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent("选择团队");
    await user.click(screen.getByRole("combobox"));
    const team = await screen.findByRole("option", { name: "Home (team-original-id)" });
    expect(screen.queryByRole("option", { name: /首页/ })).not.toBeInTheDocument();
    await user.click(team);
    expect(onChange).toHaveBeenCalledWith("team-original-id");
  });
});
