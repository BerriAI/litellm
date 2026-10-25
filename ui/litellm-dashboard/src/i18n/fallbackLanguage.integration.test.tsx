import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import AgentVirtualKeys from "@/app/(dashboard)/agents/_components/agent_virtual_keys";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("缺省文案的语言切换", () => {
  it("翻译未命名密钥的兜底，不翻译与词典同名的真实别名", () => {
    // 两种密钥共用真实组件，防止把后台返回的别名当作词典键
    const keys = [
      { token: "preview-empty", key_alias: "", key_name: "empty-key" },
      { token: "preview-home", key_alias: "Home", key_name: "home-key" },
    ] as KeyResponse[];
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="fallback-keys" storageScope="fallback-language">
        <LanguageSwitch />
        <AgentVirtualKeys keys={keys} isLoading={false} onKeyClick={vi.fn()} />
      </LanguageProvider>,
    );
    expect(screen.getByText("未命名密钥")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("Unnamed key")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
  });
});
