import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TRANSPORT_ITEMS } from "@/components/mcp_tools/types";
import { STDIO_DISABLED_MESSAGE, StdioDisabledBanner, TransportSelectItems } from "./StdioAvailability";

describe("stdio 禁用说明的语言", () => {
  beforeEach(() => localStorage.clear());

  it("警告与禁用选项的帮助说明随语言切换，传输类型和禁用状态不变", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="stdio-review" storageScope="/stdio-review">
        <LanguageSelector />
        <StdioDisabledBanner />
        <Select items={TRANSPORT_ITEMS} value={null} defaultOpen>
          <SelectTrigger>
            <SelectValue placeholder="Transport" />
          </SelectTrigger>
          <SelectContent>
            <TransportSelectItems stdioEnabled={false} />
          </SelectContent>
        </Select>
      </LanguageProvider>,
    );

    // 独立断言完整中文句子，同时保护配置变量名称不被翻译。
    const message = "此代理已禁用 stdio MCP 服务器。请在代理上设置 LITELLM_ENABLE_MCP_STDIO=true 并重启以启用。";
    expect(screen.getByText("此代理已禁用 stdio")).toBeInTheDocument();
    expect(screen.getByText(message, { exact: false })).toHaveTextContent(
      `${message}在此之前，此服务器无法以 stdio 方式启动或保存。`,
    );
    expect(screen.getByRole("option", { name: /Standard Input\/Output \(stdio\)/ })).toHaveAttribute("data-disabled");

    await user.hover(screen.getByLabelText("帮助"));
    expect(await screen.findByText(message)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });

    expect(screen.getByText("stdio is disabled on this proxy")).toBeInTheDocument();
    expect(screen.getByText(STDIO_DISABLED_MESSAGE)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Standard Input\/Output \(stdio\)/ })).toHaveAttribute("data-disabled");
    expect(screen.getByRole("option", { name: /Streamable HTTP/ })).not.toHaveAttribute("data-disabled");
  });
});
