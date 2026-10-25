import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";
import MoyaiChatHome from "@/components/chat/moyai/MoyaiChatHome";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import type { ConnectFlowStatus } from "@/components/networking";

vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "https://gateway.example.com" }));
vi.mock("@/components/chat/MCPAppsPanel", () => ({
  OAuth2ConnectButton: ({ buttonLabel }: { buttonLabel?: string }) => <button>{buttonLabel}</button>,
}));

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>Switch to English</button>;
}

// 验证界面文案切换不会修改会话标题、示例提示词或授权 POST 参数。
describe("聊天界面语言", () => {
  beforeEach(() => localStorage.clear());

  it("翻译主页入口但保留会话标题和发送的提示词", async () => {
    const onPrompt = vi.fn();
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="chat-test" storageScope="chat-test">
        <MoyaiChatHome
          imageMode={false}
          draft=""
          busy={false}
          conversations={[
            {
              id: "one",
              title: "Settings",
              model: "Home",
              messages: [],
              mcpServerNames: [],
              createdAt: 1,
              updatedAt: 1,
            },
          ]}
          onSelectConversation={vi.fn()}
          onPrompt={onPrompt}
        >
          <input aria-label="draft" />
        </MoyaiChatHome>
        <SwitchLanguage />
      </LanguageProvider>,
    );
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: /审查代码/ }));
    expect(onPrompt).toHaveBeenCalledWith("Review my changes for bugs and improvements.");
    await user.click(screen.getByRole("button", { name: "Switch to English" }));
    expect(screen.getByRole("button", { name: /Review code/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
  });

  it("整句翻译多服务器授权，保留客户端来源、资源名和授权表单", () => {
    const flow: ConnectFlowStatus = {
      state: "multi",
      client_origin: "https://client.example.com",
      server_id: null,
      server_name: null,
      connected: null,
      servers: [{ server_id: "server-id", server_name: "Settings", connected: false }],
    };
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="consent-test" storageScope="chat-test">
        <ConnectFlowBanner
          flowHandle="opaque-handle"
          flow={flow}
          accessToken="test"
          onConnected={vi.fn()}
          failed={false}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("允许 https://client.example.com 使用 Settings")).toBeVisible();
    expect(screen.getByRole("button", { name: "连接 Settings" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "完成连接" })).not.toBeInTheDocument();
    const cancel = screen.getByRole<HTMLButtonElement>("button", { name: "取消" });
    const form = cancel.form;
    expect(form).toHaveAttribute("method", "POST");
    expect(form).toHaveAttribute("action", "https://gateway.example.com/authorize/complete");
    expect(screen.getByDisplayValue("opaque-handle")).toHaveAttribute("name", "flow");
    expect(cancel).toHaveAttribute("value", "deny");
  });

  it("图片模式的固定标题显示中文", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="image-test" storageScope="chat-test">
        <MoyaiChatHome
          imageMode
          draft=""
          busy={false}
          conversations={[]}
          onSelectConversation={vi.fn()}
          onPrompt={vi.fn()}
        >
          <input aria-label="draft" />
        </MoyaiChatHome>
      </LanguageProvider>,
    );
    expect(screen.getByRole("heading", { name: "我们要创作什么？" })).toBeVisible();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Home" } });
    expect(screen.getByRole("textbox")).toHaveValue("Home");
  });
});
