import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "@/../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import PromptCompressionTab from "./PromptCompressionTab";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
}));

describe("提示词压缩表单汉化", () => {
  beforeEach(() => localStorage.clear());

  it("汉化固定 API 说明和必填校验", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="compression-test" storageScope="compression-test">
        <PromptCompressionTab accessToken="test-token" />
      </LanguageProvider>,
    );
    expect(screen.getByText("Headroom API 基础地址")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "添加防护规则" }));
    expect(await screen.findByText("请输入名称")).toBeInTheDocument();
    expect(await screen.findByText("请输入 API 基础地址")).toBeInTheDocument();
  });
});
