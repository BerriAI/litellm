import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { MessageScrollerButton, MessageScrollerProvider } from "./message-scroller";

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("消息滚动按钮语言", () => {
  it.each([
    ["end", "滚动到底部", "Scroll to end"],
    ["start", "滚动到顶部", "Scroll to start"],
  ] as const)("%s 方向的辅助名称可从中文切回英文", (direction, chineseLabel, englishLabel) => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId={direction} storageScope="scroll-label-test">
        <LanguageSwitch />
        <MessageScrollerProvider>
          <MessageScrollerButton direction={direction} />
        </MessageScrollerProvider>
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: chineseLabel })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "English" }));

    expect(screen.getByRole("button", { name: englishLabel })).toBeInTheDocument();
  });

  it("调用方提供的按钮内容和辅助名称保持原样", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="custom" storageScope="scroll-label-test">
        <MessageScrollerProvider>
          <MessageScrollerButton aria-label="Home">Scroll to end</MessageScrollerButton>
        </MessageScrollerProvider>
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: "Home" })).toHaveTextContent("Scroll to end");
  });
});
