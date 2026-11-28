import { act, fireEvent, render, screen } from "@testing-library/react";
import i18next from "i18next";
import type { ComponentProps } from "react";
import { I18nextProvider } from "react-i18next";
import { describe, expect, it, vi } from "vitest";
import en from "@/i18n/locales/en.json";
import zh from "@/i18n/locales/zh-CN.json";
import MoyaiChatHome from "./MoyaiChatHome";

function homeProps(overrides: Partial<ComponentProps<typeof MoyaiChatHome>> = {}) {
  return {
    children: <span>Chat composer</span>,
    imageMode: false,
    researchMode: false,
    draft: "",
    busy: false,
    onPrompt: vi.fn(),
    ...overrides,
  };
}

describe("MoyaiChatHome", () => {
  it("should render the welcome and composer", () => {
    render(<MoyaiChatHome {...homeProps()} />);
    expect(screen.getByRole("heading", { level: 1, name: "What are we working on?" })).toBeInTheDocument();
    expect(screen.getByText("Chat composer")).toBeInTheDocument();
  });

  it("should render the research welcome when research mode is enabled", () => {
    render(<MoyaiChatHome {...homeProps({ researchMode: true })} />);
    expect(screen.getByRole("heading", { level: 1, name: "What should we research?" })).toBeInTheDocument();
  });

  it("should fill the composer with the original prompt from a translated starter", async () => {
    const i18n = i18next.createInstance();
    await i18n.init({ lng: "zh-CN", resources: { en: { translation: en }, "zh-CN": { translation: zh } } });
    const props = homeProps();
    render(
      <I18nextProvider i18n={i18n}>
        <MoyaiChatHome {...props} />
      </I18nextProvider>,
    );
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: zh.chat.moyaiHome.starters.review.label }));
    });
    expect(props.onPrompt).toHaveBeenCalledWith("Review my changes for bugs and improvements.");
  });

  it("should disable starters during a request", () => {
    const props = homeProps({ busy: true });
    render(<MoyaiChatHome {...props} />);
    const starter = screen.getByRole("button", { name: /Review code/ });
    expect(starter).toBeDisabled();
    act(() => {
      fireEvent.click(starter);
    });
    expect(props.onPrompt).not.toHaveBeenCalled();
  });

  it("should hide starter choices from assistive technology once a draft is present", () => {
    render(<MoyaiChatHome {...homeProps({ draft: "My current prompt" })} />);
    expect(screen.queryByRole("group", { name: "Start with a prompt" })).not.toBeInTheDocument();
    for (const button of screen.getAllByRole("button", { hidden: true })) expect(button).toBeDisabled();
  });

  it("should show the image welcome without code starter prompts in image mode", () => {
    render(<MoyaiChatHome {...homeProps({ imageMode: true })} />);
    expect(screen.getByRole("heading", { level: 1, name: "What are we creating?" })).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Start with a prompt" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Review code/ })).not.toBeInTheDocument();
  });
});
