import { act, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
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

  it("should pass a starter prompt to the composer without submitting it", () => {
    const props = homeProps();
    render(<MoyaiChatHome {...props} />);
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /Review code/ }));
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
