import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { ClaudeCodePrompt } from "./ClaudeCodePrompt";

vi.mock("./NudgePrompt", () => ({
  NudgePrompt: ({
    title,
    description,
    buttonText,
    onOpen,
    onDismiss,
    isVisible,
  }: {
    title: string;
    description: string;
    buttonText: string;
    onOpen: () => void;
    onDismiss: () => void;
    isVisible: boolean;
  }) => {
    if (!isVisible) return null;
    return (
      <div>
        <span>{title}</span>
        <span>{description}</span>
        <button onClick={onOpen}>{buttonText}</button>
        <button onClick={onDismiss}>Dismiss</button>
      </div>
    );
  },
}));

describe("ClaudeCodePrompt", () => {
  it("should render with the Claude Code Feedback title when visible", () => {
    renderWithProviders(
      <ClaudeCodePrompt isVisible={true} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.getByText("Claude Code Feedback")).toBeInTheDocument();
  });

  it("should render the correct description text", () => {
    renderWithProviders(
      <ClaudeCodePrompt isVisible={true} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.getByText(/Help us improve your Claude Code experience/i)).toBeInTheDocument();
  });

  it("should call onOpen when the share feedback button is clicked", async () => {
    const onOpen = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <ClaudeCodePrompt isVisible={true} onOpen={onOpen} onDismiss={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /Share feedback/i }));

    expect(onOpen).toHaveBeenCalled();
  });

  it("should call onDismiss when the dismiss button is clicked", async () => {
    const onDismiss = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <ClaudeCodePrompt isVisible={true} onOpen={vi.fn()} onDismiss={onDismiss} />
    );

    await user.click(screen.getByRole("button", { name: /Dismiss/i }));

    expect(onDismiss).toHaveBeenCalled();
  });

  it("should not render when isVisible is false", () => {
    renderWithProviders(
      <ClaudeCodePrompt isVisible={false} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.queryByText("Claude Code Feedback")).not.toBeInTheDocument();
  });
});
