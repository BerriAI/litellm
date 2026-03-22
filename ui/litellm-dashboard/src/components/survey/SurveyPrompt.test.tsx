import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { SurveyPrompt } from "./SurveyPrompt";

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

describe("SurveyPrompt", () => {
  it("should render with the Quick feedback title when visible", () => {
    renderWithProviders(
      <SurveyPrompt isVisible={true} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.getByText("Quick feedback")).toBeInTheDocument();
  });

  it("should render the correct description text", () => {
    renderWithProviders(
      <SurveyPrompt isVisible={true} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.getByText(/Help us improve LiteLLM/i)).toBeInTheDocument();
  });

  it("should call onOpen when the share feedback button is clicked", async () => {
    const onOpen = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <SurveyPrompt isVisible={true} onOpen={onOpen} onDismiss={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /Share feedback/i }));

    expect(onOpen).toHaveBeenCalled();
  });

  it("should call onDismiss when the dismiss button is clicked", async () => {
    const onDismiss = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <SurveyPrompt isVisible={true} onOpen={vi.fn()} onDismiss={onDismiss} />
    );

    await user.click(screen.getByRole("button", { name: /Dismiss/i }));

    expect(onDismiss).toHaveBeenCalled();
  });

  it("should not render when isVisible is false", () => {
    renderWithProviders(
      <SurveyPrompt isVisible={false} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.queryByText("Quick feedback")).not.toBeInTheDocument();
  });
});
