import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { ClaudeCodeModal } from "./ClaudeCodeModal";

describe("ClaudeCodeModal", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should render nothing when isOpen is false", () => {
    renderWithProviders(
      <ClaudeCodeModal isOpen={false} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(screen.queryByText(/Help us improve your experience/i)).not.toBeInTheDocument();
  });

  it("should render the feedback modal content when isOpen is true", () => {
    renderWithProviders(
      <ClaudeCodeModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(screen.getByText(/Help us improve your experience/i)).toBeInTheDocument();
  });

  it("should show the survey description text", () => {
    renderWithProviders(
      <ClaudeCodeModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(screen.getByText(/your experience using LiteLLM with Claude Code/i)).toBeInTheDocument();
  });

  it("should open the Google Form and call onComplete when the feedback button is clicked", async () => {
    const onComplete = vi.fn();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const user = userEvent.setup();

    renderWithProviders(
      <ClaudeCodeModal isOpen={true} onClose={vi.fn()} onComplete={onComplete} />
    );

    await user.click(screen.getByRole("button", { name: /Open Feedback Form/i }));

    expect(openSpy).toHaveBeenCalledWith(
      "https://forms.gle/LZeJQ3XytBakckYa9",
      "_blank",
      "noopener,noreferrer"
    );
    expect(onComplete).toHaveBeenCalled();
  });

  it("should call onClose when the close button is clicked", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <ClaudeCodeModal isOpen={true} onClose={onClose} onComplete={vi.fn()} />
    );

    // The X close button is the first button; the "Open Feedback Form" button is the second
    const buttons = screen.getAllByRole("button");
    await user.click(buttons[0]);

    expect(onClose).toHaveBeenCalled();
  });
});
