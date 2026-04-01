import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { SurveyModal } from "./SurveyModal";

describe("SurveyModal", () => {
  beforeEach(() => {
    vi.spyOn(global, "fetch").mockResolvedValue(new Response());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should render nothing when isOpen is false", () => {
    renderWithProviders(
      <SurveyModal isOpen={false} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(
      screen.queryByText(/Are you using LiteLLM at your company\?/i)
    ).not.toBeInTheDocument();
  });

  it("should render step 1 when the modal is opened", () => {
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(
      screen.getByText(/Are you using LiteLLM at your company\?/i)
    ).toBeInTheDocument();
  });

  it("should disable the Next button until a step 1 choice is made", () => {
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: /Next/i })).toBeDisabled();
  });

  it("should enable the Next button after selecting Yes", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /We use it for work/i }));

    expect(screen.getByRole("button", { name: /Next/i })).not.toBeDisabled();
  });

  it("should navigate to the company name step when Yes is selected and Next is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /We use it for work/i }));
    await user.click(screen.getByRole("button", { name: /Next/i }));

    expect(
      screen.getByText(/What company are you using LiteLLM at\?/i)
    ).toBeInTheDocument();
  });

  it("should skip the company name step when No is selected and go straight to step 3", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /Personal project/i }));
    await user.click(screen.getByRole("button", { name: /Next/i }));

    expect(screen.getByText(/When did you start using LiteLLM\?/i)).toBeInTheDocument();
  });

  it("should show 5 total steps when using at a company", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /We use it for work/i }));

    expect(screen.getByText(/Step 1 of 5/i)).toBeInTheDocument();
  });

  it("should show 4 total steps when not using at a company", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /Personal project/i }));

    expect(screen.getByText(/Step 1 of 4/i)).toBeInTheDocument();
  });

  it("should navigate back to step 1 from step 3 when No was previously selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /Personal project/i }));
    await user.click(screen.getByRole("button", { name: /Next/i }));
    await user.click(screen.getByRole("button", { name: /Back/i }));

    expect(
      screen.getByText(/Are you using LiteLLM at your company\?/i)
    ).toBeInTheDocument();
  });

  describe("when step 4 (reasons) is reached", () => {
    async function navigateToStep4(user: ReturnType<typeof userEvent.setup>) {
      // No path: step 1 → 3 → 4
      await user.click(screen.getByRole("button", { name: /Personal project/i }));
      await user.click(screen.getByRole("button", { name: /Next/i }));
      await user.click(screen.getByRole("radio", { name: /Less than a month ago/i }));
      await user.click(screen.getByRole("button", { name: /Next/i }));
    }

    it("should show a text input when the Other reason is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
      );

      await navigateToStep4(user);
      await user.click(screen.getByRole("button", { name: /Something else not listed above/i }));

      expect(screen.getByPlaceholderText(/Please specify/i)).toBeInTheDocument();
    });

    it("should keep the Next button disabled when Other is selected but the text field is empty", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
      );

      await navigateToStep4(user);
      await user.click(screen.getByRole("button", { name: /Something else not listed above/i }));

      expect(screen.getByRole("button", { name: /Next/i })).toBeDisabled();
    });

    it("should enable Next when a standard reason is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={vi.fn()} />
      );

      await navigateToStep4(user);
      await user.click(
        screen.getByRole("button", { name: /Stars, contributors, forks, community support/i })
      );

      expect(screen.getByRole("button", { name: /Next/i })).not.toBeDisabled();
    });
  });

  it("should call onComplete after successfully submitting the form", async () => {
    const onComplete = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={vi.fn()} onComplete={onComplete} />
    );

    // Navigate through the No path: step 1 → 3 → 4 → 5 → submit
    await user.click(screen.getByRole("button", { name: /Personal project/i }));
    await user.click(screen.getByRole("button", { name: /Next/i }));
    await user.click(screen.getByRole("radio", { name: /Less than a month ago/i }));
    await user.click(screen.getByRole("button", { name: /Next/i }));
    await user.click(
      screen.getByRole("button", { name: /Stars, contributors, forks, community support/i })
    );
    await user.click(screen.getByRole("button", { name: /Next/i }));
    // Step 5: email is optional
    await user.click(screen.getByRole("button", { name: /Submit/i }));

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });

  it("should call onClose when the close button is clicked", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <SurveyModal isOpen={true} onClose={onClose} onComplete={vi.fn()} />
    );

    // X close button is the first button in the modal header
    const buttons = screen.getAllByRole("button");
    await user.click(buttons[0]);

    expect(onClose).toHaveBeenCalled();
  });
});
