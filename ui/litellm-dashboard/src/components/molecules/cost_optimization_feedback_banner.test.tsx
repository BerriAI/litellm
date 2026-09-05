import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import CostOptimizationFeedbackBanner from "./cost_optimization_feedback_banner";

const STORAGE_KEY = "hideCostOptimizationFeedbackBanner";

describe("CostOptimizationFeedbackBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
  });

  it("renders with a link to the feedback discussion", () => {
    render(<CostOptimizationFeedbackBanner />);
    const link = screen.getByText("Share Feedback").closest("a");
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32172");
  });

  it("hides itself and persists the dismissal when the dismiss button is clicked", () => {
    render(<CostOptimizationFeedbackBanner />);
    expect(screen.getByText("Help shape cost optimization")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Dismiss banner"));

    expect(screen.queryByText("Help shape cost optimization")).not.toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("stays dismissed on remount once persisted", () => {
    localStorage.setItem(STORAGE_KEY, "true");
    render(<CostOptimizationFeedbackBanner />);
    expect(screen.queryByText("Help shape cost optimization")).not.toBeInTheDocument();
  });
});
