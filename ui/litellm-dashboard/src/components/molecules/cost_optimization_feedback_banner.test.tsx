import { fireEvent, render } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import CostOptimizationFeedbackBanner from "./cost_optimization_feedback_banner";

const STORAGE_KEY = "hideCostOptimizationFeedbackBanner";

describe("CostOptimizationFeedbackBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
  });

  it("renders with a link to the feedback discussion", () => {
    const { getByText } = render(<CostOptimizationFeedbackBanner />);
    const link = getByText("Share Feedback").closest("a");
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32172");
  });

  it("hides itself and persists the dismissal when the dismiss button is clicked", () => {
    const { queryByText, getByLabelText } = render(<CostOptimizationFeedbackBanner />);
    expect(queryByText("Help shape cost optimization")).toBeInTheDocument();

    fireEvent.click(getByLabelText("Dismiss banner"));

    expect(queryByText("Help shape cost optimization")).not.toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("stays dismissed on remount once persisted", () => {
    localStorage.setItem(STORAGE_KEY, "true");
    const { queryByText } = render(<CostOptimizationFeedbackBanner />);
    expect(queryByText("Help shape cost optimization")).not.toBeInTheDocument();
  });
});
