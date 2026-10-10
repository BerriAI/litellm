import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DecisionModelsBanner from "./DecisionModelsBanner";

const STORAGE_KEY = "hideDecisionModelsBanner";

describe("DecisionModelsBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem("hideCostOptimizationFeedbackBanner");
  });

  it("links directly to the System One playground", () => {
    render(<DecisionModelsBanner />);
    const link = screen.getByRole("link", { name: "Try decision models" });
    expect(link).toHaveAttribute("href", "/ui/playground?tab=system-one");
  });

  it("links to the docs on how to call decision models", () => {
    render(<DecisionModelsBanner />);
    expect(screen.getByRole("link", { name: "How to call them" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions",
    );
  });

  it("offers Add a decision model only when the page can add models, and calls back on click", () => {
    const onAddModel = vi.fn();
    const { rerender } = render(<DecisionModelsBanner />);
    expect(screen.queryByRole("button", { name: "Add a decision model" })).not.toBeInTheDocument();

    rerender(<DecisionModelsBanner onAddModel={onAddModel} />);
    fireEvent.click(screen.getByRole("button", { name: "Add a decision model" }));

    expect(onAddModel).toHaveBeenCalledTimes(1);
  });

  it("points only users who can add models at Add Model", () => {
    const { rerender } = render(<DecisionModelsBanner />);
    expect(screen.getByRole("paragraph")).not.toHaveTextContent("Add Model");
    expect(screen.getByRole("paragraph")).toHaveTextContent("Call them at /v1/decisions or /v1/systemone");

    rerender(<DecisionModelsBanner onAddModel={vi.fn()} />);

    expect(screen.getByRole("paragraph")).toHaveTextContent(
      "Search decision in Add Model to find them. Call them at /v1/decisions",
    );
  });

  it("shows the announcement even if the old feedback banner was dismissed", () => {
    localStorage.setItem("hideCostOptimizationFeedbackBanner", "true");
    render(<DecisionModelsBanner />);
    expect(screen.getByText("Decision models are now supported")).toBeInTheDocument();
  });

  it("hides itself and persists the dismissal when the dismiss button is clicked", () => {
    render(<DecisionModelsBanner />);
    expect(screen.getByText("Decision models are now supported")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Dismiss banner"));

    expect(screen.queryByText("Decision models are now supported")).not.toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("stays dismissed on remount once persisted", () => {
    localStorage.setItem(STORAGE_KEY, "true");
    render(<DecisionModelsBanner />);
    expect(screen.queryByText("Decision models are now supported")).not.toBeInTheDocument();
  });
});
