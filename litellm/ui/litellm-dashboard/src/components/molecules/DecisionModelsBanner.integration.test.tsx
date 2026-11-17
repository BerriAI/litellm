import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import DecisionModelsBanner from "./DecisionModelsBanner";

const STORAGE_KEY = "hideDecisionModelsBanner";

describe("DecisionModelsBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem("hideCostOptimizationFeedbackBanner");
  });

  it("links directly to the Decisions playground", () => {
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

  it("links Supported providers to the provider table in the docs", () => {
    render(<DecisionModelsBanner />);
    expect(screen.getByRole("link", { name: "Supported providers" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions#supported-providers",
    );
    expect(screen.getByRole("paragraph")).not.toHaveTextContent("Add Model");
    expect(screen.getByRole("paragraph")).toHaveTextContent("Call them at /v1/decisions or /v1/systemone");
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
