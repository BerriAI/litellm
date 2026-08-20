import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { UiLoadingSpinner } from "./ui-loading-spinner";

describe("UiLoadingSpinner", () => {
  it("should render an SVG element", () => {
    render(<UiLoadingSpinner data-testid="spinner" />);
    expect(screen.getByTestId("spinner")).toBeInTheDocument();
  });

  it("should apply custom className alongside default classes", () => {
    render(<UiLoadingSpinner data-testid="spinner" className="text-destructive" />);
    const svg = screen.getByTestId("spinner");
    expect(svg).toHaveClass("text-destructive");
    expect(svg).toHaveClass("animate-spin");
  });

  it("should spread additional SVG props onto the element", () => {
    render(<UiLoadingSpinner data-testid="spinner" aria-label="Loading" />);
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  });
});
