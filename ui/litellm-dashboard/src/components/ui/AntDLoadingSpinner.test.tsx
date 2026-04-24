import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AntDLoadingSpinner } from "./AntDLoadingSpinner";

describe("AntDLoadingSpinner", () => {
  it("should render a lucide spinner svg", () => {
    const { container } = render(<AntDLoadingSpinner />);
    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(svg).toHaveClass("animate-spin");
  });

  it("should use large size class when size=large", () => {
    const { container } = render(<AntDLoadingSpinner size="large" />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveClass("h-6", "w-6");
  });

  it("should use small size class when size=small", () => {
    const { container } = render(<AntDLoadingSpinner size="small" />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveClass("h-3", "w-3");
  });

  it("should default to size 4 when no size prop is provided", () => {
    const { container } = render(<AntDLoadingSpinner />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveClass("h-4", "w-4");
  });

  it("should apply custom fontSize to the svg style", () => {
    const { container } = render(<AntDLoadingSpinner fontSize={32} />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveStyle({ fontSize: "32px" });
  });

  it("should not set an inline fontSize when not provided", () => {
    const { container } = render(<AntDLoadingSpinner />);
    const svg = container.querySelector("svg") as SVGElement;
    expect(svg.style.fontSize).toBe("");
  });
});
