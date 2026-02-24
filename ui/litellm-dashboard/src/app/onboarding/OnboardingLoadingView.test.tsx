import React from "react";
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OnboardingLoadingView } from "./OnboardingLoadingView";

describe("OnboardingLoadingView", () => {
  it("renders a spinner container", () => {
    const { container } = render(<OnboardingLoadingView />);
    expect(container.firstChild).toBeInTheDocument();
  });

  it("applies centering layout classes", () => {
    const { container } = render(<OnboardingLoadingView />);
    expect(container.firstChild).toHaveClass("flex", "justify-center");
  });
});
