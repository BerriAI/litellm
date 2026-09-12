import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OnboardingLoadingView } from "./OnboardingLoadingView";

describe("OnboardingLoadingView", () => {
  it("should expose the loading state to assistive technology", () => {
    render(<OnboardingLoadingView />);
    expect(screen.getByRole("status", { name: "Loading invitation" })).toBeInTheDocument();
  });

  it("should apply centering layout classes", () => {
    const { container } = render(<OnboardingLoadingView />);
    expect(container.firstChild).toHaveClass("flex", "justify-center");
  });
});
