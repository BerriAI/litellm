import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OnboardingErrorView } from "./OnboardingErrorView";

describe("OnboardingErrorView", () => {
  it("shows the failed to load invitation message", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByText("Failed to load invitation")).toBeInTheDocument();
  });

  it("shows the expiry description", () => {
    render(<OnboardingErrorView />);
    expect(
      screen.getByText("The invitation link may be invalid or expired.")
    ).toBeInTheDocument();
  });

  it("renders a Back to Login link pointing to /ui/login", () => {
    render(<OnboardingErrorView />);
    // antd Button with href renders as an <a> element
    const link = screen.getByRole("link", { name: "Back to Login" });
    expect(link).toHaveAttribute("href", "/ui/login");
  });
});
