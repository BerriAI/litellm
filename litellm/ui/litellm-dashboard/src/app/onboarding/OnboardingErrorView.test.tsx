import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

const { mockGetProxyBaseUrl } = vi.hoisted(() => ({ mockGetProxyBaseUrl: vi.fn(() => "") }));
vi.mock("@/components/networking", () => ({ getProxyBaseUrl: mockGetProxyBaseUrl }));

import { OnboardingErrorView } from "./OnboardingErrorView";

describe("OnboardingErrorView", () => {
  it("should show the failed to load invitation message", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByText("Failed to load invitation")).toBeInTheDocument();
  });

  it("should expose the failure as an alert to assistive technology", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByRole("alert")).toHaveTextContent("Failed to load invitation");
  });

  it("should show the expiry description", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByText("The invitation link may be invalid or expired.")).toBeInTheDocument();
  });

  it("should render a Back to Login link pointing to /ui/login/", () => {
    render(<OnboardingErrorView />);
    const link = screen.getByRole("link", { name: "Back to Login" });
    expect(link).toHaveAttribute("href", "/ui/login/");
  });

  it("keeps the server root path in the login link", () => {
    mockGetProxyBaseUrl.mockReturnValueOnce("http://proxy.example/9000");
    render(<OnboardingErrorView />);
    expect(screen.getByRole("link", { name: "Back to Login" })).toHaveAttribute(
      "href",
      "http://proxy.example/9000/ui/login/",
    );
  });
});
