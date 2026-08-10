import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { OnboardingErrorView } from "./OnboardingErrorView";

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources.en.auth) ?? key,
    }),
  };
});

describe("OnboardingErrorView", () => {
  it("should show the failed to load invitation message", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByText("Failed to load invitation")).toBeInTheDocument();
  });

  it("should show the expiry description", () => {
    render(<OnboardingErrorView />);
    expect(screen.getByText("The invitation link may be invalid or expired.")).toBeInTheDocument();
  });

  it("should render a Back to Login link pointing to /ui/login/", () => {
    render(<OnboardingErrorView />);
    // antd Button with href renders as an <a> element
    const link = screen.getByRole("link", { name: "Back to Login" });
    expect(link).toHaveAttribute("href", "/ui/login/");
  });
});
