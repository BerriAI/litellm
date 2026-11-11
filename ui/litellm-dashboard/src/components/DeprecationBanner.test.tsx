import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DeprecationBanner } from "./DeprecationBanner";

describe("DeprecationBanner", () => {
  it("names the deprecated feature", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    expect(screen.getByRole("alert")).toHaveTextContent("Workflows");
  });

  it("links to the deprecation discussion in a new tab without leaking the opener", () => {
    render(<DeprecationBanner featureName="Workflows" />);

    const link = screen.getByRole("link", { name: "deprecation discussion" });
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32090");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });
});
